#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: First version of NLQ Cohort builder implemented in LangGraph with in-memory checkpointing.
# pip install -U langgraph or pip install -qU "langchain[openai]"
"""

# import libraries
from openai import OpenAI
from dotenv import load_dotenv
import os

from datetime import datetime
import json
from typing import TypedDict, List, Dict, Any, Literal, Tuple, Optional
from rich import print
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import re, ast
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


# OpenAI API Key
os.environ['OPENAI_API_KEY'] = ""
client = OpenAI()

# -------------------------
# Helper function to call LLM
# -------------------------

# ---------- helper parser / cleaner ----------
def _extract_balanced_json(text: str) -> Optional[str]:
    """
    Find the first balanced JSON substring (starting at first '{' or '[').
    This scanner:
     - respects string quoting and escape sequences,
     - ignores extraneous closing braces/brackets (skips them if stack empty),
     - if the substring ends with missing closers, will append the needed closers to balance.
    Returns the substring (possibly with appended closers), or None if no JSON start found.
    """
    if not text:
        return None

    # find first JSON opener
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    stack = []
    in_string = False
    escape = False
    end_index = None

    open_to_close = {"{": "}", "[": "]"}

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"' :
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch in ("{", "["):
            stack.append(ch)
            continue

        if ch in ("}", "]"):
            # if stack empty => extraneous closing brace; skip it
            if not stack:
                # skip extraneous closing char
                continue
            # if top matches, pop; else skip the mismatched closing
            top = stack[-1]
            expected = open_to_close.get(top)
            if ch == expected:
                stack.pop()
                if not stack:
                    end_index = i + 1
                    break
            else:
                # mismatched closing: skip
                continue

    # if we never closed all opens, append needed closers
    substr = text[start : end_index if end_index is not None else len(text)]
    if stack:
        # append the required closing brackets in reverse order
        closers = "".join(open_to_close[o] for o in reversed(stack))
        substr = substr + closers

    return substr

def _sanitize_braces(text: str) -> str:
    """
    Removes extraneous unmatched closing braces/brackets from text.
    Keeps JSON balanced enough for json.loads.
    """
    result = []
    stack = []
    open_to_close = {"{": "}", "[": "]"}
    close_to_open = {"}": "{", "]": "["}

    in_string = False
    escape = False

    for ch in text:
        if escape:
            result.append(ch)
            escape = False
            continue

        if ch == "\\":
            result.append(ch)
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if in_string:
            result.append(ch)
            continue

        if ch in "{[":
            stack.append(ch)
            result.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == close_to_open[ch]:
                stack.pop()
                result.append(ch)
            else:
                # skip extraneous closing
                continue
        else:
            result.append(ch)

    # add missing closers
    while stack:
        opener = stack.pop()
        result.append(open_to_close[opener])

    return "".join(result)


def _safe_parse_json(text: str) -> Optional[Any]:
    """Try multiple strategies to parse JSON-ish strings returned by LLMs."""
    if not isinstance(text, str):
        return None

    # 1) direct json
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) lightweight cleaning
    cleaned = text.strip()
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3) sanitize braces/brackets and try again
    try:
        sanitized = _sanitize_braces(cleaned)
        return json.loads(sanitized)
    except Exception:
        pass

    # 4) try extracting substring
    try:
        candidate = _extract_balanced_json(text)
        if candidate:
            return json.loads(candidate)
    except Exception:
        pass

    # 5) fallback to ast.literal_eval
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    return None


# ---------- call_llm with optional enforce_json ----------
def call_llm(
    prompt: str,
    examples: List[Dict[str, Any]] = None,
    enforce_json: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    max_repair_attempts: int = 2,
) -> Any:
    """
    Call LLM with a prompt and optional examples.

    If enforce_json is False -> returns a str (raw assistant text).
    If enforce_json is True -> returns a parsed Python object (dict/list).
      - If parsing fails, it will attempt a single repair call to the LLM and re-parse.
      - If still unparsable, raises ValueError with the raw LLM output attached.

    If json_schema is provided, the function will attempt to pass a `response_format`
    parameter to the API (Structured Outputs) if supported by the client. If that call
    fails, we fall back to prompting + parse+repair.
    """
    messages = [{"role": "system", "content": "You are a biomedical expert agent."}]
    # examples are optional
    if examples:
        for ex in examples:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": prompt})

    # If enforce_json, give the model explicit instruction to only output valid JSON.
    if enforce_json:
        # Add an extra system-level instruction so model is primed for JSON-only output.
        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "IMPORTANT: When you respond, return ONLY valid JSON (no explanation, no markdown, "
                    "no code fences). The response must be strict JSON that can be parsed with json.loads()."
                ),
            },
        )

    # Try to use Structured Outputs / response_format when a json_schema is provided.
    api_kwargs = {}
    if json_schema is not None:
        # This may be supported by newer SDKs / models (Structured Outputs).
        # We attempt to set a response_format; if the client rejects it, we'll fall back.
        api_kwargs["response_format"] = {
            "type": "json_object",
            "json_schema": json_schema,
            "strict": True,
        }

    # 1) Make the API call (try with structured outputs if requested)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0 if enforce_json else 0.001,
            **api_kwargs,
        )
    except Exception:
        # If response_format isn't supported by the library, retry without api_kwargs
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0 if enforce_json else 0.001,
        )

    raw = resp.choices[0].message.content.strip()

    # If JSON enforcement not requested, return raw string (backwards compatible)
    if not enforce_json:
        return raw

    # 2) Try to parse
    parsed = _safe_parse_json(raw)
    if parsed is not None:
        return parsed

    # 3) If parse failed, attempt a repair pass (ask the model to fix broken JSON)
    repair_prompt = (
        "The following text should be STRICT JSON but may contain minor formatting errors. "
        "Please fix it and return only valid JSON (no explanation, no quotes around the JSON, no markdown):\n\n"
        + raw
    )

    for attempt in range(max_repair_attempts):
        repair_resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON fixer. Return only corrected JSON."},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=0.0,
        )
        fixed_raw = repair_resp.choices[0].message.content.strip()
        parsed = _safe_parse_json(fixed_raw)
        if parsed is not None:
            return parsed
        # else: loop and try again (up to max_repair_attempts)

    # 4) If still failing — raise with the raw content for debugging
    raise ValueError(f"LLM returned unparsable JSON (after repair attempts). Raw:\n{raw}")


# -------------------------
# Function 1: extract_raw_criteria
# -------------------------
def extract_raw_criteria(query: str, current_criteria: Dict[str, str] = {}, feedback: str = "") -> List[Dict[str, str]]:
    """
    Breaks down a query into a list of inclusion/exclusion criteria.

    Args:
        query (str): The user's original query, assumed to have COHORT_SELECTION intent.
        current_criteria: relevant if using this function to update existing criteria with user feedback
        feedback: additional user input besides the original query (corrections, clarifications)

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each representing a single
                               granular condition.
    """
    
    prompt = f"""
    TASK:
    Break down the query into a list of granular logical criteria for building a cohort.
    
    APPROACH:
    - Focus on the part of the query that is relevant to cohort building.
    - Ignore parts of the query asking for analysis, plotting, or specific attributes. 
    - Each distinct `AND` condition should be a separate item in the list.
    - Do not split `OR` clauses.
    - Handle exclusive conditions by adding an exclusion criterion over the Complement Set.
    - Remove redundancy, if any.

    IMPORTANT: 
    * Pay attention to the [current_criteria] supplied and the [feedback], if any, as well as the original query.
    * If current_criteria is not null, use feedback to modify/update current_criteria only where needed, leave rest unchanged.
    
    Return the result as a JSON list of objects. Each object is a single `AND` condition and has:
    - "type": "include" or "exclude"
    - "text": The string phrase for the condition.
    Always Return only a JSON list and nothing else. Do not include markdown code blocks.

    <Examples>
    Query: "Find all women who have diabetes or hypertension but not smokers and are older than 50."
    Result:
    [
        {{"type": "include", "text": "are women"}},
        {{"type": "include", "text": "have diabetes or hypertension"}},
        {{"type": "include", "text": "are older than 50"}},
        {{"type": "exclude", "text": "are smokers"}}
    ]
    Query: "male diabetic pts on warfarin and no other drug"
    Result:
    [
        {{"type": "include", "text": "are male"}},
        {{"type": "include", "text": "have diabetes"}},
        {{"type": "include", "text": "on warfarin"}},
        {{"type": "exclude", "text": "on any other drug besides warfarin"}}  # added exclusion over complement
    ]
    </Examples>

    Original User Query: {query}
    Current Criteria: {json.dumps(current_criteria, indent=2)}
    User Feedback: {feedback}

    Result (strict JSON):
    """
    response = call_llm(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Error: LLM did not return valid JSON for raw criteria extraction.")
        return response
    
# -------------------------
# Function 2: extract_criteria_entities
# -------------------------
def extract_criteria_entities(criteria_list: List[Dict[str, str]], max_workers: int = 8) -> List[Dict[str, Any]]:
    """
    Extracts key entities from the 'text' of each criterion.

    Args:
        criteria_list (List[Dict[str, str]]): The output from extract_raw_criteria.
        max_workers (int): No. of workers.

    Returns:
        List[Dict[str, Any]]: The updated list of dicts, each now containing
                               an 'entities' key with a list of extracted terms.
    """
    
    def process_criterion(criterion: Dict[str, str]) -> Dict[str, Any]:
        prompt = f"""
            You are given criterion text. Extract all specific entities present, to create a cohort.
            Specific Entities can be nouns/noun phrases, names, groups, identifiers/codes, numbers, 
            ranges, durations, measurements.
            Ignore non-specific category or property terms.
            Always Return a JSON list and nothing else. Do not include markdown code blocks.

            <Examples>
                - Criterion: "have diabetes or hypertension"
                - Result: ["diabetes", "hypertension"]
                
                - Criterion: "are women"
                - Result: ["female"]

                - Criterion: "are older than 50"
                - Result: ["age > 50"]

                - Criterion Text: "born between 1990 and 1997"
                - Result: ["1990-1997"]

                - Criterion Text: "diagnosis of melanoma"
                - Result: ["melanoma"]
            </Examples>

            Input Criterion: "{criterion['text']}"
            Result (strict JSON):
        """
        response = call_llm(prompt)
        new_criterion = criterion.copy()
        try:
            new_criterion['entities'] = json.loads(response)
        except json.JSONDecodeError:
            print(f"Error: Could not parse entities from text: '{criterion['text']}'")
            new_criterion['entities'] = []
        return new_criterion

    updated_criteria = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Launch tasks
        futures = [executor.submit(process_criterion, c) for c in criteria_list]
        for future in as_completed(futures):
            updated_criteria.append(future.result())
    
    # Preserve input order
    updated_criteria.sort(key=lambda x: criteria_list.index(next(c for c in criteria_list if c['text'] == x['text'])))

    return updated_criteria


# --------------------------------------------------------------
# Major changes start from here 
# --------------------------------------------------------------


# ---------------------------
# Define Agent State
# ---------------------------
class AgentState(TypedDict):
    """
    Holds everything the agent and tools need.
    """
    original_query: str
    conversation_history: List[str]
    user_input: str
    current_criteria: Dict[str, Any]
    stage_counter: int
    agent: Dict[str, Any]   # {"thinking":..., "action":..., "question":...}

# ---------------------------
# Agent Node (Decisions)
# ---------------------------
def react_agent(state: AgentState) -> AgentState:
    """
    LLM decides the next action.
    """
    prompt = f"""
        You're an assistant helping user build inclusion/exclusion criteria in a multi-turn conversation.

        Analyze the recent conversation and current state, then decide your next action.

        <Actions>
        - `advance`: move forward if user has posed new query or has approved current result
        - `edit`: if user has provided feedback to update or modify current criteria
        - `clarify`: if user input is unclear (acronyms, too broad entities, vague attributes), ask back a clarifying question
        - `reject`: stop process (reject irrelevant input)
        </Actions>

        Conversation so far:
        {state['conversation_history']}
        User just said:
        {state['user_input']}

        Current criteria: {state['current_criteria']}
        Stage: {state['stage_counter']}

        Always Return JSON: 
        {{ 
            "thinking": "...", # be concise
            "action": <advance|edit|clarify|reject>, # required
            "question": "...", # optional; required only if action = `clarify`
        }}
        Response (strict JSON):
    """
    llm_response = call_llm(prompt)
    # update state with action outcome
    state["agent"] = json.loads(llm_response)
    state["conversation_history"][-1]["Agent"] = state['agent'].get('action')
    print(f"\nAgent action: {state['agent'].get('action')}")
    
    if state['agent']['action']=='clarify' and 'question' in state["agent"]:
        state["conversation_history"][-1]["Question"] = state['agent'].get('question')
        print(f"\nQuestion: {state['agent'].get('question')}")
    return state


# ---------------------------
# Tool Nodes
# ---------------------------
def process_query(state: AgentState) -> AgentState:
    """
    Handles 'advance': run next processing step
    """
    if state["stage_counter"] == 0:
        print(f"> Agent to User: Generating eligibility criteria")
        user_input = state["conversation_history"][-1]["User"]
        if user_input != state["original_query"]:
            feedback = user_input
        else:
            feedback = ""
        result = extract_raw_criteria(
            state["original_query"],
            state["current_criteria"],
            feedback,
        )
        state["current_criteria"] = result
        state["stage_counter"] += 1
    elif state["stage_counter"] == 1:
        print(f"> Agent to User: Entity extraction from criteria")
        result = extract_criteria_entities(state["current_criteria"])
        state["current_criteria"] = result
        state["stage_counter"] += 1

    print(f"\nCurrent Criteria:\n{json.dumps(state['current_criteria'], indent=2)}\n")
    return state

def edit_with_feedback(state: AgentState) -> AgentState:
    """
    Handles 'edit': apply user feedback to criteria.
    """
    recent_feedback = '\n'.join([f'- {s["User"]}' for s in state["conversation_history"][-2:]])

    if state["stage_counter"] == 1:
        result = extract_raw_criteria(
            state["original_query"],
            state["current_criteria"],
            recent_feedback, # only the last 2 feedbacks used, earlier clarifications ignored
        )
        state["current_criteria"] = result
    else:
        edit_prompt = f"""
            Instructions (think step-by-step):
            - Examine the state JSON and the user edit request.
            - Identify the target object for the edit.
            - Decide the minimal change needed to implement user edit in the object.
            - Proceed to make the edit in the precise place(s) while preserving overall structure and consistency.
            - Return the updated JSON: the orignal state JSON with the minimal edit made.

            Current State: {state["current_criteria"]}
            User Input: {recent_feedback}
            Response (strict JSON):
        """
        result = call_llm(edit_prompt, enforce_json=True)
        state["current_criteria"] = result
    print(f"\nCurrent Criteria:\n{json.dumps(state['current_criteria'], indent=2)}\n")
    return state

# ---------------------------
# User Node
# ---------------------------
def user_input(state: AgentState) -> Tuple[AgentState, Command[Literal["agent", END]]]:
    """
    Handles 'user': user inputs which update state object.
    If user types 'quit', state is updated and flow ends.
    """
    user_in = input("User > ")
    
    # Always update state first
    if not state["original_query"]:
        state["original_query"] = user_in
    state["user_input"] = user_in
    state["conversation_history"].append({"User": user_in})

    # If quit, stop graph flow
    if user_in.lower().strip() in ["quit", "exit"]:
        return state, Command(goto=END)

    # Otherwise, continue to agent node
    return state, Command(goto="agent")


# ---------------------------
# Router
# ---------------------------
def route_action(state: AgentState) -> str:
    """
    Decide which tool node to go to, based on agent output.
    """
    if state["stage_counter"] == 2 and state["agent"]["action"] == "advance":
        return "stop"
    return state["agent"].get("action", "stop")


# ---------------------------
# Build LangGraph
# ---------------------------
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", react_agent)
workflow.add_node("advance", process_query)
workflow.add_node("edit", edit_with_feedback)
workflow.add_node("user", user_input)

# Add edge from START to user
workflow.add_edge(START, "user")

# Conditional edges out of agent
workflow.add_conditional_edges(
    "agent", route_action,
    {
        "advance": "advance",
        "edit": "edit",
        "clarify": "user",
        "reject": END,
        "stop": END
    }
)

# After each tool, go back to user input; user input flows to agent node
workflow.add_edge("advance", "user")
workflow.add_edge("edit", "user")

# Start and End
#workflow.set_entry_point("user")
#workflow.set_finish_point(END) # don't need this

checkpointer = InMemorySaver()  # session state held in-memory, can be saved to persistent storage as well  
graph = workflow.compile(checkpointer)

# ---------------------------
# 7. Run Multi-Turn Loop
# ---------------------------
if __name__ == "__main__":

    # Initialize state object and assign a thread_id
    initial_state: AgentState = {
        "original_query": None,
        "conversation_history": [],
        "user_input": "",
        "current_criteria": {},
        "stage_counter": 0,
        "agent": {}
    }
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    print("\n--- Session Initialized ---")
    print(f"session_id = {session_id}. Type 'quit' to exit.\n")

    graph.update_state(config, initial_state) # copy initial state to graph
    
    # run the loop, it runs until user quits or state reaches END node
    graph.invoke({}, config=config)
  
    print("\n------ Exited Loop ------")

    # last state chkpt
    snapshot = graph.get_state(config)
    state_dict = snapshot.values  
    print(f"\nFinal criteria:\n{json.dumps(state_dict['current_criteria'], indent=2)}")

    # If you also want metadata (timestamps etc.), convert to strings
    serializable_snapshot = {
        "values": state_dict,
        "config": snapshot.config,
        "metadata": snapshot.metadata,
        "created_at": snapshot.created_at.isoformat() if isinstance(snapshot.created_at, datetime) else snapshot.created_at,
    }
    # save last state to file
    os.makedirs("langgraph_cb_runs", exist_ok=True)
    with open(f"langgraph_cb_runs/v1_agent_session_{session_id}.json", "w") as f:
        json.dump(serializable_snapshot, f, indent=2)

    #print(list(graph.get_state_history(config))) # to get the full session history
    
