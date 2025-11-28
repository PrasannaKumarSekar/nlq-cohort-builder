"""
Basic version of NLQ Cohort Builder implemented as a LangGraph ReAct agent with in-memory checkpointing.
# `pip install -U langgraph` or `pip install -qU "langchain[openai]"`
"""

from rich import print
from typing import List, Dict, Any, Literal, Tuple, Optional, Union, Type, Annotated
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import numpy as np
import pandas as pd

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import os, json
from datetime import datetime
import uuid

from polly.auth import Polly
from polly.atlas import Atlas

from default_system_prompt import COHORT_BUILDER_SYSTEM_PROMPT
from tool_functions import *

from openai import OpenAI

# ----- OpenAI access -----
#os.environ["OPENAI_API_KEY"] = "sk-proj-"  # set openai api key as env variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ----- Polly setup -----
AUTH_KEY = os.getenv("POLLY_AUTH_KEY", "")
ATLAS_ID = os.getenv("POLLY_ATLAS_ID", "beataml2")

if not AUTH_KEY:
    raise ValueError("Missing Polly Auth Key. Please set POLLY_AUTH_KEY environment variable.")

print(f"\nAuthenticating with Polly and connecting to Atlas: {ATLAS_ID}")
Polly.auth(AUTH_KEY, env="polly")
atlas = Atlas(atlas_id=ATLAS_ID)

# ----- initialize llm -----
llm = init_chat_model("openai:gpt-4.1-mini", temperature=0.0)

# ----- Load schema json file and print a short summary of the tables present. -----
BEATAML_DATA_DIR = os.getenv("BEATAML_DATA_DIR", "../../BeatAML") # this is the local dir containing metadata files created at DB ingestion time
os.environ["BEATAML_DATA_DIR"] = BEATAML_DATA_DIR  # Ensure it's set for tools

with open(os.path.join(BEATAML_DATA_DIR, 'BeatAML_schema.json'), 'r') as f:
        DB_SCHEMA = json.load(f)
table_descriptions = {}
for table in DB_SCHEMA.keys():
    table_descriptions[table] = DB_SCHEMA[table]['table_description'].split('.')[0]
print('\nTables Present in DB:\n', json.dumps(table_descriptions, indent=2))

# ----- Get current date ----- 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# ----- load the system prompt -----
system_prompt = COHORT_BUILDER_SYSTEM_PROMPT.format(current_date=current_date, 
                    table_descriptions=json.dumps(table_descriptions,indent=2))

# ----- initialize agent ----- 
# define the State object
class State(TypedDict):
    messages: Annotated[list, add_messages]

# initialize langgraph
graph_builder = StateGraph(State)

# add tools
tools = [inspect_table, inspect_field, 
         get_relevant_database_fields, get_relevant_field_values, 
         transform_query_to_sql, run_sql_query,
         execute_analysis_code, save_process_summary]
llm_with_tools = llm.bind_tools(tools)

# interfacing agent
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}

# add nodes and edges to graph
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# checkpointing
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

'''
## Equivalent ReAct agent template
checkpointer = InMemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[inspect_table, inspect_field, get_relevant_database_fields, get_relevant_field_values],
    prompt = SYSTEM_PROMPT,
    checkpointer=checkpointer  
)
'''

# --- Session Setup ---
session_id = str(uuid.uuid4())[:8]
artifacts_dir = f"./artifacts/{session_id}"
os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs(f"{artifacts_dir}/plots", exist_ok=True)

print(f"\n--- Session Initialized ---")
print(f"Session ID: {session_id}")
print(f"Artifacts Directory: {artifacts_dir}")
print(f"Current Date: {current_date}")

# Set env var for tools to use
os.environ["SESSION_ARTIFACTS_DIR"] = artifacts_dir

# ----- Function to log messages to a file ----- 
log_file = f"{artifacts_dir}/log_{now.strftime('%Y-%m-%d_%H:%M:%S')}.txt"
def log_message(role, content, metadata=None):
    """Append messages (and optional metadata) to log file with timestamp."""
    with open(log_file, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] {role.upper()}:\n{content}\n")
        if metadata:
            f.write("  └─ Metadata: " + json.dumps(metadata, indent=2) + "\n")
        f.write("\n" + "-" * 80 + "\n\n")

# ----- Query and run agent ----- 
config = {"configurable": {"thread_id": session_id}}

print("Start by providing a query. Type 'quit' to exit at any point.")
start = True
while True:
    user_input = input("\nUser > ")
    if user_input.lower() in ["quit", "exit"]:
        print("\nExiting... Bye!\n")
        break

    if start is True:
        msgs = [{"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_input}]
        start = False
    else:
        msgs = [{"role": "user", "content": user_input}]
    events = graph.stream(
        {"messages": msgs},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            msg = event["messages"][-1]
            if not isinstance(msg, HumanMessage) and not isinstance(msg, ToolMessage):
                msg.pretty_print() # not printing user or tool messages
            
            # Determine message role based on type
            if isinstance(msg, HumanMessage):
                role = "Human"
            elif isinstance(msg, AIMessage):
                role = "AI"
            elif isinstance(msg, SystemMessage):
                role = "System"
            else:
                role = type(msg).__name__

            # Extract token usage if available
            token_usage = getattr(msg, "response_metadata", {}).get("token_usage", None)
            if token_usage:
                print(f"\nTotal tokens used: {token_usage.get('total_tokens', 'NA')} (prompt tokens = {token_usage.get('prompt_tokens', 'NA')}, completion tokens = {token_usage.get('completion_tokens', 'NA')})")

            # Log message + metadata
            log_message(role, msg.content, metadata=getattr(msg, "response_metadata", None))
