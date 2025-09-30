#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Simple workflow for cohort criteria generation from a user query + feedback with the help of LLM 
"""

# import libraries
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime
import json
import ast
from typing import List, Dict, Any, Literal, Tuple, Optional, Union, Type, TypedDict
from pydantic import BaseModel, Field
from rich import print
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
from sqlalchemy import Table, Column, Integer, String, MetaData, select, or_, and_, not_, case, func
from sqlalchemy.sql import text
import collections
from collections import deque
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

# OpenAI API Key
os.environ['OPENAI_API_KEY'] = "your_api_key"
client = OpenAI()
#load_dotenv()
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# Helper function to call LLM
# -------------------------

def call_llm(
    user_prompt: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    system_prompt: str = "You are a biomedical expert agent.",
    response_model: Optional[Type[BaseModel]] = None,
) -> BaseModel | str:
    """
    Generalized function to call LLM with flexible settings and structured output.

    Args:
        user_prompt (str): The main input prompt for the model.
        model (str): The LLM model to use (default: gpt-4.1-mini).
        temperature (float): Sampling temperature (default: 0.0).
        system_prompt (str): The system-level instruction for the agent.
        response_model (Optional[Type[BaseModel]]): Pydantic model class for enforcing structured output.

    Returns:
        BaseModel | str: A validated Pydantic model instance if response_model is provided, otherwise raw text output.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = {
        "model": model,
        "input": messages,
        "temperature": temperature,
    }

    if response_model:
        kwargs["text_format"] = response_model

    response = client.responses.parse(**kwargs)

    return response.output_parsed if response_model else response.output_text


# -------------------------
# Function 1: extract_raw_criteria
# -------------------------
class Criterion(BaseModel):
    type: Literal["include", "exclude"]
    text: str

class CriteriaList(BaseModel):
    criteria: List[Criterion]


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
    Focus on the part of the query that is relevant to cohort building.
    Ignore parts of the query asking for analysis, plotting, or specific attributes. 
    Each distinct `AND` condition should be a separate item in the list.
    Do not split `OR` clauses.

    IMPORTANT: 
    * Pay attention to the [current_criteria] supplied and the [feedback], if any, as well as the original query.
    * If current_criteria is not null, use feedback to modify/update current_criteria only where needed, leave rest unchanged.
    
    Return the result as a JSON list of objects. Each object is a single `AND` condition and has:
    - "type": "include" or "exclude"
    - "text": The string phrase for the condition.
    Always Return only a JSON list and nothing else. Do not include markdown code blocks.

    <Example>
    Query: "Find all women who have diabetes or hypertension but not smokers and are older than 50."
    Result:
    [
        {{"type": "include", "text": "are women"}},
        {{"type": "include", "text": "have diabetes or hypertension"}},
        {{"type": "include", "text": "are older than 50"}},
        {{"type": "exclude", "text": "are smokers"}}
    ]
    </Example>

    Original User Query: {query}
    Current Criteria: {json.dumps(current_criteria, indent=2)}
    User Feedback: {feedback}

    Result (strict JSON):
    """
    
    # Call LLM with enforced response model
    response = call_llm(user_prompt=prompt, response_model=CriteriaList)

    if isinstance(response, CriteriaList):
        # Extract list of Criterion objects into plain dicts
        return [c.model_dump() for c in response.criteria]
    else:
        print("Error: LLM did not return valid structured criteria.")
        return []

# -------------------------
# Function 2: extract_criteria_entities
# -------------------------

class EntitiesList(BaseModel):
    entities: List[str]

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
            You are given criterion text. Extract all specific entities present, for cohort building.
            Specific Entities can be nouns, noun phrases, names, groups, identifiers, codes, ranges, or numbers.
            Return the result as a JSON object with a key "entities".
            Always Return only JSON and nothing else. Do not include markdown code blocks.

            <Examples>
                - Criterion: "have diabetes or hypertension"
                - Result: {{"entities": ["diabetes", "hypertension"]}}
                
                - Criterion: "are women"
                - Result: {{"entities": ["female"]}}

                - Criterion: "are older than 50"
                - Result: {{"entities": ["age > 50"]}}

                - Criterion Text: "born between 1990 and 1997"
                - Result: {{"entities": ["1990-1997"]}}

                - Criterion Text: "diagnosis of melanoma"
                - Result: {{"entities": ["melanoma"]}}
            </Examples>

            Input Criterion: "{criterion['text']}"
            Result (strict JSON):
        """

        response = call_llm(user_prompt=prompt, response_model=EntitiesList)
        new_criterion = criterion.copy()
        if isinstance(response, EntitiesList):
            new_criterion['entities'] = response.entities
        else:
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

# -------------------------
# Function 3: map_criteria_to_schema
# -------------------------

## Utils
def get_embedding(text: str, model="text-embedding-3-small") -> List[float]:
    """ 
    Embed a query text string (entity or phrase) using an OpenAI model
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def find_matches(
    query_embedding: List[float],
    embeddings: Dict[str, Dict[str, List[float]]],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Return top_k most similar table.field entries to query_embedding
    using sklearn's cosine_similarity.
    """
    keys = list(embeddings.keys())
    emb_matrix = np.array([v['embedding'] for v in embeddings.values()])  # shape (n, d)
    query_vec = np.array(query_embedding).reshape(1, -1)  # shape (1, d)

    sims = cosine_similarity(query_vec, emb_matrix)[0]  # shape (n,)

    # Get top_k indices
    top_idx = np.argsort(sims)[::-1][:top_k]

    return [(keys[i], sims[i]) for i in top_idx]


class Ranking(BaseModel):
    rank: int = Field(..., description="Rank position, starting from 1")
    candidate: str = Field(..., description="Candidate name")

class RerankResult(BaseModel):
    rankings: List[Ranking] = Field(..., description="List of ranked candidates")


def rerank_with_llm(
    query: str, 
    candidates: List[Tuple[str, str]], 
    context: str = ""
) -> Dict[str, str]:
    """
    Rerank candidates using an LLM + structured Pydantic output.
    Each candidate is (name, text).
    Factors in the context (if provided). 
    Returns dict of candidate names ranked { "1": ..., "2": ..., ... }.
    """

    candidates_list = [{"name": n, "text": t} for n, t in candidates]

    prompt = f"""
    TASK:
    Rank all candidates by relevance to the query.
    Factor in the context if provided.

    INPUT:
    Query: {query}
    Context: {context}

    Candidates:
    {json.dumps(candidates_list, indent=2)}

    OUTPUT REQUIREMENTS:
    Return ONLY strict JSON, matching this schema:

    {{
        "rankings": [
            {{ "rank": 1, "candidate": "<best candidate name>" }},
            {{ "rank": 2, "candidate": "<second best candidate name>" }},
            {{ "rank": 3, "candidate": "<third best candidate name>" }}
        ]
    }}
    """

    response = call_llm(user_prompt=prompt, model='gpt-4o-mini', system_prompt='You are a ranking assistant', response_model=RerankResult)

    if isinstance(response, RerankResult):
        # Convert list back into dict { "1": candidate, ... } if needed
        return {str(r.rank): r.candidate for r in response.rankings}
    else:
        print("Error: LLM did not return valid rankings.")
        return {}


#------------------------------
class EntityClassResponse(BaseModel):
    """LLM response for entity classification."""
    result: str = Field(..., description="Biomedical category tag for the entity")

class TableMappingResponse(BaseModel):
    """LLM response for mapping entity to most probable table."""
    table: str = Field(..., description="Most relevant table name")

class FieldMappingResponse(BaseModel):
    """LLM response for mapping entity to most probable field in a given table."""
    field: str = Field(..., description="Most relevant field name")

def map_entity_to_table_field(entity: str, context_text: str, 
                              schema: Dict, schema_embeddings: Dict, method: str = 'sequential') -> Dict[str, str]:
    """Helper function to map a single entity to the most likely table and field.
    
        Args:
            entity (str): entity string
            context_text (str): the logical condition from which the entity was parsed
            schema (Dict): JSON of the DB schema 
            schema_embeddings (Dict): dict mapping table.field names to descriptions and text embeddings
            method (str): how the mapping from entity to table.field is to be done

        Returns:
            Dict[str, Dict]: entity mapped to a dict with the table, field and ranked matches

       `method` options:
        - 'sequential': map entity to likely table, then to likely column in the chosen table using an LLM;
          then retrieve similar fields with a similarity search on text summary embeddings and finally 
          rerank top_k matches 
        - 'embed_rerank': embed entity + class, do similarity search followed by reranking of top_k matches 
          against field text embeddings
    """

    entity_class = call_llm(
        user_prompt=f"""
            Assign a biomedical category to the entity: "{entity}", given added context "{context_text}".
            Always Return only the tag. 
            Result (str):""",
        response_model=EntityClassResponse
    ).result
        
    if method == 'embed_rerank':
        # Append inferred class to entity, embed the combination, then map to most relevant field.
        entity_emb = get_embedding(f'{entity}_{entity_class}')  # embed the entity + class string
        # Step 1: kNN matches
        top_matches = find_matches(entity_emb, schema_embeddings, top_k=5)
        candidates = [(name, schema_embeddings[name]['text']) for name, _ in top_matches]
        # Step 2: LLM-reranking of top_k matches, make use of the criterion text as added context
        ranked_matches = rerank_with_llm(entity_emb, candidates, context_text)
        if ranked_matches:
            try:
                mapped_table_field = ranked_matches["1"]
                return {"entity_class": entity_class, "table.field": mapped_table_field, "ranked_matches": list(ranked_matches.values())}
            except Exception as e:
                print(f'Error with LLM-reranking step for entity {entity}: {e}')
                return {"entity_class": entity_class, "table.field": None, "ranked_matches": None}    
        return {"entity_class": entity_class, "table.field": None, "ranked_matches": None}
    
    elif method == 'sequential':
        # Step 1: Map entity to the most probable Table
        table_descriptions = {name: details['table_description'] for name, details in schema.items()}
        mapped_table = call_llm(
            user_prompt=f"""
                Given entity "{entity}" with tag "{entity_class}" from context "{context_text}", 
                map the entity to the most probable Table.
                Respond with only the Table name.

                Tables:
                {json.dumps(table_descriptions, indent=2)}

                Most relevant Table (str):
                """,
            response_model=TableMappingResponse
        ).table

        if mapped_table not in schema:
            return {"entity_class": entity_class, "table.field": None, "ranked_matches": None}

        # Step 2: Map entity to the most probable Field within the selected table
        field_descriptions = schema[mapped_table]['fields']
        mapped_field = call_llm(
            user_prompt=f"""
                Given entity "{entity}" with tag "{entity_class}" from context "{context_text}", 
                map the entity to the most probable Field in "{mapped_table}" Table.
                Respond with only the Field name.

                Fields in "{mapped_table}":
                {json.dumps(field_descriptions, indent=2)}

                Most relevant Field (str):
                """,
            response_model=FieldMappingResponse
        ).field
        
        if mapped_field not in field_descriptions:
            return {"entity_class": entity_class, "table.field": f'{mapped_table}.?', "ranked_matches": None}
            
        # Find other fields similar to the chosen option based on vector embeddings, then rerank (Optional)
        key = f'{mapped_table}.{mapped_field}'
        top_matches = find_matches(schema_embeddings[key]['embedding'], schema_embeddings, top_k=5)
        candidates = [(name, schema_embeddings[name]['text']) for name, _ in top_matches]
        ranked_matches = rerank_with_llm(entity, candidates, context_text)
        if ranked_matches:
            try:
                mapped_table, mapped_field = ranked_matches["1"].split('.')
            except Exception as e:
                print(f'Error with LLM-reranking step for entity {entity}: {e}')
                return {"entity_class": entity_class, "table.field": f'{mapped_table}.{mapped_field}', "ranked_matches": None}
                
        return {"entity_class": entity_class, "table.field": f'{mapped_table}.{mapped_field}', "ranked_matches": list(ranked_matches.values())}


def map_criteria_to_schema(criteria_list: List[Dict[str, Any]],
                           db_schema: Dict,
                           schema_embeddings: Dict,
                           method: str = 'sequential', max_workers: int = 8) -> List[Dict[str, Any]]:
    """
    Maps extracted entities in each criterion to the database schema.

    Args:
        criteria_list (List[Dict[str, Any]]): The output from extract_criteria_entities.
        db_schema (Dict): The database schema.
        method (str): Mapping logic followed (`sequential` or `embedding_reranking`)
        max_workers (int): No. of workers.

    Returns:
        List[Dict[str, Any]]: The updated list of dicts, each now containing
                               a 'db_mappings' key with entity -> table.field assignments.
    """

    def process_entity(idx: int, entity: str, text: str):
        mapping = map_entity_to_table_field(entity, text, db_schema, schema_embeddings, method)
        return idx, entity, mapping

    results = [{} for _ in criteria_list]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_entity, i, entity, crit['text'])
            for i, crit in enumerate(criteria_list)
            if crit.get('entities')
            for entity in crit['entities']
        ]
        for future in as_completed(futures):
            idx, entity, mapping = future.result()
            if mapping['table.field']:
                results[idx][entity] = mapping

    updated_criteria = []
    for i, crit in enumerate(criteria_list):
        new_crit = crit.copy()
        new_crit['db_mappings'] = results[i]
        updated_criteria.append(new_crit)

    return updated_criteria


class ConceptMappingResult(BaseModel):
    concept: Optional[str] = Field(None, description="The best matching concept from candidates or None if no match")
    reason: str = Field(..., description="One short sentence explaining the choice")
    top_5: List[str] = Field(..., description="Ranked list of up to 5 candidate concepts")

def _llm_choose_best_concept(entity: str, table: str, field: str, candidates: List[str], model: str = "gpt-4o-mini"):
    bullet_list = "\n".join(f"- {c}" for c in candidates)
    prompt = f"""
    You are helping map a user-provided entity to the best matching concept for a specific database field.
    
    TASK:
    - Pick exactly one best concept (verbatim from the list) or return null if none match well.
    - Provide a short reason for the choice.
    - Provide a ranked list of up to 5 candidate concepts.

    CONTEXT:
    - Table: {table}
    - Field: {field}
    - Entity to map: "{entity}"

    CANDIDATES:
    {bullet_list}

    OUTPUT FORMAT:
    Return strict JSON only, matching this schema:
    {{
      "concept": "<best concept or null>",
      "reason": "<short sentence>",
      "top_5": ["<concept1>", "<concept2>", ...]
    }}
    """

    try:
        result = call_llm(
            user_prompt=prompt,
            model=model,
            system_prompt="You are a precise data-mapping assistant who chooses the best concept label.",
            response_model=ConceptMappingResult,
        )
        if isinstance(result, ConceptMappingResult):
            return result.concept, result.reason, result.top_5
        else:
            return None, "Invalid response format", []
    except Exception as e:
        return None, f"LLM call failed: {e}", []

def search_concept_for_entity(
    entity,
    table,
    field,
    concept_df,
    concept_lookup,
    top_k=5,
    model_embed="text-embedding-3-small",
    model_llm="gpt-4o-mini"
):
    subset = concept_df[
        (concept_df["table_name"] == table) &
        (concept_df["field_name"] == field)
    ].copy()

    if subset.empty:
        return {
            "concept_name": None,
            "concept_with_context": None,
            "similarity": None,
            "method": "semantic_search",
            "reason": None,
            "top_candidates": []
        }

    subset_unique = subset.drop_duplicates(subset=["concept_name"]).reset_index(drop=True)
    num_unique = len(subset_unique)

    # ---- Small set: LLM decision ----
    if num_unique <= 10:
        candidates = subset_unique["concept_name"].tolist()
        chosen, reason, top_candidates = _llm_choose_best_concept(entity, table, field, candidates, model=model_llm)
        if chosen is None or str(chosen).strip().lower() in ("", "null"):
            return {
                "concept_name": None,
                "concept_with_context": None,
                "similarity": None,
                "method": "LLM_decision",
                "reason": reason,
                "top_candidates": top_candidates
            }
        row = subset_unique[subset_unique["concept_name"].str.lower().str.strip() == chosen.lower().strip()].iloc[0]
        return {
            "concept_name": row["concept_name"],
            "concept_with_context": row["concept_with_context"],
            "similarity": None,
            "method": "LLM_decision",
            "reason": reason,
            "top_candidates": top_candidates
        }

    # ---- Large set: semantic search ----
    subset_ctx = subset_unique["concept_with_context"].tolist()
    subset_embs = np.vstack([concept_lookup[c] for c in subset_ctx])

    query_emb = client.embeddings.create(
        model=model_embed,
        input=entity
    ).data[0].embedding
    query_emb = np.array(query_emb).reshape(1, -1)

    sims = cosine_similarity(query_emb, subset_embs)[0]
    if sims.size == 0:
        return {
            "concept_name": None,
            "concept_with_context": None,
            "similarity": None,
            "method": "semantic_search",
            "reason": None,
            "top_candidates": []
        }

    # Top-k by cosine similarity
    best_idxs = sims.argsort()[::-1][:top_k]
    top_candidates = [subset_unique.iloc[i]["concept_name"] for i in best_idxs]

    best_row = subset_unique.iloc[best_idxs[0]]
    best_score = float(sims[best_idxs[0]])

    return {
        "concept_name": best_row["concept_name"],
        "concept_with_context": best_row["concept_with_context"],
        "similarity": best_score,
        "method": "semantic_search",
        "reason": None,
        "top_candidates": top_candidates
    }


def map_entity_to_concept(criteria_list,
                           concept_df,
                           concept_lookup,
                           max_workers=8):
    """
    Maps entities in each criterion to concepts in the database.
    Updates db_mappings in-place with mapped_concept, mapping_method, and reason.
    """

    def process_entity(entity, table_field, concept_df, concept_lookup):
        if "." not in table_field:
            return entity, None
        table, field = table_field.split(".", 1)

        match = search_concept_for_entity(entity, table, field, concept_df, concept_lookup)
        if match:
            return entity, {
                "mapped_concept": match["concept_name"],
                "mapping_method": match["method"],
                "reason": match["reason"],
                "top_candidates": match["top_candidates"]
            }
        return entity, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for q in criteria_list:
            db_mappings = q.get("db_mappings", {})
            for entity, mapping in db_mappings.items():
                table_field = mapping.get("table.field")
                if table_field:
                    futures.append(
                        executor.submit(process_entity, entity, table_field, concept_df, concept_lookup)
                    )

        enriched_results = {}
        for future in as_completed(futures):
            entity, mapping_update = future.result()
            if mapping_update:
                enriched_results[entity] = mapping_update

    # update in place
    for q in criteria_list:
        db_mappings = q.get("db_mappings", {})
        for entity, mapping in db_mappings.items():
            if entity in enriched_results:
                mapping.update(enriched_results[entity])

    return criteria_list


# -------------------------
# Function 4: rewrite_user_criteria
# -------------------------
class RewrittenCondition(BaseModel):
    result: str = Field(..., description="The rewritten logical condition string using table.field and mapped values")

def rewrite_text_with_mapping(text: str, mapping: dict) -> str:
    """
    Reframe a single condition using the entity mappings to the DB values.
    Uses both 'table.field' and 'mapped_concept' to generate SQL-like conditions.
    Ensures correct quoting for strings vs. numeric types.
    """

    mapping_context = {}
    for entity, m in mapping.items():
        field = m.get("table.field")
        concept = m.get("mapped_concept")
        if field and concept:
            mapping_context[entity] = f"{field} -> {concept}"
        elif field:
            mapping_context[entity] = field

    prompt = f"""
    Rewrite the criterion text into a logical condition string.
    Use BOTH the field name (`table.field`) and the mapped value (`mapped_concept`) 
    when available. Strings should be in single quotes, numerics unquoted.
    Preserve logical operators (`AND`/`OR`/`NOT`) and numeric constraints. 
    Expand out numeric ranges if any.

    <Examples>
        - Criterion Text: "have diabetes or hypertension"
        - Mapped Entities: {{"diabetes": "Donor.comorbidity -> diabetes", "hypertension": "Donor.preexisting_condition -> hypertension"}}
        - Result: "Donor.comorbidity = 'diabetes' OR Donor.preexisting_condition = 'hypertension'"
        --------
        - Criterion Text: "born between 1990 and '97"
        - Mapped Entities: {{"1990-1997": "Donor.year_of_birth -> 1990-1997"}}
        - Result: "Donor.year_of_birth > 1990 AND Donor.year_of_birth < 1997" 
    </Examples>

    CONTEXT:
    Criterion: {text}
    Mapped Entities: {mapping_context}

    OUTPUT FORMAT:
    Return JSON strictly in this schema:
    {{
      "result": "<rewritten logical condition>"
    }}
    """

    response = call_llm(
        user_prompt=prompt,
        response_model=RewrittenCondition
    )

    return response.result if isinstance(response, RewrittenCondition) else ""


def rewrite_user_criteria(criteria_list: List[Dict[str, Any]], max_workers: int = 8) -> List[Dict[str, Any]]:
    """
    Rewrite criteria into structured include/exclude dicts using mapped table.field & mapped_concept context
    while preserving the original query logic.

    Args:
        criteria_list (List[Dict[str, Any]]): The output from map_entity_to_concept.
        max_workers (int): No. of workers.

    Returns:
        List[Dict[str, Any]]: Updated list of criteria, with key `revised_criterion` added to each dict.
    """

    def process_criterion(criterion: Dict[str, Any]) -> Dict[str, Any]:
        new_criterion = criterion.copy()
        new_criterion["revised_criterion"] = rewrite_text_with_mapping(
            criterion["text"], criterion["db_mappings"]
        )
        return new_criterion
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(process_criterion, criteria_list))


# -------------------------
# Validation of rewritten criteria
# -------------------------
class ValidatedExpr(BaseModel):
    valid: bool
    reason: str

def _validate_single_value(value, col_type, col_info):
    """Helper function to validate one value against the column schema."""
    val_type = "str"
    # Determine the type of the value (not a string representation)
    if isinstance(value, int):
        val_type = "int"
    elif isinstance(value, float):
        val_type = "float"

    NUMERIC_TYPES = {'int64', 'float64', 'int', 'float'}
    is_val_numeric = val_type in NUMERIC_TYPES
    is_col_numeric = col_type in NUMERIC_TYPES

    # if both val and col are numeric
    if is_val_numeric and is_col_numeric:
        return {"valid": True, "reason": "Numeric value is compatible with numeric column."}
    
    # type mismatch
    if is_val_numeric != is_col_numeric:
        return {"valid": False, "reason": f"Type Mismatch: Value {value} ({val_type}) is incompatible with column type '{col_type}'."}

    # Both are non-numeric, escalate to LLM for format check
    # Note: This is WIP and may give some false positives
    prompt = f"""
        You are a format validator.

        TASK:
        - Check if the filtering value's string format is compatible with the column's string format.
        - Do not expect a specific match with the provided sample column values; only check for pattern compatibility.
        - Always respond in JSON that strictly follows the schema below.

        CONTEXT:
        Filtering Value: {value}
        Column Summary: {col_info}

        SCHEMA:
        {{
            "valid": bool,      // boolean: whether the filtering value is compatible
            "reason": "string"  // short explanation of the decision
        }}

        Return only a JSON object that matches this schema.
    """
    response = call_llm(
        user_prompt=prompt,
        model="gpt-4o-mini",
        system_prompt="You are a strict data format validation agent.",
        response_model=ValidatedExpr
    )
    # Return dict instead of Pydantic object
    return response.model_dump()

def _validate_expr(expr, db_schema):
    """
    Validates an expression, handling standard operators and IN clauses with list value.
    Expression format: '<table.field> <operator> <value>'.
    """
    # 1. Parse the expression
    try:
        field_part, op, val_str = expr.strip().split(maxsplit=2)
        table, col = field_part.split(".")
    except ValueError:
        return {"valid": False, "reason": f"Invalid expression format. Expected '<table.field> <op> <value>'."}

    # 2. Get column info from DB schema
    try:
        col_info = db_schema[table]['fields'][col]
        col_type = col_info["field_data_type"].lower()
    except KeyError:
        return {"valid": False, "reason": f"Column '{table}.{col}' not found in database schema."}

    op = op.upper() # Standardize operator to uppercase

    # 3. Handle IN clause
    if op == 'IN':
        try:
            # Safely evaluate the string as a Python literal (e.g., "['a', 'b']")
            value_list = ast.literal_eval(val_str)
            if not isinstance(value_list, list):
                raise TypeError
        except (ValueError, SyntaxError, TypeError):
            return {"valid": False, "reason": "Operator 'IN' must be followed by a valid list (e.g., ['a', 'b'])."}
        
        if not value_list: # Empty list is valid
            return {"valid": True, "reason": "Expression with empty IN list is valid."}

        # Validate each item in the list
        in_results = []
        for item in value_list:
            result = _validate_single_value(item, col_type, col_info)
            if not result["valid"]:
                # Prepend context to the reason from the helper function
                result["reason"] = f"Invalid item in IN list. {result['reason']}"
                #return result
            in_results.append(result)
        if any(res["valid"] is False for res in in_results):
            return in_results
        else:
            return {"valid": True, "reason": "All items in the IN list are valid."}

    # 4. Handle other operators
    else:
        val = val_str.strip("'\"")
        parsed_val = val
        try:
            parsed_val = int(val)
        except ValueError:
            try:
                parsed_val = float(val)
            except ValueError:
                pass # Stays a str
        return _validate_single_value(parsed_val, col_type, col_info)

def validate_criteria(criteria_list: List[Dict[str, Any]], db_schema, max_workers: int = 8) -> List[Dict[str, Any]]:
    """
    Validate value type (and format for strs) compatibility with the db column for each expression in the list of criteria.
    Also checks if each table.col in criteria is present in the DB.
    """
    def process_criterion(criterion: Dict[str, Any]) -> Dict[str, Any]:
        crit = criterion["revised_criterion"]
        exprs = []
        # get single criteria by splitting on `OR` or `AND` clause
        if "OR" in crit:
            exprs = crit.strip().split("OR")
        elif "AND" in crit:
            exprs = crit.strip().split("AND")
        else:
            exprs = [crit.strip()]

        validation_results = []
        for expr in exprs:
            flag = _validate_expr(expr.strip(), db_schema)
            validation_results.append({
                "expression": expr.strip(),
                "validation": flag
            })

        new_criterion = criterion.copy()
        new_criterion["validation_results"] = validation_results
        return new_criterion

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(process_criterion, criteria_list))


# -------------------------
# Translate criteria to SQL
# -------------------------
# --- Helper functions to create the joins needed to combine data across tables ---

def find_join_path(start_table, end_table, schema_keys):
    """
    Finds the shortest sequence of tables to join from start to end
    using a schema definition with primary and foreign keys.
    """
    if start_table == end_table:
        return [start_table]

    queue = deque([[start_table]])
    visited = {start_table}

    while queue:
        path = queue.popleft()
        current_table = path[-1]

        # --- Find all neighbors using the PK/FK schema ---
        neighbors = set()

        # 1. Find tables that `current_table` points to (via its own FKs)
        for referenced_table in schema_keys[current_table]['fks'].values():
            neighbors.add(referenced_table)

        # 2. Find tables that point to `current_table` (reverse lookup)
        for other_table, details in schema_keys.items():
            if current_table in details['fks'].values():
                neighbors.add(other_table)

        # Process the found neighbors
        for neighbor in neighbors:
            if neighbor not in visited:
                new_path = list(path)
                new_path.append(neighbor)
                if neighbor == end_table:
                    return new_path  # Path found

                visited.add(neighbor)
                queue.append(new_path)

    return None # No path exists

def get_join_keys(table1_name, table2_name, schema_keys):
    """
    Finds the foreign key and primary key to join two adjacent tables.
    """
    t1_details = schema_keys[table1_name]
    t2_details = schema_keys[table2_name]

    # Case 1: Table 1 has an FK pointing to Table 2's PK
    for fk_col, ref_table in t1_details['fks'].items():
        if ref_table == table2_name:
            # Join on table1.fk_col = table2.pk
            return (fk_col, t2_details['pk'])

    # Case 2: Table 2 has an FK pointing to Table 1's PK
    for fk_col, ref_table in t2_details['fks'].items():
        if ref_table == table1_name:
            # Join on table1.pk = table2.fk_col
            return (t1_details['pk'], fk_col)
            
    return None # Should not happen if a path was found

# --- SQL query builder ---
def build_query_from_criteria(criteria, tables, schema_keys, root="main"):
    """
    Builds an SQL query with ORM functions, handling complex JOINs and AND/OR in condition strings 
    and multi-entry conditions via GROUP BY/HAVING.
    """
    
    # 1. Condition parser
    def _build_expression(condition_str):
        """Recursively builds a SQLAlchemy expression from a string."""
        # Base case of recursion: a simple 'table.col op value' clause
        def _parse_simple_clause(clause_str):
            parts = clause_str.strip().split(maxsplit=2)
            table, col = parts[0].split(".")
            op, val_str = parts[1], parts[2]  
            
            col_expr = tables[table].c[col]

            # Handle IN clause
            if op.upper() == 'IN':
                try:
                    # Safely evaluate the string representation of the list
                    list_of_values = ast.literal_eval(val_str)
                    if not isinstance(list_of_values, list):
                        raise ValueError("Value for IN operator must be a list.")
                    return col_expr.in_(list_of_values)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Could not parse list for IN condition: {val_str}") from e

            # Handle other operators
            else:
                # Strip quotes if string, else cast to int/float
                if val_str.startswith("'") and val_str.endswith("'"): 
                    val = val_str.strip("'")
                else:
                    try: val = int(val_str)
                    except ValueError: val = float(val_str)

                # Operator mapping
                ops = { "=": lambda c, v: c == v, ">": lambda c, v: c > v, "<": lambda c, v: c < v,
                        ">=": lambda c, v: c >= v, "<=": lambda c, v: c <= v, "!=": lambda c, v: c != v }
                if op not in ops:
                    raise ValueError(f"Unsupported operator: {op}")
            return ops[op](col_expr, val)

        # Recursive step: split by OR, then by AND
        or_parts = [p.strip() for p in condition_str.split(' OR ')]
        or_clauses = []
        for part in or_parts:
            and_parts = [p.strip() for p in part.split(' AND ')]
            and_clauses = [_parse_simple_clause(p) for p in and_parts]
            or_clauses.append(and_(*and_clauses))
            #print(part, and_parts)
        
        return or_(*or_clauses)

    # 2. Parse and categorize criteria (WHERE v/s GROUP_BY/HAVING)
    include_counts = collections.Counter()
    for c in criteria:
        if c['type'] == 'include':
            # This simple check is sufficient to find multi-entry cases
            simple_cond = c['revised_criterion'].split(' AND ')[0].split(' OR ')[0]
            table, col = simple_cond.strip().split(maxsplit=2)[0].split('.')
            include_counts[(table, col)] += 1
    
    where_filters = []
    having_filters = []
    for c in criteria:
        simple_cond = c['revised_criterion'].split(' AND ')[0].split(' OR ')[0]
        table, col = simple_cond.strip().split(maxsplit=2)[0].split('.')
        is_multi_entry = include_counts.get((table, col), 0) > 1

        # A condition is for HAVING only if it's simple and part of a multi-entry set
        is_simple_condition = ' AND ' not in c['revised_criterion'] and ' OR ' not in c['revised_criterion']
        if c['type'] == 'include' and is_multi_entry and is_simple_condition:
            having_filters.append(c)
        else:
            where_filters.append(c)
    #print(f"Where:\n{where_filters}")
    #print(f"Having:\n{having_filters}")

    # 3. Initialize query, build joins across reqd tables, handle joins across tables not sharing PK/FK with bridging
    root_table = tables[root]
    root_pk = schema_keys[root]['pk']
    grouping_key = root_table.c[root_pk]   # assume the pk of root table is the reference key for grouping
    # initialize query obj
    query = select(grouping_key) # TO-DO: provide option to supply more keys for data fetching
    
    # Get the tables from the criteria
    required_tables = set()
    for c in criteria:
        # extract table names from table.field mentions in each condition
        words = c["revised_criterion"].split()
        for word in words:
            if '.' in word:
                table_name = word.split('.')[0]
                if table_name in tables:
                    required_tables.add(table_name)
    #print(f'Tables in criteria: {required_tables}')

    # Find the join path for each required table and add necessary joins
    tables_in_query = {root}
    for target_table in required_tables:
        if target_table in tables_in_query:
            continue

        # Call the new pathfinder
        path = find_join_path(root, target_table, schema_keys)
        if path is None:
            raise Exception(f"No join path found from '{root}' to '{target_table}'")

        # Iterate through the path to build joins
        for i in range(len(path) - 1):
            from_table_name = path[i]
            to_table_name = path[i+1]
            if to_table_name not in tables_in_query:
                # Get the join columns using the new helper
                from_key, to_key = get_join_keys(from_table_name, to_table_name, schema_keys)

                from_table = tables[from_table_name]
                to_table = tables[to_table_name]

                query = query.join(to_table, from_table.c[from_key] == to_table.c[to_key])
                tables_in_query.add(to_table_name)
    #print(f'Tables in the framed query: {tables_in_query}') # this can include bridging tables

    # 4. Apply row-level filters (WHERE caluse)
    for f in where_filters:
        expression = _build_expression(f['revised_criterion'])
        query = query.filter(expression if f['type'] == 'include' else not_(expression))

    # 5. APPLY group-level filters (HAVING clause)
    if having_filters:
        query = query.group_by(grouping_key)
        having_clauses = []
        for f in having_filters:
            # Having filters are guaranteed to be simple by our categorization logic
            parts = f['revised_criterion'].strip().split(maxsplit=2)
            table, col = parts[0].split('.'); val_str = parts[2]
            if val_str.startswith("'"): val = val_str.strip("'")
            else: val = int(val_str)
            col_expr = tables[table].c[col]
            condition = func.count(case((col_expr == val, 1))) > 0
            having_clauses.append(condition)
        query = query.having(and_(*having_clauses))

    query_str = str(query.compile(compile_kwargs={"literal_binds": True}))
    return {
        "criteria": criteria,
        "sql_query": query_str
    }

# -------------------------
# LLM decision agent
# -------------------------
class AgentDecision(BaseModel):
    thinking: str = Field(..., description="Concise reasoning behind the choice")
    action: Literal["advance", "edit", "undo", "clarify", "reject"] = Field(
        ..., description="Next action chosen by the agent"
    )
    question: Optional[str] = Field(
        None, description="Optional clarifying question if action is 'clarify'"
    )

def agent_decide(conversation_history: List[str], current_criteria: Dict) -> AgentDecision:
    """
    Prepares a prompt and calls an LLM to get the next action based on history and current input.
    """

    history_str = "\n".join([f"- {msg}" for msg in conversation_history])
    prompt = f"""
        You're a biomedical expert agent helping a user build query criteria to retrieve database records.

        TASK: Analyze the recent conversation history and current state, then decide on your next action.

        <ACTIONS>
        - `advance`: run the next step after user explicilty approves current result
        - `edit`: refine the current state based on feedback
        - `undo`: undo the last edit made and go back to previous state
        - `clarify`: ask a clarifying question if user input is unclear (acronyms, vague attributes, conflicts) 
                        or does not apply to current state
        - `reject`: flag irrelevant or out-of-scope input; stop
        </ACTIONS>

        <USER_INPUT_TYPES>
        - initial query
        - clarifying comment on earlier query/feedback
        - approval of current result, to advance to next step
        - feedback or edits to current result state
        - irrelevant for cohort building or DB retrieval; reject
        </USER_INPUT_TYPES>

        <STATES>
        - 0: initial user query, no logical criteria generated
        - 1: eligibility criteria generated
        - 2: entities parsed from criteria
        - 3: parsed entities mapped to DB schema
        - 3: entities mapped to DB concepts/values
        - 5: Criteria rewritten with the mapped entities
        - 6: Criteria validated for DB compatibility
        - 7: SQL query generated from criteria
        </STATES>

        Current State: {current_criteria}
        Conversation History: {history_str}
        Latest User Message: {conversation_history[-1]}

        Always respond with strict JSON:
        {{
          "thinking": "...", # concise summary
          "action": "...",
          "question": "..."  # optional
        }}

        Response (strict JSON):
        """
    # Call the LLM with structured output
    llm_response = call_llm(user_prompt=prompt, response_model=AgentDecision)

    if isinstance(llm_response, AgentDecision):
        return llm_response
    else:
        raise ValueError("Invalid LLM response format")

# -------------------------
# 'Tools'
# -------------------------
def process_query(state_dict: Dict, stage_counter: int, db_schema, db_embeddings, concept_df, concept_lookup, schema_keys, root, method):
    """
    Calls the right function on the current state to generate/structure criteria depending on stage.
    Each call to this function advances the sequence by a stage.

    Args:
        state_dict (Dict[str, Any]): The current state object stoing the original query, current criteria and conv history.
        stage_counter (int): Step in the processing sequence the state is currently at.
        db_schema (Dict[str, str]): DB schema with desctiptions of tables and fields
        db_embeddings (Dict[str, Dict[str, List]]): Embedding vectors for DB columns (text summaries) 
        concept_df: Keeps track of concept's table & field
        concept_lookup: stores concepts and its embeddings

    Returns:
        List[Dict[str, Any]]: Updated list of criteria.
    """
    if stage_counter == 0:
        print(f"> Agent to User: Generating eligibility criteria")
        user_input = state_dict["conversation_history"][-1]["user"]
        feedback = user_input if user_input != state_dict["original_query"] else ""  # only last feedback included
        result = extract_raw_criteria(
            state_dict["original_query"],
            state_dict["current_criteria"],
            feedback,
        )
    elif stage_counter == 1:
        print(f"> Agent to User: Entity extraction from criteria")
        result = extract_criteria_entities(state_dict["current_criteria"])
    elif stage_counter == 2:
        print(f"> Agent to User: Mapping parsed entities to schema")
        result = map_criteria_to_schema(state_dict["current_criteria"], db_schema, db_embeddings, method)
    elif stage_counter == 3:
        print(f"> Agent to User: Mapping entities to concepts")
        result = map_entity_to_concept(state_dict["current_criteria"], concept_df, concept_lookup)
    elif stage_counter == 4:
        print(f"> Agent to User: Rewriting criteria based on mapped entities")
        result = rewrite_user_criteria(state_dict["current_criteria"])
    elif stage_counter == 5:
        print(f"> Agent to User: Validating criteria for database compatibility")
        result = validate_criteria(state_dict["current_criteria"], db_schema)
    elif stage_counter == 6:
        print(f"> Agent to User: Generating SQL query from criteria")
        schema = {}
        for table, table_info in db_schema.items():
            fields = list(table_info.get("fields", {}).keys())
            schema[table] = fields
        # Tables in SQAlchemy compatible format
        metadata = MetaData()
        tables = {
            t: Table(t, metadata, *(Column(c, String) for c in cols)) 
            for t, cols in schema.items()
        }
        result = build_query_from_criteria(state_dict["current_criteria"], tables, schema_keys, root)
    return result


class FeedbackValidation(BaseModel):
    validation: Literal["pass", "fail"] = Field(
        ..., description="Indicates whether the feedback is valid or not"
    )
    reason: str = Field(
        ..., description="Concise reason for pass/fail decision"
    )

def edit_with_feedback(user_input: str, state_dict: Dict, stage_counter: int):
    """
    Calls the right edit tool depending on stage to modify the current state based on recent user feedback.
    """
    recent_feedback = '\n'.join([f'- {s["user"]}' for s in state_dict["conversation_history"][-2:]])
    check_prompt = f"""
        Examine the current state JSON and the provided user feedback.
        Validate the feedback -> it should:
            - make sense in context of the current state,
            - instruct to either modify any element(s) in any object, or add a new object, to the state JSON. 

        Always return a strict JSON:
        {{
            "validation": "pass" or "fail",
            "reason": "...", # be concise
        }}

        Current State: {state_dict["current_criteria"]}
        User Input: {recent_feedback}
        Response (JSON):
    """
    # Call LLM with structured parsing into FeedbackValidation
    checked_status = call_llm(user_prompt=check_prompt, response_model=FeedbackValidation)

    if isinstance(checked_status, FeedbackValidation):
        if checked_status.validation == "fail":
            print(f"\n> Agent Response: {checked_status.reason}\nPls provide valid instructions to edit your criteria.")
            return None
        else:
            return edit_tool(user_input, state_dict, stage_counter)
    else:
        raise ValueError("Invalid LLM response format")


class BaseCriterion(BaseModel):
    type: Literal["include", "exclude"]
    text: str

# Stage 0/1 → Raw criteria
class RawCriterion(BaseCriterion):
    pass

class RawCriteriaState(BaseModel):
    criteria: List[RawCriterion]


# Stage 2 → With entities extracted
class EntityCriterion(BaseCriterion):
    entities: List[str]

class EntityCriteriaState(BaseModel):
    criteria: List[EntityCriterion]


# Stage 3 → Schema mapping
class MappingInfo(BaseModel):
    entity_class: Optional[str] = None
    table_field: Optional[str] = Field(
        None, alias="table.field", description="Schema table + field"
    )
    ranked_matches: Optional[List[str]] = None

    class Config:
        populate_by_name = True

class DbMappingEntry(BaseModel):
    entity: str
    mapping: MappingInfo

class SchemaCriterion(EntityCriterion):
    db_mappings: List[DbMappingEntry]

class SchemaCriteriaState(BaseModel):
    criteria: List[SchemaCriterion]


# Stage 4 → Concept mapping
class ConceptMappingInfo(MappingInfo):
    mapped_concept: Optional[str] = None
    mapping_method: Optional[str] = None
    reason: Optional[str] = None
    top_candidates: Optional[List[str]] = None

class ConceptDbMappingEntry(BaseModel):
    entity: str
    mapping: ConceptMappingInfo

class ConceptCriterion(SchemaCriterion):
    db_mappings: List[ConceptDbMappingEntry]

class ConceptCriteriaState(BaseModel):
    criteria: List[ConceptCriterion]


# Stage 5 → Rewritten criteria
class RewrittenCriterion(ConceptCriterion):
    revised_criterion: str

class RewrittenCriteriaState(BaseModel):
    criteria: List[RewrittenCriterion]

# Stage 6 → Validated criteria
class ValidationInfo(BaseModel):
    valid: bool
    reason: str

class ValidationResult(BaseModel):
    expression: str
    validation: ValidationInfo

class ValidatedCriterion(RewrittenCriterion):
    validation_results: List[ValidationResult]

class ValidatedCriteriaState(BaseModel):
    criteria: List[ValidatedCriterion]

# Stage 7 → SQL query
class SQLCriteriaState(ValidatedCriteriaState):
    sql_query: str

STAGE_MODELS: Dict[int, Type[BaseModel]] = {
    1: RawCriteriaState,       # only type/text
    2: EntityCriteriaState,    # + entities
    3: SchemaCriteriaState,    # + db_mappings
    4: ConceptCriteriaState,   # + concept mapping
    5: RewrittenCriteriaState,  # + revised_criterion
    6: ValidatedCriteriaState,   # + validation_results
    7: SQLCriteriaState         # + sql_query
}


def db_mappings_list_to_dict(db_map_list: Optional[List[Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert list-of-DbMappingEntry (or list-of-dicts) into dict keyed by entity.
    Each mapping object will expose "table.field" as the canonical key (converted from table_field).
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not db_map_list:
        return out

    for entry in db_map_list:
        # entry can be a DbMappingEntry instance or a plain dict (from model_dump)
        if hasattr(entry, "model_dump"):
            ed = entry.model_dump()
        else:
            ed = dict(entry)

        entity = ed.get("entity")
        mapping = ed.get("mapping", {}) or {}

        # normalize nested mapping keys: if mapping uses "table_field", convert to "table.field"
        if "table_field" in mapping and "table.field" not in mapping:
            mapping["table.field"] = mapping.pop("table_field")

        out[entity] = mapping

    return out


def dict_to_db_mappings_list(db_map_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert dict-shaped db_mappings (entity -> mapping dict) into a list of entries:
    [{ "entity": "<ent>", "mapping": { ... } }, ...] with mapping using 'table_field' key
    (preferred for our Pydantic model validation).
    """
    entries: List[Dict[str, Any]] = []
    for entity, mapping in (db_map_dict or {}).items():
        # make a shallow copy to avoid mutating caller dict
        m = dict(mapping or {})

        # prefer table.field -> convert to table_field for model validation
        table_field = m.get("table.field", m.get("table_field", None))
        normalized = {
            "entity_class": m.get("entity_class"),
            "table_field": table_field,
            "ranked_matches": m.get("ranked_matches"),
            "mapped_concept": m.get("mapped_concept"),
            "mapping_method": m.get("mapping_method"),
            "reason": m.get("reason"),
            "top_candidates": m.get("top_candidates"),
        }
        entries.append({"entity": entity, "mapping": normalized})
    return entries


def coerce_raw_to_canonical(parsed: Any) -> Dict[str, Any]:
    """
    Coerce raw parsed JSON (possibly a list or an object) into a dict {'criteria': [ ... ]}
    where each criterion uses db_mappings as a LIST of {entity, mapping}.
    This is a best-effort coercion to make validation possible.
    """
    if isinstance(parsed, dict) and "criteria" in parsed:
        criteria = parsed["criteria"]
    elif isinstance(parsed, list):
        criteria = parsed
    else:
        raise ValueError("Unexpected JSON shape from LLM; expected list or {'criteria': [...] }")

    coerced = []
    for crit in criteria:
        crit_copy = dict(crit or {})
        dbmap = crit_copy.get("db_mappings")

        # If `db_mappings` is a dict (entity -> mapping), convert to list representation
        if isinstance(dbmap, dict):
            crit_copy["db_mappings"] = dict_to_db_mappings_list(dbmap)

        # If `db_mappings` is already a list, make sure each entry uses mapping.table_field if table.field present
        elif isinstance(dbmap, list):
            normalized_list = []
            for e in dbmap:
                ecopy = dict(e or {})
                mapping = dict(ecopy.get("mapping") or {})
                # if LLM returned "table.field" inside mapping, convert to "table_field"
                if "table.field" in mapping and "table_field" not in mapping:
                    mapping["table_field"] = mapping.pop("table.field")
                ecopy["mapping"] = mapping
                normalized_list.append(ecopy)
            crit_copy["db_mappings"] = normalized_list

        # else: None or missing -> keep as-is (validation will accept Optional)

        coerced.append(crit_copy)

    return {"criteria": coerced}


def edit_tool(user_input: str, state_dict: Dict, stage_counter: int):
    """
    Selector for function to edit criteria JSON state based on the stage.
    Stages:
    1: Raw inferred criteria as incl/excl conditions
    2: Criteria with entities extracted per condition
    3: Criteria with parsed entities mapped to schema table/field
    4: Criteria with parsed entities mapped to schema concepts
    5: Criteria rewritten using the entity mappings
    6: Criteria validated for DB compatibility
    7: SQL query generated from criteria

    TO-DO: Make this more deterministic by reducing reliance on LLM edit.
    """

    recent_feedback = '\n'.join([f'- {s["user"]}' for s in state_dict["conversation_history"][-2:]])

    if stage_counter == 1:
        raw_criteria = extract_raw_criteria(
            state_dict["original_query"],
            state_dict["current_criteria"],
            recent_feedback, #user_input, only the last 2 feedbacks used, earlier clarifications not used
        )
        return raw_criteria
    else:
        edit_prompt = f"""
        Instructions (step-by-step):
        - Examine the CURRENT STATE JSON and the USER EDIT/FEEDBACK.
        - Identify the target object for the edit.
        - Decide the minimal change needed to implement user edit in the object.
        - Proceed to make the edit in the precise place(s).
        - Return a strict JSON: the orignal state JSON with the minimal edit made. 
        - Strictly replicate/retain only keys present in the CURRENT STATE JSON in your response even if you notice extra optional keys in the provided SCHEMA. Also preserve the order of keys.
        - Return ONLY a strict JSON object that matches the schema described below. No explanations, no markdown.

        SCHEMA (Note that this is a generic schema with all possible fields, and not all fields may be present in the CURRENT STATE JSON):
        Return an object: {{
        "criteria": [
            {{
            "type": "include" | "exclude",
            "text": "...",
            "entities": ["..."],               # optional
            "db_mappings": [                   # optional; LIST of mapping entries
                {{
                "entity": "<entity_string>",
                "mapping": {{
                    "entity_class": "...",
                    "table_field": "...",
                    "ranked_matches": [...],
                    "mapped_concept": "...",
                    "mapping_method": "...",
                    "reason": "...",
                    "top_candidates": [...]
                }}
                }}
            ],
            "revised_criterion": "..."         # optional
            "validation_results": [            # optional
                {{
                "expression": "...",
                "validation": {{
                    "valid": true | false,
                    "reason": "..."
                }}
            }},
            ...
        ],
        "sql_query": "..."  # optional
        }}

        CURRENT STATE (for reference):
        {json.dumps(state_dict.get("current_criteria", []), indent=2)}

        USER EDIT / FEEDBACK:
        {recent_feedback}

        Return the entire updated object (with key 'criteria') in that schema.
        """
        # Pick stage-specific response model
        response_model = STAGE_MODELS.get(stage_counter, SQLCriteriaState)

        # 1) Try the structured path (preferred)
        try:
            response = call_llm(user_prompt=edit_prompt, response_model=response_model)
            if isinstance(response, response_model):
                final_state: List[Dict[str, Any]] = []
                for crit in response.criteria:
                    crit_dict = crit.model_dump()
                    crit_dict["db_mappings"] = db_mappings_list_to_dict(crit_dict.get("db_mappings"))
                    final_state.append(crit_dict)

                # preserve sql_query only if stage 7
                if stage_counter == 7 and hasattr(response, "sql_query") and response.sql_query:
                    state_dict["sql_query"] = response.sql_query

                if stage_counter == 7:
                    return {"criteria": final_state, "sql_query": state_dict.get("sql_query", "")}
                else:
                    return final_state
            else:
                print(f"[WARN] Structured parse returned unexpected type ({type(response)}); falling back to raw parse.")
        except Exception as e_struct:
            print(f"[WARN] Structured parse failed ({e_struct}); falling back to raw JSON parse.")

        # 2) Fallback: raw text + coercion + validation
        raw = call_llm(user_prompt=edit_prompt, response_model=None)
        parsed = None
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception as e_json:
                print(f"[ERROR] LLM returned non-JSON text and could not parse: {e_json}\nRaw:\n{raw}")
                return None
        else:
            parsed = raw

        try:
            coerced = coerce_raw_to_canonical(parsed)
        except Exception as e_coerce:
            print(f"[ERROR] Could not coerce raw LLM output into canonical shape: {e_coerce}")
            return None

        try:
            validated = response_model.model_validate(coerced)
        except Exception as e_val:
            print(f"[ERROR] Validation of coerced LLM result failed: {e_val}")
            return None

        final_state: List[Dict[str, Any]] = []
        for crit in validated.criteria:
            crit_dict = crit.model_dump()
            crit_dict["db_mappings"] = db_mappings_list_to_dict(crit_dict.get("db_mappings"))
            final_state.append(crit_dict)

        # preserve sql_query only if stage 7
        if stage_counter == 7 and hasattr(validated, "sql_query") and validated.sql_query:
            state_dict["sql_query"] = validated.sql_query

        if stage_counter == 7:
            return {"criteria": final_state, "sql_query": state_dict.get("sql_query", "")}
        else:
            return final_state

# -------------------------
# LangGraph nodes: user, agent, process_query, edit, undo
# -------------------------
class AgentState(TypedDict):
    original_query: Optional[str]
    conversation_history: List[Dict[str, Any]]
    user_input: str
    current_criteria: Any
    stage_counter: int
    last_state: Any
    agent: Dict[str, Any]


def react_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: decide next action via agent_decide()
    """
    decision = agent_decide(state["conversation_history"], state["current_criteria"])
    state["agent"] = decision.model_dump()
    # attach action to the last conversation_history entry
    if state["conversation_history"]:
        state["conversation_history"][-1]["action"] = decision.action
    if decision.action == "clarify" and decision.question:
        state["conversation_history"][-1]["question"] = decision.question
    print(f"\nAgent action: {decision.action}")
    print(f"\nAgent thinking: {decision.thinking}")
    return state


def process_query_node(state: AgentState) -> AgentState:
    """
    LangGraph node: advance (process next stage)
    """
    sc = state["stage_counter"]
    result = process_query(state, sc, DB_SCHEMA, DB_EMBEDDINGS, CONCEPT_DF, CONCEPT_LOOKUP, SCHEMA_KEYS, ROOT_TABLE, FIELD_MAPPING_METHOD)
    state["current_criteria"] = result
    state["stage_counter"] = sc + 1
    print(f"\nProcessed stage {sc}. Updated criteria:")
    print(json.dumps(result, indent=2, default=str))
    return state


def edit_node(state: AgentState) -> AgentState:
    """
    LangGraph node: edit current state using edit_with_feedback()
    """
    sc = state["stage_counter"]
    user_input = state["conversation_history"][-1]["user"] if state["conversation_history"] else ""
    state["last_state"] = state["current_criteria"]
    result = edit_with_feedback(user_input, state, sc)
    if result:
        state["current_criteria"] = result
        state["conversation_history"][-1]["result"] = "edited"
        print("\nUpdated state (after edit):")
        print(json.dumps(result, indent=2, default=str))
    else:
        state["conversation_history"][-1]["result"] = "none"
    return state


def user_node(state: AgentState) -> Tuple[AgentState, Command[Literal["agent", END]]]:
    """
    LangGraph node: prompt user, update state, route to agent (or END on quit)
    """
    user_in = input("User > ")
    if not state.get("original_query"):
        state["original_query"] = user_in
    state["user_input"] = user_in
    state["conversation_history"].append({"user": user_in})
    if user_in.lower().strip() in ["quit", "exit"]:
        return state, Command(goto=END)
    return state, Command(goto="agent")

def undo_node(state: AgentState) -> AgentState:
    """
    Undo last edit
    """
    state["current_criteria"] = state["last_state"].copy()
    print(f"\nUndo: reverted to previous state:\n{json.dumps(state['current_criteria'], indent=2)}")
    return state

def route_action(state: AgentState) -> str:
    """
    Decide which node to go to from agent.
    """
    action = state.get("agent", {}).get("action", "stop")
    # Prevent infinite run: if agent chooses advance while stage_counter at max stage (7) -> END
    if action == "advance" and state.get("stage_counter", 0) >= 7:
        return "stop"
    return action

# -------------------------
# Main: build LangGraph workflow and run interactive loop
# -------------------------
def main():
    global DB_SCHEMA, DB_EMBEDDINGS, CONCEPT_DF, CONCEPT_LOOKUP, SCHEMA_KEYS, ROOT_TABLE, FIELD_MAPPING_METHOD

    ROOT_TABLE = "patient"
    FIELD_MAPPING_METHOD = "sequential"

    # load DB schema, embeddings, concept table etc.
    with open('./BeatAML/BeatAML_schema.json', 'r') as f:
        DB_SCHEMA = json.load(f)
    with open('./BeatAML/BeatAML_schema_keys.json', 'r') as f:
        SCHEMA_KEYS = json.load(f)
    with open('./BeatAML/db_table_field_embeddings.json', 'r') as f:
        DB_EMBEDDINGS = json.load(f)
    CONCEPT_DF = pd.read_csv("BeatAML/concept_table.csv")
    with open("BeatAML/concept_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    concepts = data["concepts"]
    embeddings = data["embeddings"]
    CONCEPT_LOOKUP = dict(zip(concepts, embeddings))

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", react_agent)
    workflow.add_node("advance", process_query_node)
    workflow.add_node("edit", edit_node)
    workflow.add_node("undo", undo_node)
    workflow.add_node("user", user_node)
    workflow.add_edge(START, "user")
    workflow.add_conditional_edges(
        "agent", route_action,
        {
            "advance": "advance",
            "edit": "edit",
            "clarify": "user",
            "undo": "undo",
            "reject": END,
            "stop": END
        }
    )
    workflow.add_edge("advance", "user")
    workflow.add_edge("edit", "user")
    workflow.add_edge("undo", "user")
    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer)

    initial_state: AgentState = {
        "original_query": None,
        "conversation_history": [],
        "user_input": "",
        "current_criteria": {},
        "last_state": {},
        "stage_counter": 0,
        "agent": {}
    }
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id},
              "recursion_limit": 100}
    
    print("\n--- Session Initialized ---")
    print("You can start by providing a query. Type 'quit' to exit at any point.")
    print("\nUser Options:\n* ask - initial request\n* clarify - provide follow-up clarification if needed\n* approve - advance to next step\n" \
    "* edit - modify or correct current state\n* undo - reverse the last edit made")
    print("-" * 40)

    graph.update_state(config, initial_state)
    graph.invoke({}, config=config)

    print("\n------ Exited Loop ------")
    snapshot = graph.get_state(config)
    state_dict = snapshot.values
    print(f"\nFinal criteria:\n{json.dumps(state_dict['current_criteria'], indent=2, default=str)}")
    state_dict["stage_counter"] = state_dict.get("stage_counter", 0)
    state_dict["session_end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs("langgraph_cb_runs", exist_ok=True)
    with open(f"langgraph_cb_runs/v1_agent_session_{session_id}.json", "w") as f:
        json.dump(state_dict, f, indent=2, default=str)


if __name__ == "__main__":
    main()
