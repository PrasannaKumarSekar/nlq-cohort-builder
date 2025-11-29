"""
This module contains the functions to be used as tools by the Cohort Builder Agent.
"""

from rich import print
from typing import List, Dict, Any, Literal, Tuple, Optional, Union, Type, Annotated
from pydantic import BaseModel, Field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from langchain_core.tools import tool

import os, json
from concurrent.futures import ThreadPoolExecutor
import ast
import subprocess
import sys
from datetime import datetime

from polly.atlas import Atlas

from openai import OpenAI
#os.environ["OPENAI_API_KEY"] = "sk-proj-"  # set openai api key as env variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# --------- Helper functions and Tools ------------
def call_llm(
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    system_prompt: str = "You are a biomedical expert agent.",
    response_model: Optional[Type[BaseModel]] = None,
) -> BaseModel | str:
    """
    Generalized function to call LLM with flexible settings and structured output.

    Args:
        user_prompt (str): The main input prompt for the model.
        model (str): The LLM model to use (default: gpt-4o-mini).
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

# ----- Tool to query/inspect specific DB tables -----
@tool
def inspect_table(table: str) -> Dict:
    """
    Fetches schema for a table as a JSON to understand the data.
    ------
    Args:
        table (str): Name of table for which details are sought
    Returns:
        A dictionary with table description and column details incl summary, data type and sample values.    
    """
    data_dir = os.getenv("BEATAML_DATA_DIR", "../../BeatAML")
    with open(os.path.join(data_dir, 'BeatAML_schema.json'), 'r') as f:
        DB_SCHEMA = json.load(f)

    if table in DB_SCHEMA.keys():
        schema = {
                    "desc": DB_SCHEMA[table]['table_description'].split('.')[0],
                    "fields": {}
                }
        for field, details in DB_SCHEMA[table]["fields"].items():
            desc = "desc: " + details["field_description"].split('.')[0]
            type = "type: " + details["field_data_type"]
            sample_values = "sample vals: " + str(details["field_sample_values"])
            summary = "\n".join([desc, type, sample_values])
            schema["fields"][field] = summary
        return schema
    else:
        print(f"No table `{table}` not found in DB schema.")
        return None


# ----- Tool to inspect content of specific field -----
@tool
def inspect_field(table: str, field: str) -> str|List:
    """
    Fetches unique set of concepts/values stored in a DB field.
    ------
    Args:
        table (str): Name of table
        field (str): Name of selected field
    Returns:
        List of unique values present in the selected field, upto 100 (or a flag if no concepts found).
    """
    # TO-DO: Better handling of fields with very large number of unique values to limit response length to LLM
    data_dir = os.getenv("BEATAML_DATA_DIR", "../../BeatAML")
    concept_df = pd.read_csv(os.path.join(data_dir, "concept_table.csv"))  
    # cols: concept_name, table_name, field_name, concept_with_context
    
    subset = concept_df[
        (concept_df["table_name"] == table) &
        (concept_df["field_name"] == field)
    ].copy()
    if subset.empty:
        return 'No concepts/values found for field.'
    unique_vals = list(set(subset['concept_name']))
    if len(unique_vals)>100:
        return f'Returning first 100 of total {len(unique_vals)} unique values', unique_vals[:100]
    return unique_vals


# ----- Tool and helper functions to perform entity mapping -----
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
    using sklearn cosine_similarity.
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
    rankings: List[Ranking] = Field(..., description="List of ranked candidates, starting from 1")

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
    Rank all candidates by relevance to the query, from most to least relevant.
    **Important**: Make use of the description, data type, and sample values format to base your ranking.
    Use the added context if provided.
    
    INPUT:
    Query: {query}
    Context: {context}

    Candidates:
    {json.dumps(candidates_list, indent=2)}

    OUTPUT FORMAT:
    Return ONLY strict JSON.
    """

    response = call_llm(user_prompt=prompt, 
                        model='gpt-4o-mini', 
                        system_prompt='You are a ranking assistant', 
                        response_model=RerankResult)

    if isinstance(response, RerankResult):
        # Convert list back into dict { "1": candidate, ... } if needed
        return {str(r.rank): r.candidate for r in response.rankings}
    else:
        print("Error: LLM did not return valid rankings.")
        return {}
    
class TableMappingResponse(BaseModel):
    """LLM response for mapping entity to most probable table."""
    table: str = Field(..., description="Most relevant table name")

class FieldMappingResponse(BaseModel):
    """LLM response for mapping entity to most probable field in a given table."""
    field: str = Field(..., description="Most relevant field name")

def map_entity_to_table_field(item: Tuple, 
                              method: str = 'sequential'
                              ) -> Dict[str, Any]:
    """
    Helper function to map an item to the most likely table and field.
    ------
    Args:
        item (Tuple): A tuple in format (attribute, entity, context_text) to be mapped to schema
        method (str): Mapping method 

    Returns:
        Dict[str, Any]: input item mapped to a dict with the entity, attribute, and ranked table.field matches
    """

    # load DB schema
    data_dir = os.getenv("BEATAML_DATA_DIR", "../../BeatAML")
    with open(os.path.join(data_dir, 'BeatAML_schema.json'), 'r') as f:
        schema = json.load(f)
    # load vector embeddings for field descr texts
    with open(os.path.join(data_dir, 'db_table_field_embeddings.json'), 'r') as f:
        DB_EMBEDDINGS = json.load(f)
    
    attribute, entity, context_text = item  # Parse input tuple

    if method == 'sequential':
        # Step 1: Map attr-entity to the most probable Table
        table_descriptions = {name: details['table_description'] for name, details in schema.items()}
        mapped_table = call_llm(
            user_prompt=f"""
                Given entity "{entity}" from attribute "{attribute}" and context "{context_text}", 
                map it to the most relevant Table. Respond with only the Table name.

                Tables:
                {json.dumps(table_descriptions, indent=2)}
                """,
            response_model=TableMappingResponse
        ).table

        if mapped_table not in schema:
            return {"attribute": attribute, "entity": entity,
                    "table_field_matches": 'UNKNOWN'}

        # Step 2: Map attr-entity to the most probable Field within the selected table
        field_descriptions = schema[mapped_table]['fields']
        mapped_field = call_llm(
            user_prompt=f"""
                Given entity "{entity}" from attribute "{attribute}" and context "{context_text}", 
                map it to the most relevant Field in "{mapped_table}" Table. Respond with only the Field name.

                Fields in "{mapped_table}":
                {json.dumps(field_descriptions, indent=2)}
                """,
            response_model=FieldMappingResponse
        ).field
        
        if mapped_field not in field_descriptions:
            return {"attribute": attribute, "entity": entity,
                    "table_field_matches": 'UNKNOWN'}
        
        # Step 3: Find other fields similar to the chosen option based on vector embeddings, then rerank
        key = f'{mapped_table}.{mapped_field}'
        top_matches = find_matches(DB_EMBEDDINGS[key]['embedding'], DB_EMBEDDINGS, top_k=5)
        candidates = [(name, DB_EMBEDDINGS[name]['text']) for name, _ in top_matches]
        ranked_matches = rerank_with_llm(entity, candidates, context_text)
        if not ranked_matches:
            return {"attribute": attribute, "entity": entity, 
                "table_field_matches": "none"}
        return {"attribute": attribute, "entity": entity, 
                "table_field_matches": ranked_matches}
        '''
        if ranked_matches:
            try:
                mapped_table, mapped_field = ranked_matches["1"].split('.')
            except Exception as e:
                print(f'Error with LLM-reranking step for entity {entity}: {e}')
                return {"attribute": attribute, "entity": entity, 
                        "table.field": f'{mapped_table}.{mapped_field}', "ranked_matches": None}
                
        return {"attribute": attribute, "entity": entity, 
                "table.field": f'{mapped_table}.{mapped_field}', "ranked_matches": list(ranked_matches.values())}
        '''

# ----- Tool to map entities to specific tables and fields -----
@tool
def get_relevant_database_fields(items: List[List[str]]) -> Dict[str, Any]:
    """
    Map a list of (attribute, entity, context) tuples parsed from user query to most relevant DB tables/fields.
    ------
    Args:
        items (list): A list of tuples. Each tuple has format (attribute, entity, context), 
        which is to be mapped to DB schema.
    Returns:
        Each input item mapped to a dict with attribute, entity, and a ranked list of table.field matches.

    Usage example:
    items = [('drug', 'insulin', 'diabetics not on insulin'),]
    mapping = get_relevant_database_fields(items) 
    # {'attribute':'drug', 'entity':'insulin', 'table_field_matches': {1:'Patient.insulin_status', 2:'Patient.treatment_name'}}
    """
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(map_entity_to_table_field, items))


# ----- Helper functions and tool to map extracted entities to specific values in DB -----
class ConceptMappingResult(BaseModel):
    concepts: List[str] = Field(..., description="All relevant concepts from candidates")
    reason: str = Field(..., description="One short sentence explaining the choice")
    
def _llm_choose_best_concept(entity: str, table: str, field: str, candidates: List[str], model: str = "gpt-4o-mini"):
    bullet_list = "\n".join(f"- {c}" for c in candidates)
    prompt = f"""
    You're helping map a user-provided entity to the best matching concepts under a specific database field.
    
    TASK:
    - Pick all values/concepts relevant to the entity (verbatim from the list), return a ranked list.
    - Provide a short reason for the choice.
    
    CONTEXT:
    - Table: {table}
    - Field: {field}
    - Entity to map: "{entity}"

    CANDIDATES:
    {bullet_list}

    OUTPUT FORMAT:
    Return strict JSON only, matching this schema:
    {{
      "concepts": ["<concept1>", "<concept2>", ...],  # can be empty if no matching concept found
      "reason": "<short sentence>",
    }}
    """
    try:
        result = call_llm(
            user_prompt=prompt,
            model=model,
            system_prompt="You are a precise data-mapping assistant who chooses the best concept label(s).",
            response_model=ConceptMappingResult,
        )
        if isinstance(result, ConceptMappingResult):
            return result.concepts, result.reason
        else:
            return [], "Invalid response format"
    except Exception as e:
        return [], f"LLM call failed: {e}"
    
def map_entity_to_concept(item: Tuple) -> Dict[str, Any]:
    """
    Map an entity to the relevant set of concepts/synonyms under a previously mapped field.
    If the mapped field does not contain concepts, returns a flag (e.g. for identifier or numeric columns).
    ------
    Args:
        item (Tuple): A tuple in format (entity, table, field).
    Returns:
        Dict with keys `concepts` and `reason`.
    """

    # Load concept table & embeddings
    data_dir = os.getenv("BEATAML_DATA_DIR", "../../BeatAML")
    concept_df = pd.read_csv(os.path.join(data_dir, "concept_table.csv"))  
    # cols: concept_name, table_name, field_name, concept_with_context

    #with open("BeatAML/concept_embeddings.pkl", "rb") as f:
    #    data = pickle.load(f)

    entity, table, field = item  # parse the input into entity, table and field
    subset = concept_df[
        (concept_df["table_name"] == table) &
        (concept_df["field_name"] == field)
    ].copy()
    if subset.empty:
        # default to the raw entity
        return {'entity': entity, 'table.field': f'{table}.{field}',
                'concepts': entity, 
                'reason': f'Concept mapping skipped as column `{table}.{field}` does not hold concepts.'}
    
    subset_unique = subset.drop_duplicates(subset=["concept_name"]).reset_index(drop=True)
    candidates = subset_unique["concept_name"].tolist()
    mapped_values, reason = _llm_choose_best_concept(entity, table, field, candidates)
    return {'entity': entity, 'table.field': f'{table}.{field}', 'concepts': mapped_values, 'reason': reason}

@tool
def get_relevant_field_values(items: List[List[str]]) -> Dict[str, Any]:
    """
    Map a list of (entity, table, field) tuples from previous `entity -> table.field` mappings to the most 
    relevant normalized concepts/values under the selected fields (entity->values).
    ------
    Args:
        items (list): A list of tuples. Each tuple has format (entity, table, field), which is to be mapped to field values.
    Returns:
        Each input item mapped to a dict with entity, table.field, and list of mapped concepts/values.

    Usage example:
    items = [('anticoagulant', 'Patient', 'treatment'),]
    result = get_relevant_field_values(items) 
    # {'entity':'anticoagulant', 'table.field':'Patient.treatment', 'concepts':['heparin','warfarin']}
    """
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(map_entity_to_concept, items))


# ----- Tool to convert structured criteria to SQL query -----
class SqlQuery(BaseModel):
    query: str = Field(..., description="Generated SQL query as a string")

@tool
def transform_query_to_sql(structured_input: str, feedback: str = '') -> str:
    """
    LLM-assisted conversion of structured logical criteria or data-fetching request to SQL format.
    ------
    Args:
        structured_input (str): Criteria or precise instructions about the data operation needed.
        feedback (str): Optional; any additional instructions/error messages from previous runs.
    Returns:
        str: Read-only SQL statement.
    """
    # Load the schema pk/fk mappings
    data_dir = os.getenv("BEATAML_DATA_DIR", "../../BeatAML")
    with open(os.path.join(data_dir, 'BeatAML_schema_keys.json'), 'r') as f:
        schema_keys = json.load(f)
     
    sql_prompt = f"""
        You're a specialized secure SQL query generator. 
        TASK:
        You will convert the provided instructions to a read-only SQL statement. 
        Make use of the added feedback, if provided.
        CRITICAL SECURITY CONSTRAINTS:
        1.  You MUST generate ONLY a single `SELECT` statement.
        2.  You MUST NOT generate any `UPDATE`, `INSERT`, `DELETE`, `DROP`, `TRUNCATE`, `ALTER`, or any other data-modifying or DDL statements.
        3.  You MUST NOT include any comments (`--`, `/*`) in the final SQL output.
        4.  The query must be read-only.
        QUERY LOGIC REQUIREMENTS:
        1. JOINS: Build multi-step joins as necessary to connect tables appearing in the input query. 
            - Make use of `schema_keys` dict to create correct joins.
        2. Row-Level Filters (WHERE):
            - Translate criteria with complex `AND`/`OR` logic into a standard `WHERE` clause.
            - Correctly handle `exclude` types by wrapping the condition in `NOT()`.
        3. Set-Based Filters (GROUP BY / HAVING):
            - If multiple `include` filters are provided as separate conditions, use `GROUP BY` and `HAVING` clauses for each respective condition.
        OUTPUT FORMAT:
        Return only a JSON object matching the schema: {{ "query": "<SQL STRING HERE>" }}
        The value of "query" MUST contain only the SQL SELECT statement as plain text 
        with no backticks, no quotes around the entire SQL, no markdown formatting, 
        no explanation. 
        Fetch all columns (`*`) from the root table and joined tables to ensure data is available for analysis. 
        
        INPUTS:
        Query: {structured_input}
        Feedback: {feedback}
        Schema primary/foreign keys: {schema_keys}
        """
    try:
        result = call_llm(
            user_prompt=sql_prompt,
            system_prompt="You are an expert SQL assistant.",
            response_model=SqlQuery,
        )
        return result.query
    except Exception as e:
        return f"SQL error: {e}"
    

# ----- Tool to execute SQL query and return summarized result -----
@tool
def run_sql_query(sql_input: str) -> Dict[str, Any]:
    """
    Executes SQL query against the DB to retrieve a cohort table.
    ------
    Args:
        sql_input (str): The read-only SQL SELECT statement to be executed. Strictly string only.
    Returns:
        Dict: Returned dataframe summary and first 5 rows as key/value pairs. 
    
    Note:
        The returned results table is saved in full to a CSV file named `returned_cohort_table.csv` in the session artifacts directory. 
    """
    atlas = Atlas(atlas_id=os.getenv("POLLY_ATLAS_ID", "beataml2"))
    artifacts_dir = os.environ.get("SESSION_ARTIFACTS_DIR", ".")
    
    try:
        result = atlas.query(sql_input)
        df = pd.DataFrame(result)
        save_path = os.path.join(artifacts_dir, 'returned_cohort_table.csv')
        df.to_csv(save_path, sep=',')
        print(f"\nData saved to {save_path}.")
        return {"summary": df.info(), "first_5_rows": df.head(5)}
    except Exception as e:
        print(f'\nError at SQL execution: {e}')
        return {"summary": None, "first_5_rows": None}


# ----- Tools for Analysis and Plotting -----

def validate_code_safety(code: str) -> Tuple[bool, str]:
    """
    Validates Python code using AST to ensure it only uses whitelisted libraries and functions.
    """
    # Whitelisted libraries
    ALLOWED_IMPORTS = {
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'sklearn', 'math', 'datetime', 'pysurvival'
    }
    
    # Banned functions/attributes
    BANNED_NODES = {
        'exec', 'eval', 'compile', 'open', 'input', '__import__',
        'subprocess', 'os', 'sys', 'shutil', 'requests', 'urllib', 'socket'
    }

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names = []
            if isinstance(node, ast.Import):
                module_names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_names = [node.module.split('.')[0]]
            
            for name in module_names:
                if name not in ALLOWED_IMPORTS:
                    return False, f"Importing '{name}' is not allowed. Allowed: {ALLOWED_IMPORTS}"

        # Check for banned function calls and attribute access
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BANNED_NODES:
                    return False, f"Function '{node.func.id}' is banned."
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in BANNED_NODES:
                    return False, f"Attribute/Method '{node.func.attr}' is banned."

    return True, "Code is safe."

@tool
def execute_analysis_code(code: str) -> str:
    """
    Executes Python code for data analysis and plotting.
    The code must only use safe libraries (pandas, numpy, matplotlib, seaborn, etc.).
    It should read data from 'returned_cohort_table.csv' in the artifacts directory.
    ------
    Args:
        code (str): The Python code to execute.
    Returns:
        str: Standard output and error from the execution.
    """
    artifacts_dir = os.environ.get("SESSION_ARTIFACTS_DIR", ".")
    
    # 1. Validate Safety
    is_safe, message = validate_code_safety(code)
    if not is_safe:
        return f"Security Violation: {message}"

    # 2. Prepare environment
    # Ensure plots directory exists
    plots_dir = os.path.join(artifacts_dir, "plots")
    try:
        os.makedirs(plots_dir, exist_ok=True)
    except Exception as e:
        return f"Error creating plots directory: {e}"
        
    # Ensure analysis_codes directory exists
    codes_dir = os.path.join(artifacts_dir, "analysis_codes")
    try:
        os.makedirs(codes_dir, exist_ok=True)
    except Exception as e:
        return f"Error creating analysis_codes directory: {e}"

    # 3. Write to file with backend setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{timestamp}.py"
    filepath = os.path.join(codes_dir, filename)
    
    # Prepend backend setup to force non-interactive mode and set working directory context
    # We need to tell the script where the CSV is and where to save plots
    setup_header = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n" 
        "import matplotlib.pyplot as plt\n"
        "import os\n"
        f"os.chdir('{os.path.abspath(artifacts_dir)}')\n" # Change CWD to artifacts dir so relative paths work
    )

    try:
        with open(filepath, "w") as f:
            f.write(setup_header + code)
    except Exception as e:
        return f"Error writing code to file: {e}"

    # 4. Execute with timeout
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=30  # 30 seconds timeout
        )
        output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
             output += f"\nProcess exited with code {result.returncode}"
        return output

    except subprocess.TimeoutExpired:
        return "Execution timed out after 30 seconds."
    except Exception as e:
        return f"Execution failed: {e}"

@tool
def check_analysis_feasibility(analysis_request: str) -> Dict[str, Any]:
    """
    Checks if the requested analysis can be performed on the fetched cohort data.
    It reads the 'returned_cohort_table.csv' from the artifacts directory and uses an LLM to validate
    if the necessary columns exist and contain sufficient non-null data.
    ------
    Args:
        analysis_request (str): The user's analysis request (e.g., "plot age distribution").
    Returns:
        Dict: A dictionary containing 'feasible' (bool), 'reason' (str), and 'columns_present' (list).
    """
    artifacts_dir = os.environ.get("SESSION_ARTIFACTS_DIR", ".")
    csv_path = os.path.join(artifacts_dir, 'returned_cohort_table.csv')
    
    if not os.path.exists(csv_path):
        return {"feasible": False, "reason": "No cohort data found. Please run a query to fetch data first.", "columns_present": []}
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"feasible": False, "reason": f"Error reading data file: {e}", "columns_present": []}
        
    columns = df.columns.tolist()
    head_data = df.head(3).to_markdown(index=False)
    data_info = df.info(buf=None) # Capture info string if possible, or just use dtypes
    dtypes = df.dtypes.to_dict()
    
    # Construct a prompt for the LLM to evaluate feasibility
    prompt = f"""
    You are a data analyst validator.
    
    TASK:
    Determine if the user's analysis request can be fulfilled by the available dataset.
    
    USER REQUEST: "{analysis_request}"
    
    DATASET SUMMARY:
    - Columns: {columns}
    - Data Types: {dtypes}
    - Sample Data (first 3 rows):
    {head_data}
    
    CRITERIA:
    1. Are the necessary columns for the analysis present?
    2. Is the data type suitable (e.g., numerical for histograms, categorical for bar charts)?
    3. If the request implies specific fields (e.g. "survival"), are they in the columns?
    
    OUTPUT:
    Return a JSON object with:
    - "feasible": boolean (true/false)
    - "reason": string (explanation of why it is or isn't feasible, mentioning missing columns if any)
    """
    
    class FeasibilityResponse(BaseModel):
        feasible: bool
        reason: str

    try:
        response = call_llm(
            user_prompt=prompt,
            model="gpt-4o-mini",
            system_prompt="You are a strict data validator.",
            response_model=FeasibilityResponse
        )
        return {
            "feasible": response.feasible,
            "reason": response.reason,
            "columns_present": columns
        }
    except Exception as e:
        return {"feasible": False, "reason": f"Validation failed: {e}", "columns_present": columns}

@tool
def save_process_summary(content: str, filename: str = "process_summary.md") -> str:
    """
    Saves a summary of the agent's process to a Markdown file.
    ------
    Args:
        content (str): The markdown content to save.
        filename (str): The name of the file (default: process_summary.md).
    Returns:
        str: Confirmation message.
    """
    artifacts_dir = os.environ.get("SESSION_ARTIFACTS_DIR", ".")
    
    if not filename.endswith(".md"):
        filename += ".md"
    
    save_path = os.path.join(artifacts_dir, filename)
    
    try:
        with open(save_path, "w") as f:
            f.write(content)
        return f"Summary saved to {save_path}"
    except Exception as e:
        return f"Error saving summary: {e}"
