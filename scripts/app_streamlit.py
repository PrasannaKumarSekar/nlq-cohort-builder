# app.py
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from typing import List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import OllamaLLM
from typing import TypedDict, Annotated, List

from scripts.plotter import plot_custom_oncoplot  

# ========================= CONFIG =========================
DB_PATH = "../databases/BeatAML/BeatAML.db" 
LLM_MODEL = "llama3.2" 
TOP_N_GENES = 30

st.set_page_config(page_title="BeatAML NL Explorer", layout="centered")

# ========================= LLM =========================
@st.cache_resource
def get_llm():
    return OllamaLLM(model=LLM_MODEL, temperature=0.1, base_url="http://localhost:11434")

llm = get_llm()

# ========================= STATE =========================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    sql_code: str
    plot_b64: str
    plot_type: str
    interpretation: str
    error: Optional[str]

# ========================= HELPERS =========================
def get_base_survival_df():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT DISTINCT
        co.patient_id,
        co.overall_survival,
        co.vital_status,
        p.age_at_diagnosis,
        p.gender,
        s.eln2017_criteria
    FROM clinical_outcome co
    LEFT JOIN patient p ON co.patient_id = p.patient_id
    LEFT JOIN sample s ON co.patient_id = s.patient_id
    WHERE co.overall_survival IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['event'] = df['vital_status'].map({'Dead': 1, 'Alive': 0})
    df['time'] = pd.to_numeric(df['overall_survival'], errors='coerce')
    return df.dropna(subset=['time', 'event'])

def run_sql_safely(sql: str):
    if not sql.strip():
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        # First try EXPLAIN to catch syntax errors
        conn.execute("EXPLAIN " + sql)
        df = pd.read_sql_query(sql, conn)
        patients = df['patient_id'].dropna().unique().tolist()
        conn.close()
        return patients
    except Exception as e:
        conn.close()
        return None, str(e)

def generate_oncoplot_data(cohort_patients=None, top_n=TOP_N_GENES):
    conn = sqlite3.connect(DB_PATH)
    placeholders = ','.join(['?'] * len(cohort_patients)) if cohort_patients else None

    mut_query = """
    SELECT m.symbol, m.variant_classification, m.hgvsp_short, m.t_vaf,
           s.patient_id
    FROM mutation m
    JOIN sample s ON m.sample_id = s.sample_id
    WHERE m.t_vaf >= 0.05 AND m.symbol IS NOT NULL
    """
    if cohort_patients:
        mut_query += f" AND s.patient_id IN ({placeholders})"
    mutations = pd.read_sql_query(mut_query, conn, params=cohort_patients or ())

    fusion_query = """
    SELECT consensus_aml_fusions AS symbol, s.patient_id 
    FROM fusion f 
    JOIN sample s ON f.sample_id = s.sample_id 
    WHERE consensus_aml_fusions IS NOT NULL
    """
    if cohort_patients:
        fusion_query += f" AND s.patient_id IN ({placeholders})"
    fusions = pd.read_sql_query(fusion_query, conn, params=cohort_patients or ())
    fusions['variant_classification'] = 'Fusion'
    fusions['hgvsp_short'] = 'Fusion'

    conn.close()

    all_vars = pd.concat([
        mutations[['symbol', 'variant_classification', 'patient_id']],
        fusions[['symbol', 'variant_classification', 'patient_id']]
    ], ignore_index=True).dropna(subset=['symbol'])

    top_genes = all_vars['symbol'].value_counts().head(top_n).index

    def classify(row):
        vc = row['variant_classification']
        if 'Fusion' in vc:
            return 'Fusion'
        elif any(x in vc for x in ['Frame_Shift_Del', 'Frame_Shift_Ins']):
            return 'Frame_Shift'
        elif 'Nonsense' in vc or 'Stop' in vc:
            return 'Nonsense_Mutation'
        elif 'Splice' in vc:
            return 'Splice_Site'
        elif 'Missense' in vc or 'Substitution' in vc:
            return 'Missense_Mutation'
        return 'Multi_Hit' if (all_vars['patient_id'] == row['patient_id']).sum() > 1 else 'Missense_Mutation'

    all_vars['alteration'] = all_vars.apply(classify, axis=1)
    matrix = all_vars[all_vars['symbol'].isin(top_genes)].copy()
    matrix = matrix.rename(columns={'patient_id': 'sample', 'symbol': 'gene'})

    n_patients = matrix['sample'].nunique() if not matrix.empty else 0
    title = f"OncoPrint: Top {top_n} Altered Genes (n={n_patients})"
    if cohort_patients:
        title += " — Cohort"
    return matrix[['sample', 'gene', 'alteration']], title

def fig_to_b64():
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ========================= NODES =========================
def planner(state: AgentState):
    user_query = state["messages"][-1].content.lower()
    is_oncoplot = any(k in user_query for k in ["oncoplot", "oncoprint", "waterfall", "mutation landscape"])
    is_km = any(k in user_query for k in ["kaplan", "survival", "km ", "os ", "overall survival"])

    prompt = f"""
You are an expert on the BeatAML database. Return ONLY a valid SQLite query that returns patient_id column for the requested cohort.
No markdown, no explanations, no Python.

Examples:
TP53 mutated → SELECT DISTINCT s.patient_id FROM mutation m JOIN sample s ON m.sample_id = s.sample_id WHERE m.symbol = 'TP53' AND m.t_vaf >= 0.05
NPM1 mutated and FLT3-ITD negative → SELECT DISTINCT s.patient_id FROM ... (complex join logic)

User request: {state["messages"][-1].content}

SQL:"""

    sql = llm.invoke(prompt).strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    if sql.lower() in ["none", "all", "", "''", '""']:
        sql = ""

    print('Generated SQL', sql)
    return {
        **state,
        "sql_code": sql,
        "query": state["messages"][-1].content,
        "plot_type": "oncoplot" if is_oncoplot else "kaplan-meier",
        "error": None
    }

def sql_fixer(state: AgentState):
    prompt = f"""
Previous SQL failed with error: {state['error']}
Here is the broken SQL:
{state['sql_code']}

Fix it and return ONLY the corrected SQL query.
"""
    fixed = llm.invoke(prompt).strip()
    fixed = fixed.replace("```sql", "").replace("```", "").strip()
    return {**state, "sql_code": fixed, "error": None}

def executor(state: AgentState):
    st.write(f"Debug SQL: `{state['sql_code'][:200]}{'...' if len(state['sql_code'])>200 else ''}`")
    sql = state["sql_code"]
    print('sql state', sql)
    cohort_patients = None
    error = None

    if sql:
        result = run_sql_safely(sql)
        if isinstance(result, list):
            cohort_patients = result
        else:
            cohort_patients, error_msg = None, result
            return {**state, "error": error_msg}

    fig, ax = plt.subplots(figsize=(14, 10) if state["plot_type"] == "oncoplot" else (10, 7))

    if state["plot_type"] == "oncoplot":
        matrix, title = generate_oncoplot_data(cohort_patients)
        plot_custom_oncoplot(matrix, title=f"{title}\nQuery: {state['query']}")
    else:
        df = get_base_survival_df()
        if cohort_patients:
            mask1 = df['patient_id'].isin(cohort_patients)
            mask2 = ~mask1
            for label, mask in [("Cohort", mask1), ("Others", mask2)]:
                if mask.sum() >= 5:
                    kmf = KaplanMeierFitter()
                    kmf.fit(df[mask]['time'], df[mask]['event'], label=f"{label} (n={mask.sum()})")
                    ax = kmf.plot_survival_function(ci_show=True, ax=ax)
            add_at_risk_counts(*[kmf] * 2, ax=ax)
        else:
            for risk in ['Favorable', 'Intermediate', 'Adverse']:
                mask = df['eln2017_criteria'] == risk
                if mask.sum() >= 5:
                    kmf = KaplanMeierFitter()
                    kmf.fit(df[mask]['time'], df[mask]['event'], label=f"ELN {risk} (n={mask.sum()})")
                    ax = kmf.plot_survival_function(ci_show=True, ax=ax)
            add_at_risk_counts(*[kmf] * 3, ax=ax)

        plt.title(f"Overall Survival — {state['query']}", fontsize=14, pad=20)
        plt.xlim(0, 60)

    plot_b64 = fig_to_b64()
    plt.close(fig)

    return {**state, "plot_b64": plot_b64, "error": None}

def interpreter(state: AgentState):
    prompt = f"""
Write a concise, publication-ready figure legend (2–4 sentences) for the generated {state['plot_type'].replace('-', ' ')}.

User question was: {state['query']}
"""
    caption = llm.invoke(prompt).strip()
    return {**state, "interpretation": caption}

# ========================= GRAPH =========================
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner)
workflow.add_node("sql_fixer", sql_fixer)
workflow.add_node("executor", executor)
workflow.add_node("interpreter", interpreter)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    lambda x: "sql_fixer" if x.get("error") else "interpreter",
    {"sql_fixer": "sql_fixer", "interpreter": "interpreter"}
)
workflow.add_edge("sql_fixer", "executor")
workflow.add_edge("interpreter", END)

app = workflow.compile()

# ========================= STREAMLIT UI =========================
st.title("BeatAML Natural Language Explorer")
st.markdown("""
Ask anything in plain English → instantly get **OncoPrint** or **Kaplan-Meier** curves  
Powered by Llama 3.2 (local) + LangGraph + lifelines
""")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if "image" in msg:
                st.image(msg["image"], width='content')
            if "text" in msg:
                st.caption(msg["text"])

if prompt := st.chat_input("e.g. Show survival for NPM1 mutated and FLT3-ITD negative patients"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking + generating plot..."):
                try:
                    result = app.invoke(
                        {
                            "messages": st.session_state.messages,
                            "sql_code": "",
                            "query": "",
                            "plot_b64": "",
                            "plot_type": "kaplan-meier",
                            "interpretation": "",
                            "error": None,
                            "sql_fix_attempts": 0
                        },
                        config={
                            "recursion_limit": 15, 
                            "configurable": {"thread_id": "debug"}  
                        }
                    )
                except Exception as e:
                    if "recursion" in str(e).lower():
                        st.error("Infinite loop detected in SQL generation. Here's what went wrong:")
                        recent_msgs = st.session_state.messages[-10:]
                        sql_snippets = []
                        for msg in recent_msgs:
                            if isinstance(msg, AIMessage):
                                content = msg.content
                                if "SELECT" in content.upper() or "FROM" in content.upper():
                                    sql_snippets.append(content.strip())

                        if sql_snippets:
                            st.code("\n\n--- BAD SQL THAT CAUSED LOOP ---\n".join(sql_snippets[-3:]), language="sql")
                            st.warning("The LLM is repeatedly generating invalid/broken SQL. Common causes:\n"
                                    "• Wrong table/column names\n"
                                    "• Missing JOINs\n"
                                    "• Syntax errors (e.g. unbalanced quotes)\n"
                                    "• Trying to use PostgreSQL syntax on SQLite")
                        else:
                            st.info("No SQL found in recent messages. Try rephrasing your question more clearly.")
                        
                        result = {
                            "interpretation": "I got stuck trying to translate your question into a valid cohort query. "
                                            "Please try being more specific (e.g., 'TP53 mutated patients', "
                                            "'patients with RUNX1 mutations and adverse ELN risk', 'age > 70')."
                        }
                    else:
                        st.error(f"Unexpected error: {e}")
                        result = {"interpretation": "Something went wrong."}

                # === NOW DISPLAY RESULT AS BEFORE ===
                if result.get("error"):
                    st.error(f"Final SQL error: {result['error']}")
                elif not result.get("plot_b64"):
                    st.info(result.get("interpretation", "No plot generated."))
                else:
                    img_bytes = base64.b64decode(result["plot_b64"])
                    st.image(img_bytes, use_column_width=True)
                    st.caption(result["interpretation"])

                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "image": img_bytes,
                        "text": result["interpretation"]
                    })
                    st.session_state.messages.append(AIMessage(content=result["interpretation"]))