# NLQ Cohort Builder & Analysis Agent

This agentic system empowers users to build patient cohorts from the BeatAML database using natural language queries and automatically performs downstream data analysis and visualization.

## 🚀 Features

*   **Natural Language to SQL**: Translates complex biomedical questions (e.g., "male patients with AML over 50") into secure, optimized SQL queries.
*   **Cohort Retrieval**: Fetches real-time patient data from the Polly Atlas database.
*   **Automated Analysis**: Generates, validates, and executes Python code to analyze the retrieved cohort data.
*   **Visualization**: Automatically creates and saves plots (histograms, scatter plots, etc.) to visualize trends.
*   **Session Management**: Every session is isolated with a unique ID. All outputs—logs, data, code, plots, and summaries—are organized in a dedicated `artifacts/` directory.
*   **Safety First**: Generated analysis code is strictly validated using AST (Abstract Syntax Tree) parsing to whitelist safe libraries and ban dangerous operations.

## 🛠️ Prerequisites

*   **Python 3.10+**
*   **Key Libraries**:
    *   `langchain`, `langgraph`, `openai` (for Agentic workflow)
    *   `polly-python` (for Database access)
    *   `pandas`, `numpy`, `matplotlib`, `seaborn`, `pysurvival` (for Analysis)
    *   `rich` (for CLI UI)

## ⚙️ Setup

1.  **Install Dependencies**:
    Ensure you have the required packages installed.
    ```bash
    pip install pandas numpy matplotlib seaborn langchain langgraph openai polly-python rich
    ```

2.  **Environment Variables**:
    The agent requires access to OpenAI and Polly. Set the following variables in your environment:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export POLLY_AUTH_KEY="your-polly-key"
    export POLLY_ATLAS_ID="beataml2"  # Optional, defaults to beataml2
    export BEATAML_DATA_DIR="../../BeatAML" # Optional path to locally stored DB metadata files, defaults to ../../BeatAML
    ```

## 🖥️ Usage

Start the agent by running the main script:

```bash
python main.py
```

### Interactive Workflow
1.  **Query**: Enter your request at the prompt.
    *   *Example*: "Find all female patients with prior malignancy and plot their age distribution."
2.  **Validation**: The agent will parse your request, map terms to the database schema, and ask for your confirmation.
3.  **Data Retrieval**: It generates a read-only SQL query, fetches the data, and saves it to a CSV.
4.  **Analysis (Optional)**: If you asked for a plot or calculation, the agent generates Python code using `pandas` and `matplotlib`.
5.  **Review & Execute**: You review the code. Upon approval, the agent executes it locally after a sanity check.
6.  **Results**: Code & plots are saved, and a summary is generated.

## 📂 Output Structure

All session outputs are automatically saved in the `artifacts/` directory, organized by Session ID:

```text
artifacts/
└── session-<session_id>/
    ├── log_2023-10-27_10:00:00.txt    # Detailed execution logs
    ├── returned_cohort_table.csv      # The raw cohort data fetched from DB
    ├── process_summary.md             # A final readable summary of the session
    ├── analysis_codes/
    │   ├── analysis_20231027_103000.py # Saved analysis scripts
    │   └── analysis_20231027_103500.py 
    └── plots/
        ├── age_distribution.png       # Generated plots
        └── survival_curve.png
```

## 💡 Example Queries

*   **Cohort Selection**: "Filter for patients with FLT3 mutation."
*   **Data Exploration**: "Show me the distribution of gender for patients treated with Sorafenib."
*   **Complex Analysis**: "Get me a histogram of age for male patients diagnosed with AML who have a prior malignancy."

## 🛡️ Security

The analysis module uses a **Strict AST Whitelist** approach.
*   **Allowed**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `sklearn`, `math`, `datetime`.
*   **Blocked**: System calls (`os`, `sys`, `subprocess`), network requests (`requests`, `urllib`), and dangerous built-ins (`exec`, `eval`).

## 📁 Code Structure

*   **`main.py`**: The entry point. Initializes the LangGraph agent, sets up the session environment, and handles the chat loop.
*   **`tool_functions.py`**: Contains the tool definitions for schema inspection, SQL generation, data retrieval, and secure code execution.
*   **`default_system_prompt.py`**: Defines the agent's persona and the strict step-by-step workflow it must follow.

---
### To-Do's

* Expand the concept/value mapping step to use semantic vector search as well as direct LLM-based selection
* Aggregation of different field mapping strategies (currently implements only LLM-based field mapping followed by retrieving other semantically similar fields)

