# AI Job Market Analyst

An AI-powered web app that answers plain English questions about data science salaries, roles, and career trends.

Built by a data engineer learning to apply pipeline skills to AI systems.

## What it does

Ask questions like:
- "What is the average salary for a data engineer?"
- "Which data science jobs pay the most?"
- "How does remote work affect salary?"
- "What is the salary difference between entry level and senior roles?"

## How it works

1. Loads real salary data from the Kaggle Data Science Salaries dataset
2. Splits and embeds the data into a FAISS vector store using OpenAI embeddings
3. Retrieves relevant records based on your question
4. Uses GPT-3.5 to generate a clear, specific answer

This is the same ETL logic used in traditional data engineering — just applied to AI retrieval instead of dashboards.

## Tech stack

- Python
- LangChain
- OpenAI API
- FAISS vector store
- Streamlit
- Pandas

## Project structure
```
rag-pipeline/
├── app.py              # Streamlit web app
├── rag.py              # Core RAG pipeline
├── scraper.py          # Data collection scripts
├── ds_salaries.csv     # Salary dataset
├── requirements.txt    # Dependencies
├── .env                # API keys (not committed)
└── .streamlit/
    └── config.toml     # Streamlit theme config
```

## Run locally

1. Clone the repo
2. Create a virtual environment and activate it
3. Install dependencies
4. Add your OpenAI API key to a `.env` file
5. Run the app
```bash
git clone https://github.com/anwar1962/rag-pipeline.git
cd rag-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env
streamlit run app.py
```

## What's next

- Add more data sources including BLS government salary data and O*NET skills data
- Build a fantasy sports AI analyst
- Deploy as a public SaaS tool

## Built in public

Follow my journey from data engineer to AI engineer on LinkedIn.