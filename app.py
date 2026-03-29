import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="AI Job Market Analyst",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
.block-container { padding-top: 3.5rem; max-width: 800px; }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F8EF7, #A259FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #8B8FA8;
        font-size: 1rem;
        margin-top: 0.3rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #1c1f2e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        border: 1px solid #2e3250;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4F8EF7;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #8B8FA8;
        margin-top: 0.2rem;
    }
    .suggestion-label {
        color: #8B8FA8;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .answer-box {
        background: #1c1f2e;
        border-left: 4px solid #4F8EF7;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-top: 1rem;
    }
    .footer {
        text-align: center;
        color: #444;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #1c1f2e;
    }
    div.stButton > button {
        background-color: #1c1f2e;
        color: #c0c4d8;
        border: 1px solid #2e3250;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        width: 100%;
        text-align: left;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #252942;
        border-color: #4F8EF7;
        color: #fff;
    }
    div.stTextInput > div > div > input {
        background-color: #1c1f2e;
        color: #e0e0e0;
        border: 1px solid #2e3250;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.95rem;
    }
    div.stTextInput > div > div > input:focus {
        border-color: #4F8EF7;
        box-shadow: 0 0 0 2px rgba(79,142,247,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def build_chain():
    df = pd.read_csv("ds_salaries.csv")
    experience_map = {"EN": "Entry level", "MI": "Mid level", "SE": "Senior", "EX": "Executive"}
    employment_map = {"FT": "Full time", "PT": "Part time", "CT": "Contract", "FL": "Freelance"}
    documents = []
    for _, row in df.iterrows():
        content = f"""
Job Title: {row['job_title']}
Experience Level: {experience_map.get(row['experience_level'], row['experience_level'])}
Employment Type: {employment_map.get(row['employment_type'], row['employment_type'])}
Salary (USD): ${row['salary_in_usd']:,}
Remote Ratio: {row['remote_ratio']}% remote
Company Location: {row['company_location']}
Company Size: {'Small' if row['company_size']=='S' else 'Medium' if row['company_size']=='M' else 'Large'}
Year: {row['work_year']}
"""
        documents.append(Document(page_content=content))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    prompt = ChatPromptTemplate.from_template("""
You are a job market analyst. Answer with specific numbers, job titles, and insights.
Always mention salary ranges and experience levels when relevant.

Data:
{context}

Question: {question}
""")

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, len(df)

st.markdown('<p class="hero-title">AI Job Market Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Ask anything about data science salaries, roles, and career trends</p>', unsafe_allow_html=True)

with st.spinner("Loading job market data..."):
    chain, total_records = build_chain()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-card"><div class="stat-number">607</div><div class="stat-label">Salary records</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card"><div class="stat-number">50+</div><div class="stat-label">Job titles</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><div class="stat-number">50+</div><div class="stat-label">Countries</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

question = st.text_input("", placeholder="e.g. What is the average salary for a data engineer?")

st.markdown('<p class="suggestion-label">Try one of these:</p>', unsafe_allow_html=True)
suggestions = [
    "What is the average salary for a data engineer?",
    "Which jobs pay the most in data science?",
    "How does remote work affect salary?",
    "Salary difference between entry level and senior?",
]

cols = st.columns(2)
for i, s in enumerate(suggestions):
    if cols[i % 2].button(s):
        question = s

if question:
    with st.spinner("Analyzing job market data..."):
        answer = chain.invoke(question)
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Built with LangChain · OpenAI · FAISS · Streamlit &nbsp;|&nbsp; Data: Kaggle DS Salaries Dataset</div>', unsafe_allow_html=True)