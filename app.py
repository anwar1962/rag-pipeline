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
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

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
def build_agent():
    experience_map = {"EN": "Entry level", "MI": "Mid level", "SE": "Senior", "EX": "Executive"}

    salary_df = pd.read_csv("ds_salaries.csv")
    onet_df = pd.read_csv("onet_occupations.csv")

    salary_docs = []
    for _, row in salary_df.iterrows():
        content = f"""
Job Title: {row['job_title']}
Experience Level: {experience_map.get(row['experience_level'], row['experience_level'])}
Salary (USD): ${row['salary_in_usd']:,}
Remote Ratio: {row['remote_ratio']}% remote
Company Location: {row['company_location']}
Company Size: {'Small' if row['company_size']=='S' else 'Medium' if row['company_size']=='M' else 'Large'}
Year: {row['work_year']}
"""
        salary_docs.append(Document(page_content=content))

    onet_docs = []
    for _, row in onet_df.iterrows():
        content = f"""
Occupation: {row['title']}
Description: {row['description']}
Key Skills: {row['skills']}
Knowledge Areas: {row['knowledge']}
Typical Tasks: {row['tasks']}
Bright Outlook: {'Yes' if row['bright_outlook'] else 'No'}
"""
        onet_docs.append(Document(page_content=content))

    salary_store = FAISS.from_documents(salary_docs, OpenAIEmbeddings())
    onet_store = FAISS.from_documents(onet_docs, OpenAIEmbeddings())

    @tool
    def search_salaries(query: str) -> str:
        """Search real salary data for data science and engineering roles."""
        docs = salary_store.similarity_search(query, k=15)
        return "\n".join([d.page_content for d in docs])

    @tool
    def search_careers(query: str) -> str:
        """Search O*NET government occupation data for skills and career information."""
        docs = onet_store.similarity_search(query, k=10)
        return "\n".join([d.page_content for d in docs])

    @tool
    def calculate_salary_stats(job_title: str) -> str:
        """Calculate salary statistics for a specific job title."""
        filtered = salary_df[salary_df['job_title'].str.lower().str.contains(job_title.lower())]
        if filtered.empty:
            return f"No salary data found for {job_title}"
        return str({
            "count": len(filtered),
            "average": f"${filtered['salary_in_usd'].mean():,.0f}",
            "median": f"${filtered['salary_in_usd'].median():,.0f}",
            "min": f"${filtered['salary_in_usd'].min():,.0f}",
            "max": f"${filtered['salary_in_usd'].max():,.0f}",
        })

    tools = [search_salaries, search_careers, calculate_salary_stats]
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    system_prompt = """You are an expert career and job market analyst. You ONLY answer questions about careers, salaries, job skills, and professional development.

STRICT RULES:
- ONLY answer questions about careers, jobs, salaries, and skills
- For ANY other topic — sports, news, prices, weather, politics — say: "I can only help with career and job market questions. Try asking me about salaries, skills, or career paths."
- Never use your training knowledge to answer factual questions outside careers
- Always use your tools first before answering — never guess salary numbers
- When searching skills for data roles use terms like: "data engineer", "database", "computer systems"
- When searching AI roles use: "machine learning", "software developer", "data scientist"
- Always combine salary stats AND career skills in career transition answers
- If context from previous conversation is lost tell the user to repeat their background

When answering career transitions always cover:
1. Current role salary using calculate_salary_stats
2. Target role salary using closest matching title
3. Skills gap from O*NET data
4. Growth outlook"""
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent, len(salary_df), len(onet_df)
st.markdown('<p class="hero-title">AI Job Market Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Ask anything about data science salaries, roles, and career trends</p>', unsafe_allow_html=True)

with st.spinner("Loading career intelligence agent..."):
    agent, salary_count, onet_count = build_agent()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{salary_count}</div><div class="stat-label">Salary records</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{onet_count}</div><div class="stat-label">Occupations</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><div class="stat-number">3</div><div class="stat-label">AI tools</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if st.session_state.chat_history:
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.last_question = ""
        st.rerun()

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f'<div style="text-align:right;margin:8px 0"><span style="background:#1c1f2e;padding:8px 14px;border-radius:12px;font-size:14px;color:#e0e0e0">{msg.content}</span></div>', unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f'<div class="answer-box">{msg.content}</div>', unsafe_allow_html=True)

st.markdown('<p class="suggestion-label">Try asking:</p>', unsafe_allow_html=True)
suggestions = [
    "What skills do I need to become a data engineer?",
    "Which data science jobs pay the most?",
    "Which careers have the best job growth outlook?",
    "What does a data engineer do day to day?",
]

cols = st.columns(2)
selected = None
for i, s in enumerate(suggestions):
    if cols[i % 2].button(s, key=f"btn_{i}"):
        selected = s

with st.form(key="chat_form", clear_on_submit=True):
    question = st.text_input("", placeholder="e.g. I am a data engineer — what should I be earning?")
    submitted = st.form_submit_button("Ask")

final_question = selected if selected else (question if submitted else None)

if final_question and final_question != st.session_state.last_question:
    st.session_state.last_question = final_question
    st.session_state.chat_history.append(HumanMessage(content=final_question))

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        try:
            with st.spinner("Agent is thinking..."):
                result = agent.invoke({"messages": st.session_state.chat_history})
                full_response = result["messages"][-1].content

            words = full_response.split()
            for i, word in enumerate(words):
                full_response_so_far = " ".join(words[:i+1])
                response_container.markdown(
                    f'<div class="answer-box">{full_response_so_far}</div>',
                    unsafe_allow_html=True
                )
                import time
                time.sleep(0.05)

            st.session_state.chat_history.append(AIMessage(content=full_response))
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown('<div class="footer">Built with LangChain · LangGraph · OpenAI · FAISS · Streamlit &nbsp;|&nbsp; Data: Kaggle · O*NET</div>', unsafe_allow_html=True)