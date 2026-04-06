from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
import pandas as pd

load_dotenv()

print("Loading data...")
salary_df = pd.read_csv("ds_salaries.csv")
onet_df = pd.read_csv("onet_occupations.csv")

experience_map = {"EN": "Entry level", "MI": "Mid level", "SE": "Senior", "EX": "Executive"}

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

print("Building vector stores...")
salary_store = FAISS.from_documents(salary_docs, OpenAIEmbeddings())
onet_store = FAISS.from_documents(onet_docs, OpenAIEmbeddings())
print("Vector stores ready")

@tool
def search_salaries(query: str) -> str:
    """Search real salary data for data science and engineering roles.
    Use this to answer questions about compensation, pay ranges, and salary trends."""
    docs = salary_store.similarity_search(query, k=15)
    return "\n".join([d.page_content for d in docs])

@tool
def search_careers(query: str) -> str:
    """Search O*NET government occupation data for skills, tasks, and career information.
    Use this to answer questions about job requirements, skills needed, and career outlooks."""
    docs = onet_store.similarity_search(query, k=10)
    return "\n".join([d.page_content for d in docs])

@tool
def calculate_salary_stats(job_title: str) -> str:
    """Calculate average, min, and max salary statistics for a specific job title."""
    filtered = salary_df[salary_df['job_title'].str.lower().str.contains(job_title.lower())]
    if filtered.empty:
        return f"No salary data found for {job_title}"
    return str({
        "job_title": job_title,
        "count": len(filtered),
        "average": f"${filtered['salary_in_usd'].mean():,.0f}",
        "median": f"${filtered['salary_in_usd'].median():,.0f}",
        "min": f"${filtered['salary_in_usd'].min():,.0f}",
        "max": f"${filtered['salary_in_usd'].max():,.0f}",
    })

tools = [search_salaries, search_careers, calculate_salary_stats]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

system_prompt = """You are an expert career and job market analyst with access to 
real salary data and O*NET government occupation data.

Always use multiple tools to give complete, data-backed answers.
When asked about a career or salary question:
1. Use calculate_salary_stats for specific numbers
2. Use search_salaries for context and trends  
3. Use search_careers for skills and requirements
Combine all results into a comprehensive answer."""

agent = create_react_agent(llm, tools, prompt=system_prompt)

print("\nCareer Intelligence Agent ready with memory.")
print("Type 'quit' to exit\n")

chat_history = []

while True:
    question = input("You: ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        break
    if not question:
        continue
    try:
        chat_history.append(HumanMessage(content=question))
        result = agent.invoke({"messages": chat_history})
        messages = result["messages"]
        answer = messages[-1].content
        chat_history.append(AIMessage(content=answer))
        print(f"\nAgent: {answer}\n")
    except Exception as e:
        print(f"Error: {e}\n")