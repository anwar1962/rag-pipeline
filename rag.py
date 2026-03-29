print("Script started")

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

print("Loading salary data...")
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
Employee Location: {row['employee_residence']}
Company Size: {'Small' if row['company_size']=='S' else 'Medium' if row['company_size']=='M' else 'Large'}
Year: {row['work_year']}
"""
    documents.append(Document(page_content=content))

print(f"Loaded {len(documents)} job records")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

print("Building vectorstore...")
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

prompt = ChatPromptTemplate.from_template("""
You are a data science job market analyst with access to real salary data.
Answer questions with specific numbers, job titles, and insights.
Always mention salary ranges, experience levels, and remote work when relevant.

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

print("\nJob Market AI ready.\n")

questions = [
    "What is the average salary for a data engineer?",
    "Which jobs pay the most in data science?",
    "How does remote work affect salary?",
    "What is the salary difference between entry level and senior roles?",
]

for q in questions:
    print(f"Q: {q}")
    answer = chain.invoke(q)
    print(f"A: {answer}\n")