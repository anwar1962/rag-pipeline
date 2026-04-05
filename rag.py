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
salary_df = pd.read_csv("ds_salaries.csv")
experience_map = {"EN": "Entry level", "MI": "Mid level", "SE": "Senior", "EX": "Executive"}
employment_map = {"FT": "Full time", "PT": "Part time", "CT": "Contract", "FL": "Freelance"}

salary_docs = []
for _, row in salary_df.iterrows():
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
    salary_docs.append(Document(page_content=content, metadata={"source": "salary"}))

print(f"Loaded {len(salary_docs)} salary records")

print("Loading O*NET career data...")
onet_df = pd.read_csv("onet_occupations.csv")

onet_docs = []
for _, row in onet_df.iterrows():
    content = f"""
Occupation: {row['title']}
O*NET Code: {row['code']}
Description: {row['description']}
Key Skills: {row['skills']}
Knowledge Areas: {row['knowledge']}
Typical Tasks: {row['tasks']}
Bright Outlook: {'Yes - strong job growth expected' if row['bright_outlook'] else 'Average growth'}
Search Keyword: {row['keyword']}
"""
    onet_docs.append(Document(page_content=content, metadata={"source": "onet"}))

print(f"Loaded {len(onet_docs)} O*NET occupation records")

all_docs = salary_docs + onet_docs
print(f"Total documents: {len(all_docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)
print(f"Created {len(chunks)} chunks")

print("Building vectorstore...")
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

prompt = ChatPromptTemplate.from_template("""
You are an expert career and job market analyst with access to real salary data and 
detailed O*NET occupation data. Answer questions with specific numbers, skills, 
tasks, and insights. Mention salary ranges, required skills, and job outlook when relevant.

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

print("\nCareer Intelligence AI ready.\n")

questions = [
    "What skills do I need to become a data engineer?",
    "What is the average salary for a data scientist?",
    "Which data careers have the best job growth outlook?",
    "What does a data engineer actually do day to day?",
]

for q in questions:
    print(f"Q: {q}")
    answer = chain.invoke(q)
    print(f"A: {answer}\n")