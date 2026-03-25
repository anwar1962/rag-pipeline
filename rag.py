print("Script started")

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("Step 2: imports done")
load_dotenv()
print("Step 3: env loaded")

loader = TextLoader("data.txt")
documents = loader.load()
print("Step 4: document loaded")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print("Step 5: chunks created:", len(chunks))

print("Step 6: creating vectorstore...")
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
print("Step 7: vectorstore created")

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below:
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

print("Step 8: asking question...")
answer = chain.invoke("What is this document about?")
print("Answer:", answer)