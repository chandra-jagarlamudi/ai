import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

VECTORSTORE_PATH = "vectorstore"

st.title("RAG Demo - Modern LangChain")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

from pathlib import Path as _Path

_vs_path = _Path(VECTORSTORE_PATH)
if not (_vs_path / "index.faiss").is_file():
    st.warning("Vectorstore not found. Run `rag_init.ipynb` or `rag_build.ipynb` to build it first.")
    st.info(f"Expected location: `{_vs_path.resolve()}`")
    st.stop()

vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below:

{context}

Question: {question}
""")

query = st.text_input("Ask your question")

if query:

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": query
    })

    st.write(response)