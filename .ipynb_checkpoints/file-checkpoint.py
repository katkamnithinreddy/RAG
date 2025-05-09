import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
#from langchain.tools.python.tool import PythonREPLTool
from langchain.chains import RetrievalQA
from langchain.agents import Tool

def run_python_code(code: str):
    try:
        return str(eval(code))
    except Exception as e:
        return str(e)

PythonREPLTool = Tool(
    name="PythonREPL",
    func=run_python_code,
    description="Executes basic Python expressions passed as string input."
)

openai_api_key="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"
# Load and split documents
def load_documents():
    documents = []
    for filename in os.listdir("documents"):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join("documents", filename))
            documents.extend(loader.load())
    return documents

def prepare_vector_store():
    docs = load_documents()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db

# Tool: RAG-based Q&A
def rag_tool():
    db = prepare_vector_store()
    retriever = db.as_retriever(search_type="similarity", k=3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Tool: Python REPL
output = PythonREPLTool.run("2 + 2")

# Tool: Dictionary (Simple)
def define_tool_func(word):
    return f"Definition of '{word}' (placeholder): This would normally call a dictionary API."

# Streamlit UI
st.title("🔍 RAG-Powered Multi-Agent Q&A")
query = st.text_input("Ask your question:")

if query:
    st.write("**Routing Decision:**")
    
    if "calculate" in query.lower():
        st.info("Using Python Calculator Tool")
        output = python_tool.run(query)
        st.success(output)
    
    elif "define" in query.lower():
        st.info("Using Dictionary Tool")
        word = query.split("define")[-1].strip()
        output = define_tool_func(word)
        st.success(output)
    
    else:
        st.info("Using RAG-based Retrieval + LLM")
        qa = rag_tool()
        result = qa(query)
        st.subheader("🔎 Retrieved Chunks:")
        for doc in result['source_documents']:
            st.code(doc.page_content.strip()[:300])
        st.subheader("💬 Answer:")
        st.success(result['result'])
