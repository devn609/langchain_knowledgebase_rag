from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import langchain_community

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # take environment variables from .env (especially openai api key)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
DOCS_DIR = DATA_DIR / "docs"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


# GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

kb_file_path = DOCS_DIR / 'kb.csv'
vectordb_file_path = INDEX_DIR / "faiss_index"

##max_retries=6
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # Use the modern 2026 stable version
    google_api_key= GOOGLE_API_KEY,
    temperature=0.1,
    max_retries=6
)

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")

# Most modern models (like BGE or GTE) don't need the 'Instructor' wrapper
# and are faster/more accurate.
instructor_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5", # A modern 2026 favorite
    model_kwargs={'device': 'cpu'}
)

def create_vector_db():
    # Load data from KB data file

    ##loader = CSVLoader(file_path=kb_file_path, source_column="prompt")
    loader = CSVLoader(
        file_path=kb_file_path, source_column="prompt",
        encoding='cp1252'  # or 'latin1'
    )
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    ret = vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    langchain_community.debug=True
    # Load the existing vector database

    #vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create the retriever
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly say "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 4. Build the chain
    chain = RetrievalQA.from_chain_type(
                                        llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT},
                                        verbose=True
    )

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Where is your headquarters?"))