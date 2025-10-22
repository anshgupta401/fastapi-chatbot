from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from typing import List,Tuple
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import hashlib



load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

pinecone_api = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
use_pinecone = True

app = FastAPI()
memory = {
    "docs":None, "history":[], "vector_store": None, "namespace":None
}



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pc = Pinecone(api_key = pinecone_api)
if use_pinecone and index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = "cosine",
        spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
    )





class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

def generate_namespace(url:str)->str:
    return hashlib.md5(url.encode()).hexdigest()[:10]




def get_answer(question:str, docs, history:List[Tuple[str,str]], top_k: int = 4):
    history_text = ""
    for user,bot in history:
        history_text += f"User: {user} \n Assistant: {bot} \n"
    model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
    prompt = PromptTemplate(
            input_variables=["context", "question","history"],
            template=(
                "You are a article analyst. What is your thoughts on the articles. "
                "Use the following context to answer the question.\n\n"
                "Conversations so Far:\n{history}\n\n"
                "Context:\n{context}\n\n"
                "Users New Question:\n{question}\n\n"
                "Answer:"
            )
        )    
    chain = load_qa_chain(model,chain_type = "stuff", prompt = prompt)
    response = chain(
        {"input_documents":docs, "question": question,'history':history_text}
        , return_only_outputs=True)
    return response["output_text"]

def create_embeddings(docs, namespace:str):
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    if use_pinecone:
        vector_store = PineconeVectorStore.from_documents(docs, embedding = embeddings, index_name = index_name, namespace = namespace)
    else:
        vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def query_from_pinecone(query_text: str, namespace:str, top_k: int =4):
    pc = Pinecone(api_key = pinecone_api)
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    query_vector = embeddings.embed_query(query_text)
    result = index.query(vector = query_vector, top_k = 5, include_metadata = True, namespace = namespace)
    docs = []
    for match in result.get("matches", []):
        text = match["metadata"].get("text", "")
        metadata = match["metadata"]
        docs.append(Document(page_content = text, metadata = metadata))

    return docs







@app.post("/process_url")
def insert_url(req: URLRequest):
    try:
        print(type(req.url))
        namespace = generate_namespace(req.url)
        loader = UnstructuredURLLoader(urls = [req.url])
        docs = loader.load()
        vector_stores = create_embeddings(docs,namespace)
        memory["docs"] = docs
        memory["history"] = []
        memory["vector_store"] = vector_stores
        memory["namespace"] = namespace
        return {"status":"success", "message":"website is loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code= 500, detail = str(e))

@app.post("/ask")
def ask_question(req:QueryRequest):
    if not memory["docs"]:
        raise HTTPException(status_code=404, detail="No URL loaded, Please call the process API")
    try:
        namespace = memory.get("namespace")
        ret_docs = query_from_pinecone(req.question,namespace = namespace, top_k = req.top_k) 
        if not ret_docs:
            return {"No content found in Pinecone"}           
        answer = get_answer(req.question, ret_docs, memory["history"], top_k=req.top_k)
        memory["history"].append((req.question, answer))
        print(memory["vector_store"])
        return {"answer": answer, "question": req.question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/clear")
def clear_cache():
    try:
        memory["history"].clear()
        memory["docs"] =  None
        memory["vector_store"] = None
        memory["namespace"] = None
        return {"status": "success", "message": "chat memory cleared"}
    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))

app.mount("/", StaticFiles(directory = "static", html = True), name = "static")
