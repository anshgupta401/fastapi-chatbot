from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from typing import List,Tuple
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware






load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key



app = FastAPI()
memory = {
    "docs":None, "history":[], "vector_store": None
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4




def get_answer(question:str, docs, history:List[Tuple[str,str]], top_k: int = 4):
    history_text = ""
    for user,bot in history:
        history_text += f"User: {user} \n Assistant: {bot} \n"
    model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
    prompt = PromptTemplate(
            input_variables=["context", "question","history"],
            template=(
                "You are a news analyst. What is your thoughts on the articles. "
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

def create_embeddings(docs):
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store





@app.post("/process_url")
def insert_url(req: URLRequest):
    try:
        loader = UnstructuredURLLoader(urls = [req.url])
        docs = loader.load()
        vector_stores = create_embeddings(docs)
        memory["docs"] = docs
        memory["history"] = []
        memory["vector_store"] = vector_stores
        return {"status":"success", "message":"website is loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code= 500, detail = str(e))

@app.post("/ask")
def ask_question(req:QueryRequest):
    if not memory["docs"]:
        raise HTTPException(status_code=404, detail="No URL loaded, Please call the process API")
    try:
        retriever = memory["vector_store"].as_retriever(search_kwargs={"k": req.top_k})
        ret_docs = retriever.get_relevant_documents(req.question)
        answer = get_answer(req.question, ret_docs, memory["history"], top_k=req.top_k)
        memory["history"].append((req.question, answer))
        return {"answer": answer, "question": req.question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

app.mount("/", StaticFiles(directory = "static", html = True), name = "static")
