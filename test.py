from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec








load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

pinecone_api = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
use_pinecone = True



pc = Pinecone(api_key = pinecone_api)
if use_pinecone and index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = "cosine",
        spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
    )

index = pc.Index(index_name)

stats = index.describe_index_stats()
print(stats)
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
text = "What is the main topic of this news article?"
query_vector = embeddings.embed_query(text)
namespace = index_name
result = index.query(vector = query_vector, top_k = 5, include_metadata = True, namespace = "website")
if result.get("matches"):
    for match in result["matches"]:
        print(f"Metadata: {match['metadata']}")
        print(f"Score: {match['score']}")
else:
    print("No match found")

