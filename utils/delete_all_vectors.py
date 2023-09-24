import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

index = pinecone.Index(os.getenv("PINECONE_INDEX"))

namespace = ""

print(f"Deleting all vectors in namespace: {namespace}")

index.delete(delete_all=True, namespace=namespace)

print("All vectors deleted!")