import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

print(pinecone.list_indexes())