import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

index = pinecone.Index(os.getenv("PINECONE_INDEX"))

stats = index.describe_index_stats()

namespaces = stats['namespaces']

for key, value in zip(namespaces.keys(), namespaces.values()):
    if key != "":
        print(f"Namespace: {key} => Vectors: {value}")
    else:
        print(f"Namespace: No namespace => Vectors: {value}")

print(f"Total vectors: {stats['total_vector_count']}")