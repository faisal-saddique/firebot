import pinecone

from dotenv import load_dotenv  # For loading environment variables from .env file
import os
import time

# Load environment variables from .env file
load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

pinecone.delete_index(os.getenv("PINECONE_INDEX"))

time.sleep(6)

# pinecone.create_index(os.getenv("PINECONE_INDEX"), dimension=1536, 
#                       metric='cosine', 
#                       pods=1, 
#                       replicas=1, 
#                       pod_type='p1.x1')

# create the index
pinecone.create_index(
    name=os.getenv("PINECONE_INDEX"),
    dimension=1536,  # dimensionality of dense model
    metric="dotproduct",  # sparse values supported only for dotproduct
    pod_type="'p1.x1'",
    metadata_config={"indexed": []},  # see explaination above
)