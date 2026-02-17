import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "sample_mcp")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "blog_posts")
index_name = os.getenv("MONGODB_INDEX_NAME", "vector_index")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
