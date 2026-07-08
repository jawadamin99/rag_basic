import os
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

DATA_DIR = "data"
STORAGE_DIR = "storage" #Vector DB

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 6

PROMPT_TEMPLATE = """
You are an AI Agent
Make sure to answer all the questions from the **given document**.

If the answer is not available in the provided data, You must state
"I dont have the answer to this particular question"

Context:
{context} 

Question:
{question}

Answer in 150 words
"""

os.makedirs('data', exist_ok=True)
os.makedirs('storage', exist_ok=True)
