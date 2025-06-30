# ingest_with_langchain_batch.py (parallel embedding with ThreadPoolExecutor)

import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Read PDF
reader = PdfReader("sample.pdf")
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Chunk the text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"\n📄 Total Chunks: {len(chunks)}")

# Parallel embedding function
def embed_batch_parallel(batch_chunks):
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda chunk: embedding_model.embed_query(chunk), batch_chunks))
    return results

# Batch embed and upsert
batch_size = 20
for i in tqdm(range(0, len(chunks), batch_size), desc="🔄 Uploading to Pinecone"):
    batch_chunks = chunks[i:i+batch_size]
    batch_embeddings = embed_batch_parallel(batch_chunks)

    vectors = [{
        "id": f"chunk-{i+j}",
        "values": batch_embeddings[j],
        "metadata": {"text": batch_chunks[j]}
    } for j in range(len(batch_chunks))]

    index.upsert(vectors=vectors)

print("✅ All embeddings stored in Pinecone.")
