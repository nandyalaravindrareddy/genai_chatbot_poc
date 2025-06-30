# chatbot_with_metadata.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from langchain.chains import RetrievalQA

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Set up embedding model and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
vectorstore = LangchainPinecone(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    pinecone_api_key=PINECONE_API_KEY
)

# Set up language model
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def run_query(query):
    # Use `invoke` instead of deprecated `get_relevant_documents`
    docs = retriever.invoke(query)

    context = ""
    for doc in docs:
        context += f"\n\n---\nSection: {doc.metadata.get('section_title', 'N/A')}\nChunk Summary: {doc.metadata.get('chunk_summary', 'N/A')}\nContent: {doc.page_content}"

    prompt = f"""
    Use the following extracted chunks from a document to answer the question. Each chunk has a section title and a brief summary.

    {context}

    Question: {query}
    Answer:
    """

    response = llm.invoke(prompt)
    return response.content.strip()

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = run_query(query)
        print(f"\n💬 Answer: {answer}")
