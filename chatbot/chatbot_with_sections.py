# chatbot_with_metadata.py (Enhanced for Domain Support)

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Set up embedding model and vector store
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)
vectorstore = LangchainPinecone(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    pinecone_api_key=PINECONE_API_KEY
)

# Use faster model with streaming output
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def run_query(query, domain_filter):
    # Build domain filter
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"domain": domain_filter.lower()}
        }
    )
    docs = retriever.invoke(query)

    context = ""
    for doc in docs:
        content = doc.page_content[:700]  # Truncate long content for faster processing
        context += f"\n\n---\nDocument: {doc.metadata.get('document_title', 'N/A')}\nSection: {doc.metadata.get('section_title', 'N/A')}\nChunk Summary: {doc.metadata.get('chunk_summary', 'N/A')}\nContent: {content}"

    prompt = f"""
    Use the following extracted chunks from domain '{domain_filter}' to answer the question.
    Each chunk includes document title, section title, and a brief summary.

    {context}

    Question: {query}
    Answer:
    """

    response = llm.invoke(prompt)
    return response.content.strip()

if __name__ == "__main__":
    domain = input("Enter domain to search within (e.g., health, law): ").strip()
    while True:
        query = input("\nAsk a question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break
        print("\n💬 Answer:")
        run_query(query, domain)  # Streaming output will be shown live
