import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def query_pinecone(question):
    # Embed the user question
    embedding_response = client.embeddings.create(
        input=question,
        model="text-embedding-ada-002"
    )
    query_embedding = embedding_response.data[0].embedding

    # Search Pinecone
    search_response = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    matches = search_response.matches
    if not matches:
        return "Sorry, I couldn't find anything relevant."

    # Combine all top match texts
    context = "\n".join([match.metadata['text'] for match in matches])

    # Ask GPT to answer using context
    prompt = f"""You are a helpful assistant. Based on the context below, answer the question.

Context:
{context}

Question: {question}
Answer:"""

    chat_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return chat_response.choices[0].message.content.strip()


# Main chatbot loop
if __name__ == "__main__":
    print("🧠 PDF Chatbot Ready! Ask your questions below.\nType 'exit' to quit.\n")
    while True:
        question = input("You: ")
        if question.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        answer = query_pinecone(question)
        print(f"\n🤖 Answer:\n{answer}\n")
