# ingest_with_langchain_sections.py

import os
import json
from uuid import uuid4
from tqdm import tqdm
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Load full PDF text
def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

# Use GPT to analyze PDF
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

def analyze_pdf(text):
    prompt = (
        """
        Given the following PDF content, do the following:

        1. Provide a 1-line summary.
        2. Provide a 15-line summary.
        3. Extract all major section headers with their respective section texts.
           Each section should be output as:
           {"title": "...", "content": "..."}

        Return the result as a JSON object with keys: one_line_summary, fifteen_line_summary, sections

        PDF Content:
        """ + text[:6000] + "..."
    )

    response = llm.invoke(prompt)
    try:
        json_data = json.loads(response.content.strip())
        return json_data
    except:
        raise ValueError("Unable to parse GPT output as JSON")

# Embed and store
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
vectorstore = LangchainPinecone(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    pinecone_api_key=PINECONE_API_KEY
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def chunk_text_with_summaries(section_title, section_text):
    chunks = []
    split_docs = text_splitter.create_documents([section_text])

    for doc in split_docs:
        chunk_text = doc.page_content

        prompt = f"""
        Summarize the following text in 1-2 sentences:

        {chunk_text}
        """
        summary_response = llm.invoke(prompt)
        chunk_summary = summary_response.content.strip()

        chunk_doc = Document(
            page_content=chunk_text,
            metadata={
                "section_title": section_title,
                "chunk_summary": chunk_summary
            }
        )
        chunks.append(chunk_doc)
    return chunks

def store_chunks(sections, one_line_summary, fifteen_line_summary, document_title="sample.pdf"):
    docs = []

    # Store 1-line and 15-line summary as separate documents
    summary_doc_1 = Document(
        page_content=one_line_summary,
        metadata={
            "type": "summary",
            "summary_level": "1-line",
            "document_title": document_title
        }
    )
    summary_doc_15 = Document(
        page_content=fifteen_line_summary,
        metadata={
            "type": "summary",
            "summary_level": "15-line",
            "document_title": document_title
        }
    )
    docs.extend([summary_doc_1, summary_doc_15])

    for section in sections:
        docs.extend(chunk_text_with_summaries(section["title"], section["content"]))

    print(f"📄 Total Chunks (including summaries): {len(docs)}")
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(vectorstore.add_documents, [[doc] for doc in docs]), total=len(docs)))

if __name__ == "__main__":
    text = load_pdf("sample.pdf")
    print("🧠 Analyzing PDF with GPT...")
    results = analyze_pdf(text)

    print("\n📌 1-line Summary:\n", results["one_line_summary"])
    print("\n📄 15-line Summary:\n", results["fifteen_line_summary"])

    print("\n🔄 Uploading chunks...")
    store_chunks(
        results["sections"],
        one_line_summary=results["one_line_summary"],
        fifteen_line_summary=results["fifteen_line_summary"],
        document_title="sample.pdf"
    )
    print("✅ All section-based chunks and summaries stored in Pinecone.")
