# ingest_pdf_sections_with_langchain_into_pinecone.py (Multi-PDF Version + Section Titles + 1-line/15-line Summary + Domain Support)

import os
import re
import json
import sys
import glob
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)
vectorstore = LangchainPinecone(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    pinecone_api_key=PINECONE_API_KEY
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def load_pdf_text(path):
    reader = PdfReader(path)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

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
    return json.loads(response.content.strip())

def chunk_and_summarize(section_title, section_text, doc_title, domain):
    chunks = text_splitter.create_documents([section_text])
    chunk_docs = []
    for chunk in chunks:
        summary_prompt = f"Summarize the following text in 1-2 sentences:\n\n{chunk.page_content}"
        summary = llm.invoke(summary_prompt).content.strip()
        doc = Document(
            page_content=chunk.page_content,
            metadata={
                "document_title": doc_title,
                "section_title": section_title,
                "chunk_summary": summary,
                "domain": domain
            }
        )
        chunk_docs.append(doc)
    return chunk_docs

def process_pdf(pdf_path, domain):
    print(f"📄 Processing {pdf_path}...")
    text = load_pdf_text(pdf_path)
    doc_title = os.path.basename(pdf_path)

    print(f"🧠 Asking LLM to analyze {doc_title} ...")
    results = analyze_pdf(text)

    one_line_summary = results["one_line_summary"]
    fifteen_line_summary = results["fifteen_line_summary"]
    sections = results["sections"]

    summary_doc_1 = Document(
        page_content=one_line_summary,
        metadata={
            "type": "summary",
            "summary_level": "1-line",
            "document_title": doc_title,
            "domain": domain
        }
    )
    summary_doc_15 = Document(
        page_content=fifteen_line_summary,
        metadata={
            "type": "summary",
            "summary_level": "15-line",
            "document_title": doc_title,
            "domain": domain
        }
    )

    all_docs = [summary_doc_1, summary_doc_15]
    for section in sections:
        print(f"✂️  Section: {section['title']}")
        all_docs.extend(chunk_and_summarize(section["title"], section["content"], doc_title, domain))

    print(f"🔄 Uploading {len(all_docs)} chunks from {doc_title} to Pinecone...")
    with ThreadPoolExecutor() as executor:
        list(executor.map(lambda d: vectorstore.add_documents([d]), all_docs))
    print(f"✅ {doc_title} uploaded.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest_pdf_sections_with_langchain_into_pinecone.py <domain> <pdf_path or directory>...")
        print("eg:python ingest_pdf_sections_with_langchain_into_pinecone.py health /Users/swethanandyala/mywork/iam_automation/projects/pinecone_pdf_chatbot/data/")
        exit(1)

    domain_arg = sys.argv[1].strip().lower()
    inputs = sys.argv[2:]
    pdf_files = []

    for input_path in inputs:
        if os.path.isdir(input_path):
            pdf_files.extend(glob.glob(os.path.join(input_path, "*.pdf")))
        elif input_path.endswith(".pdf"):
            pdf_files.append(input_path)
        else:
            print(f"⚠️  Skipping unsupported input: {input_path}")

    for pdf_path in pdf_files:
        process_pdf(pdf_path, domain=domain_arg)
