# src/rag/pdf_loader.py
from __future__ import annotations
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def load_and_chunk_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
