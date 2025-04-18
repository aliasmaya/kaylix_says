import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_core.tools import tool

loader = UnstructuredLoader("./data/kb.docx")
kb_data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(kb_data)
embeddings = OllamaEmbeddings(model=os.getenv("MODEL"), base_url=os.getenv("BASE_URL"))
vector_store = FAISS.from_documents(chunks, embeddings)

@tool("kb")
def knowledge_base(query: str) -> dict:
    """Query the internal knowledge base.
    
    Args:
        query: The query string for the knowledge base.
    
    Returns:
        A dictionary with the results and metadata, empty if no relevant matches.
    """
    docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
    relevance_threshold = 0.8
    filtered_docs = [
        {"content": doc.page_content, "source": "kb.docx", "metadata": doc.metadata, "score": score}
        for doc, score in docs_with_scores
        if score < relevance_threshold
    ]
    
    return {
        "results": filtered_docs,
        "query": query,
        "is_empty": len(filtered_docs) == 0
    }