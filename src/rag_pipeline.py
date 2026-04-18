"""
Main RAG pipeline for legal document retrieval and generation
"""
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import os
from config.settings import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    CHROMA_DB_PATH,
    EMBEDDING_MODEL
)


class LegalRAG:
    """Retrieval-Augmented Generation system for Malaysian law"""
    
    def __init__(self):
        """Initialize RAG system"""
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBEDDING_MODEL
        )
        
        self.llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.1  # Low temperature for factual accuracy
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.vector_store = None
    
    def create_collection(self, collection_name: str = "malaysian_law"):
        """Create a ChromaDB collection"""
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created collection: {collection_name}")
        except Exception as e:
            print(f"Collection may already exist: {e}")
            self.collection = self.client.get_collection(name=collection_name)
    
    def add_documents(self, documents: list):
        """Add documents to the vector store"""
        if not self.collection:
            self.create_collection()
        
        # Format documents for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            ids.append(doc.get("chunk_id", f"chunk_{i}"))
            texts.append(doc.get("content", ""))
            metadatas.append({
                "act": doc.get("act", ""),
                "section": doc.get("section", ""),
                "source": doc.get("source", "")
            })
        
        # Add to ChromaDB
        embeddings = self.embeddings.embed_documents(texts)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"Added {len(documents)} documents to collection")
    
    def search_documents(self, query: str, top_k: int = 5) -> list:
        """Search for relevant documents"""
        if not self.collection:
            print("Collection not initialized")
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        return results
    
    def generate_answer(self, query: str, context: str = "") -> str:
        """Generate an answer using the LLM with context"""
        
        system_prompt = """You are a Malaysian Legal Research Assistant.
1. Use ONLY the provided context to answer.
2. Every claim MUST include a citation (e.g., 'According to Section 4 of the PDPA 2010...').
3. If the answer is not in the context, state that you cannot find the relevant statute.
4. Do not offer legal advice, only legal information.
"""
        
        if context:
            full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"
        else:
            full_prompt = f"{system_prompt}\n\nQuestion: {query}"
        
        response = self.llm(full_prompt)
        return response
    
    def query(self, question: str, top_k: int = 5) -> dict:
        """Complete RAG query: retrieve and generate"""
        
        # Retrieve relevant documents
        search_results = self.search_documents(question, top_k)
        
        # Prepare context
        context = ""
        sources = []
        
        if search_results['documents']:
            for i, doc_list in enumerate(search_results['documents']):
                for doc in doc_list:
                    context += f"\n{doc}"
                if search_results['metadatas'][i]:
                    sources.extend(search_results['metadatas'][i])
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "relevant_docs": search_results['documents']
        }
