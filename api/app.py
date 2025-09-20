from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from typing import List, Optional
import json

# Initialize FastAPI app
app = FastAPI(title="RAG API", version="1.0.0")

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Initialize Qdrant vector store
try:
    # Try to create collection (will fail if it exists)
    vectorstore = Qdrant.from_texts(
        texts=[""],  # Empty initial text
        embedding=embeddings,
        url="http://qdrant:6333",
        collection_name="docs"
    )
except Exception:
    # Collection exists, use it directly
    vectorstore = Qdrant(
        embedding=embeddings,
        url="http://qdrant:6333",
        collection_name="docs"
    )

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3", base_url="http://ollama:11434")

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Pydantic models
class QueryInput(BaseModel):
    query: str

class TextInput(BaseModel):
    text: str

class DocumentResponse(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentResponse]

class AddResponse(BaseModel):
    status: str
    message: str
    text_length: int

# Endpoints
@app.post("/qa", response_model=QueryResponse)
async def ask_question(input_data: QueryInput):
    """Ask a question and get an answer based on the documents"""
    try:
        result = qa_chain.invoke({"query": input_data.query})
        
        # Get source documents
        docs = vectorstore.similarity_search(input_data.query, k=3)
        sources = []
        for doc in docs:
            sources.append(DocumentResponse(
                content=doc.page_content,
                metadata=doc.metadata
            ))
        
        return QueryResponse(
            answer=result["result"],
            sources=sources
        )
    except Exception as e:
        return QueryResponse(
            answer=f"Error: {str(e)}",
            sources=[]
        )

@app.post("/add", response_model=AddResponse)
async def add_document(input_data: TextInput):
    """Add a document to the vector store"""
    try:
        if not input_data.text:
            return AddResponse(
                status="error",
                message="No text provided",
                text_length=0
            )
        
        document = Document(page_content=input_data.text, metadata={"source": "user_input"})
        vectorstore.add_documents([document])
        
        return AddResponse(
            status="success",
            message=f"Document added: '{input_data.text[:50]}{'...' if len(input_data.text) > 50 else ''}'",
            text_length=len(input_data.text)
        )
    except Exception as e:
        return AddResponse(
            status="error",
            message=f"Error: {str(e)}",
            text_length=0
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG API",
        "version": "1.0.0",
        "endpoints": {
            "qa": "POST /qa - Ask questions",
            "add": "POST /add - Add documents",
            "docs": "GET /docs - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG API"}