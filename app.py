import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import shutil
from pathlib import Path
import uuid
from typing import List, Optional

from create_search_index import create_index_from_txt
from book_rag_cli import get_available_indexes, search_index, generate_rag_response
from config import get_logger


from azure.search.documents import SearchClient
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

load_dotenv()

logger = get_logger(__name__)

app = FastAPI(title="Books Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ASSETS_DIR = "python/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)


STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)


project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# Get the embeddings client for vector search
embeddings = project.inference.get_embeddings_client()

# Get the search connection
search_connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
)

def process_file_in_background(file_path: str, index_name: str):
    """Background task to process the file and create search index"""
    try:
        create_index_from_txt(index_name, file_path)
        logger.info(f"Successfully processed {file_path} into index {index_name}")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")

# Upload book to Search Index
@app.post("/upload-book")
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    index_name: str = None
):
    """
    Upload a text file and add it to the search index.
    If index_name is not provided, one will be generated.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(400, detail="Only .txt files are supported")
    
    # Generate a unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = Path(ASSETS_DIR) / unique_filename
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Generate index name if not provided
    if not index_name:
        base_name = os.path.basename(file.filename).replace(".txt", "").lower().replace(" ", "-")
        index_name = f"{os.environ.get('AISEARCH_INDEX_NAME', 'books')}-{base_name}"
    
    # Process the file in background to avoid blocking the request
    background_tasks.add_task(process_file_in_background, str(file_path), index_name)
    
    return {
        "filename": file.filename,
        "saved_as": unique_filename,
        "index_name": index_name,
        "status": "Processing started in background"
    }


# Get available book indexes
@app.get("/book-indexes")
async def get_book_indexes():
    """Get list of available book indexes for searching"""
    try:
        indexes = get_available_indexes()
        return {"indexes": indexes}
    except Exception as e:
        logger.error(f"Error getting book indexes: {str(e)}")
        raise HTTPException(500, detail=f"Error retrieving book indexes: {str(e)}")

# Search books API endpoint
@app.post("/search-books")
async def search_books(
    query: str, 
    index_name: Optional[str] = None,
    limit: int = 5
):
    """
    Search books with the given query.
    If index_name is not provided, searches across all available book indexes.
    """
    try:
        all_contexts = []
        active_indexes = [index_name] if index_name else get_available_indexes()
        
        # Check if query is targeting a specific book
        if not index_name:
            for index in active_indexes:
                book_name = index.replace("classic-", "")
                if book_name.lower() in query.lower():
                    active_indexes = [index]
                    logger.info(f"Query specifically mentions {book_name}, focusing on that index.")
                    break
        
        for index in active_indexes:
            try:
                results = search_index(query, index, k=limit)
                if results:
                    all_contexts.extend(results)
            except Exception as e:
                logger.error(f"Error searching index {index}: {str(e)}")
        
        if not all_contexts:
            return {"results": [], "message": "No relevant information found for your query."}
        
        # Generate RAG response
        response = generate_rag_response(query, all_contexts)
        
        return {
            "results": all_contexts,
            "response": response,
            "indexes_searched": active_indexes
        }
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        raise HTTPException(500, detail=f"Error processing search request: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    """Serve the index.html file"""
    return FileResponse(f"{STATIC_DIR}/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 