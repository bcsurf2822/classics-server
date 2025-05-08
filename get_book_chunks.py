import os
from pathlib import Path
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from config import ASSET_PATH, get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

chat = project.inference.get_chat_completions_client()
embeddings = project.inference.get_embeddings_client()

search_connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
)

search_index_client = SearchIndexClient(
    endpoint=search_connection.endpoint_url,
    credential=AzureKeyCredential(key=search_connection.key),
)

from azure.search.documents.models import VectorizedQuery

def visualize_embedding(embedding_vector, dimensions=5):
    """
    Visualize a small sample of the embedding vector for display purposes.
    
    Args:
        embedding_vector: The full embedding vector
        dimensions: Number of dimensions to display
    
    Returns:
        A string representation of the embedding sample
    """
    # Extract the first few dimensions for visualization
    sample = embedding_vector[:dimensions]
    
    # Format the sample for display
    sample_str = ", ".join([f"{val:.6f}" for val in sample])
    
    # Add information about the vector
    info = (
        f"Embedding vector sample (first {dimensions} of {len(embedding_vector)} dimensions):\n"
        f"[{sample_str}, ...]\n"
        f"Vector shape: {len(embedding_vector)} dimensions"
    )
    
    return info

@tracer.start_as_current_span(name="get_book_chunks")
def get_book_chunks(messages: list, context: dict = None) -> dict:
    if context is None:
        context = {}

    overrides = context.get("overrides", {})
    top = overrides.get("top", 5)
    
    search_query = messages[0]["content"]
    logger.debug(f"🧠 Intent mapping: {search_query}")

    # generate a vector representation of the search query
    embedding = embeddings.embed(model=os.environ["EMBEDDINGS_MODEL"], input=search_query)
    search_vector = embedding.data[0].embedding
    
    # Visualize the embedding and add to context
    embedding_visualization = visualize_embedding(search_vector)
    logger.debug(f"Embedding visualization:\n{embedding_visualization}")
    
    if "thoughts" not in context:
        context["thoughts"] = []
    
    context["thoughts"].append(
        {
            "title": "Generated search query",
            "description": search_query,
        }
    )
    
    context["thoughts"].append(
        {
            "title": "Embedding visualization",
            "description": embedding_visualization,
        }
    )
    
    # Get all available indexes
    available_indexes = [index.name for index in search_index_client.list_indexes()]
    logger.debug(f"Available indexes: {available_indexes}")
    
    all_documents = []
    
    # Search through each available index
    for index_name in available_indexes:
        try:
            # Create a client for this specific index
            current_search_client = SearchClient(
                index_name=index_name,
                endpoint=search_connection.endpoint_url,
                credential=AzureKeyCredential(key=search_connection.key),
            )
            
            # search the index for documents matching the search query
            vector_query = VectorizedQuery(vector=search_vector, k_nearest_neighbors=top, fields="contentVector")
            
            search_results = current_search_client.search(
                search_text=search_query, 
                vector_queries=[vector_query], 
                select=["id", "content", "filepath", "title", "url"]
            )
            
            # Process results from this index
            index_documents = []
            for result in search_results:
                try:
                    doc = {
                        "id": result["id"],
                        "content": result["content"],
                        "filepath": result.get("filepath", ""),
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "source_index": index_name
                    }
                    index_documents.append(doc)
                except Exception as e:
                    logger.warning(f"Error processing result from index {index_name}: {e}")
            
            logger.debug(f"📄 {len(index_documents)} documents retrieved from index '{index_name}'")
            all_documents.extend(index_documents)
            
        except Exception as e:
            logger.warning(f"Error searching index '{index_name}': {e}")
    
    # add results to the provided context
    if "grounding_data" not in context:
        context["grounding_data"] = []
    context["grounding_data"].append(all_documents)

    logger.debug(f"📄 Total: {len(all_documents)} documents retrieved across all indexes")
    return all_documents

if __name__ == "__main__":
    import logging
    import argparse

    # set logging level to debug when running this module directly
    logger.setLevel(logging.DEBUG)

    # load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Query to use to search product",
        default="What Books are Available to read?",
    )

    args = parser.parse_args()
    query = args.query

    # Create a context object to store thoughts and grounding data
    context = {}
    result = get_book_chunks(messages=[{"role": "user", "content": query}], context=context)
    
    # Display embedding visualization when run directly
    for thought in context.get("thoughts", []):
        if thought.get("title") == "Embedding visualization":
            print("\n" + thought["description"])
    
    print(f"\nFound {len(result)} documents across all indexes")