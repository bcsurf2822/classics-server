import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from config import get_logger

# initialize logging object
logger = get_logger(__name__)

# create a project client using environment variables loaded from the .env file
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# create a vector embeddings client that will be used to generate vector embeddings
embeddings = project.inference.get_embeddings_client()

# use the project client to get the default search connection
search_connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
)

# Create a search index client using the search connection
# This client will be used to create and delete search indexes
index_client = SearchIndexClient(
    endpoint=search_connection.endpoint_url, credential=AzureKeyCredential(key=search_connection.key)
)

import pandas as pd
from azure.search.documents.indexes.models import (
    SemanticSearch,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    SearchIndex,
)


def create_index_definition(index_name: str, model: str) -> SearchIndex:
    dimensions = 1536  # text-embedding-ada-002
    if model == "text-embedding-3-large":
        dimensions = 3072

    # The fields we want to index. The "embedding" field is a vector field that will
    # be used for vector search.
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="filepath", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="url", type=SearchFieldDataType.String),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            # Size of the vector created by the text-embedding-ada-002 model.
            vector_search_dimensions=dimensions,
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    # The "content" field should be prioritized for semantic ranking.
    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            keywords_fields=[],
            content_fields=[SemanticField(field_name="content")],
        ),
    )

    # For vector search, we want to use the HNSW (Hierarchical Navigable Small World)
    # algorithm (a type of approximate nearest neighbor search algorithm) with cosine
    # distance.
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=1000,
                    ef_search=1000,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            ),
        ],
    )

    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index definition
    return SearchIndex(
        name=index_name,
        fields=fields,
        semantic_search=semantic_search,
        vector_search=vector_search,
    )


import re
import uuid

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    paragraphs = re.split(r'\n{2,}', text)
    return [p.strip() for p in paragraphs if len(p.strip()) > chunk_size]
    
def create_docs_from_txt(path: str, model: str) -> list[dict[str, any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    
    # Extract book title from filename
    book_title = os.path.basename(path).replace('.txt', '').title()

    items = []
    for i, chunk in enumerate(chunks):
        emb = embeddings.embed(input=chunk, model=model)
        doc = {
            "id": str(uuid.uuid4()),
            "content": chunk,
            "filepath": f"assets/{os.path.basename(path)}",
            "title": f"{book_title} - Chunk {i+1}",
            "url": f"/books/{book_title.lower().replace(' ', '-')}#chunk-{i+1}",
            "contentVector": emb.data[0].embedding,
        }
        items.append(doc)

    return items


def create_index_from_txt(index_name: str, file_path: str):
    try:
        index_client.delete_index(index_name)
        logger.info(f"🗑️  Deleted existing index '{index_name}'")
    except Exception:
        pass

    index_definition = create_index_definition(index_name, model=os.environ["EMBEDDINGS_MODEL"])
    index_client.create_index(index_definition)
    logger.info(f"📚 Created index '{index_name}'")

    docs = create_docs_from_txt(path=file_path, model=os.environ["EMBEDDINGS_MODEL"])

    search_client = SearchClient(
        endpoint=search_connection.endpoint_url,
        index_name=index_name,
        credential=AzureKeyCredential(search_connection.key),
    )

    search_client.upload_documents(docs)
    logger.info(f"✅ Uploaded {len(docs)} chunks to '{index_name}'")

def get_book_files(assets_dir: str) -> list[str]:
    """Get all the book files in the assets directory."""
    return [os.path.join(assets_dir, f) for f in os.listdir(assets_dir) 
            if f.endswith('.txt') and f != 'intent_mapping.prompty']

def create_index_for_all_books(index_name_prefix: str, assets_dir: str):
    """Create search indexes for all book files in the assets directory."""
    book_files = get_book_files(assets_dir)
    
    for book_file in book_files:
        book_name = os.path.basename(book_file).replace('.txt', '')
        book_index_name = f"{index_name_prefix}-{book_name}"
        logger.info(f"Creating index for {book_name}...")
        create_index_from_txt(book_index_name, book_file)
        logger.info(f"Completed index creation for {book_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-name",
        type=str,
        help="Index name prefix to use when creating the AI Search indexes",
        default=os.environ["AISEARCH_INDEX_NAME"],
    )
    parser.add_argument(
        "--text-file",  
        type=str,
        help="Path to plain text file (e.g., a book) for indexing",
        default=None,  # No default, will process all books if not specified
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        help="Directory containing book text files",
        default="python/assets",
    )
    parser.add_argument(
        "--all-books",
        action="store_true",
        help="Create indexes for all books in the assets directory",
    )
    args = parser.parse_args()
    
    if args.all_books:
        create_index_for_all_books(args.index_name, args.assets_dir)
    elif args.text_file:
        create_index_from_txt(args.index_name, args.text_file)
    else:
        logger.info("Please specify either --text-file or --all-books flag")