import os
import argparse
from pathlib import Path
from opentelemetry import trace
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from config import get_logger, enable_telemetry
import random

# initialize logging object
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Initialize Azure OpenAI client
from openai import AzureOpenAI

chat = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

# Initialize Azure AI Search clients
search_client = SearchIndexClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
)

def get_available_indexes():
    """Get a list of available book indexes."""
    indexes = list(search_client.list_index_names())
    # Filter to only book indexes (assuming they all start with 'classic-')
    book_indexes = [idx for idx in indexes if idx.startswith('classic-')]
    return book_indexes

def search_index(query: str, index_name: str, k: int = 5):
    """Search the specified index with semantic search or keyword search based on query type."""
    index_search_client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=index_name,
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
    )
    
    # Determine search approach based on query
    use_semantic_search = not any(x in query.lower() for x in ["first line", "opening", "begin", "start"])
    
    if not use_semantic_search:
        # For first line queries, use keyword search with ordering by position
        results = index_search_client.search(
            search_text="",  # Empty to get all documents
            select=["id", "content", "title", "filepath"],
            top=1,
            order_by=["id asc"]  # Assuming id is organized by position in book
        )
    else:
        # Perform semantic search for normal queries
        results = index_search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="default",
            select=["id", "content", "title", "filepath"],
            top=k
        )
    
    results_list = list(results)
    
    # If semantic search returns no results, fall back to keyword search
    if not results_list and use_semantic_search:
        logger.info(f"Semantic search returned no results, falling back to keyword search for: {query}")
        results = index_search_client.search(
            search_text=query,
            select=["id", "content", "title", "filepath"],
            top=k
        )
        results_list = list(results)
    
    return results_list

def get_personality_greeting(personality: str) -> str:
    """Generate a unique greeting based on the selected personality."""
    greetings = {
        "classic_literature": [
            "Greetings, literary enthusiast! I'm your guide through the timeless world of classic literature. What would you like to explore today?",
            "Welcome to the realm of classic literature! Shall we dive into a particular book or discuss literary themes?",
            "Hello! I'm your classic literature companion. Would you like to analyze a specific work or compare different books?",
            "Greetings! As your literary guide, I can help you explore themes, characters, or historical context. What interests you today?"
        ],
        "philosopher": [
            "Greetings, seeker of wisdom! I'm here to explore the philosophical depths of classic literature. What profound questions shall we contemplate?",
            "Welcome! As a philosophical guide through literature, I can help you examine deeper meanings and existential themes. What shall we explore?",
            "Hello! I'm ready to engage in thoughtful discourse about the philosophical implications in classic works. What would you like to discuss?",
            "Greetings! Let's examine the philosophical underpinnings of classic literature together. What work or theme shall we analyze?"
        ],
        "storyteller": [
            "Welcome to the magical world of storytelling! I'm here to bring classic literature to life. What tale shall we explore today?",
            "Greetings, fellow story lover! I can help you discover the narrative wonders of classic literature. What story captures your interest?",
            "Hello! As your storytelling companion, I'm ready to weave the tales of classic literature. What would you like to hear about?",
            "Welcome! Let's embark on a journey through the pages of classic literature. What story would you like to explore?"
        ],
        "critic": [
            "Greetings! As your literary critic, I'm ready to provide detailed analysis of classic works. What shall we examine today?",
            "Welcome! I'm here to offer critical insights into classic literature. Which work or author would you like to discuss?",
            "Hello! As your literary analyst, I can help you understand the nuances of classic works. What would you like to explore?",
            "Greetings! Let's examine the literary merits and historical significance of classic works. What shall we analyze?"
        ]
    }
    
    # Get greetings for the selected personality or default to classic_literature
    personality_greetings = greetings.get(personality, greetings["classic_literature"])
    
    # Return a random greeting from the selected personality's list
    return random.choice(personality_greetings)

def generate_rag_response(user_query: str, contexts: list, personality: str = "classic_literature"):
    """Generate a response using Azure OpenAI with the retrieved contexts."""
    # If no query is provided, return a greeting
    if not user_query or user_query.strip() == "":
        return get_personality_greeting(personality)
    
    # Define different personality prompts
    personality_prompts = {
        "classic_literature": (
            "You are a helpful AI assistant with expert knowledge about classic literature. "
            "Below is information retrieved from classic books. Use it to answer the user's question as accurately "
            "and completely as possible. If you're asked about the first line, opening line, or beginning of a book, "
            "make sure to directly quote the first line from the retrieved content.\n\n"
        ),
        "philosopher": (
            "You are a philosophical AI assistant who analyzes classic literature through the lens of great thinkers. "
            "You provide deep insights and connect literary themes to philosophical concepts. "
            "Below is information retrieved from classic books. Use it to answer the user's question with philosophical depth "
            "and intellectual rigor.\n\n"
        ),
        "storyteller": (
            "You are a master storyteller AI assistant who brings classic literature to life through engaging narratives. "
            "You have a gift for making literary analysis entertaining and accessible. "
            "Below is information retrieved from classic books. Use it to answer the user's question with vivid storytelling "
            "and engaging explanations.\n\n"
        ),
        "critic": (
            "You are a literary critic AI assistant who provides detailed analysis and critique of classic literature. "
            "You examine themes, writing style, historical context, and literary devices. "
            "Below is information retrieved from classic books. Use it to answer the user's question with critical insight "
            "and scholarly analysis.\n\n"
        )
    }
    
    # Get the selected personality prompt or default to classic literature
    base_prompt = personality_prompts.get(personality, personality_prompts["classic_literature"])
    
    # Build system prompt with context information
    system_prompt = {
        "role": "system",
        "content": (
            base_prompt +
            "Important instructions:\n"
            "1. If the retrieved content contains the answer, provide it directly.\n"
            "2. For quotes, use the exact text from the source material.\n"
            "3. If the content does not fully answer the question, be clear about what you do know and what you don't.\n"
            "4. Only say you don't know if there is absolutely no relevant information in the retrieved content.\n\n"
            "Retrieved Content:\n"
        )
    }
    
    # Process and organize context by source
    book_contexts = {}
    for ctx in contexts:
        title = ctx.get("title", "Unknown")
        if title not in book_contexts:
            book_contexts[title] = []
        book_contexts[title].append(ctx)
    
    # Build retrieved text grouped by book title
    retrieved_text = ""
    for title, ctxs in book_contexts.items():
        retrieved_text += f"[Book: {title}]\n"
        for i, ctx in enumerate(ctxs, 1):
            content = ctx.get("content", "")
            filepath = ctx.get("filepath", "")
            chunk_id = ctx.get("id", f"Chunk {i}")
            retrieved_text += f"Passage {i} (ID: {chunk_id}): {content}\n\n"
    
    system_prompt["content"] += retrieved_text.strip()
    
    # Check if query is about first line
    if any(x in user_query.lower() for x in ["first line", "opening", "begin", "start"]):
        user_query = f"{user_query}\n\nPlease quote the exact first line if it appears in the retrieved passages."
    
    # Create messages array with system prompt and user query
    messages = [
        system_prompt,
        {"role": "user", "content": user_query}
    ]
    
    # Generate completion
    response = chat.chat.completions.create(
        model=os.environ.get("CHAT_MODEL", "gpt-4"),
        messages=messages,
        temperature=0.5,  # Lower temperature for more factual responses
        max_tokens=1024,
        top_p=1.0,
    )
    
    return response.choices[0].message.content

def interactive_cli():
    """Run an interactive RAG CLI."""
    print("📚 Book RAG CLI - Interactive Mode 📚")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'indexes' to list available book indexes")
    
    available_indexes = get_available_indexes()
    print(f"Available book indexes: {', '.join(available_indexes)}")
    
    if not available_indexes:
        print("No book indexes found. Please create some indexes first.")
        return
    
    # Default to searching all indexes
    active_indexes = available_indexes.copy()
    
    while True:
        print("\nCurrently searching in:", ", ".join(active_indexes))
        user_input = input("\nEnter your query: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'indexes':
            print(f"Available indexes: {', '.join(available_indexes)}")
            select = input("Select indexes (comma-separated) or 'all': ")
            if select.lower() == 'all':
                active_indexes = available_indexes.copy()
            else:
                selected = [idx.strip() for idx in select.split(',')]
                active_indexes = [idx for idx in selected if idx in available_indexes]
            continue
        
        print(f"\nSearching for: {user_input}...")
        
        # Check if query is targeting a specific book
        target_indexes = active_indexes.copy()
        for index in available_indexes:
            book_name = index.replace("classic-", "")
            if book_name.lower() in user_input.lower():
                target_indexes = [index]
                print(f"Query specifically mentions {book_name}, focusing on that index.")
                break
        
        all_contexts = []
        for index in target_indexes:
            try:
                results = search_index(user_input, index, k=2)
                if results:
                    all_contexts.extend(results)
                    print(f"✓ Found relevant content in {index}")
            except Exception as e:
                logger.error(f"Error searching index {index}: {str(e)}")
                print(f"Error searching index {index}")
                
        if not all_contexts:
            print("No relevant information found in the selected indexes.")
            continue
            
        print("\nGenerating response...")
        
        try:
            response = generate_rag_response(user_input, all_contexts)
            print("\n🤖 Response:")
            print(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            print(f"Error generating response: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Book RAG CLI")
    parser.add_argument(
        "--query", 
        type=str,
        help="One-time query (if not provided, interactive mode will be used)"
    )
    parser.add_argument(
        "--index", 
        type=str,
        help="Specific index to search (if not provided, all book indexes will be used)"
    )
    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable sending telemetry back to the project"
    )
    
    args = parser.parse_args()
    
    if args.enable_telemetry:
        enable_telemetry(True)
    
    if args.query:
        # One-time query mode
        index_name = args.index
        available_indexes = get_available_indexes()
        
        if not index_name:
            if not available_indexes:
                print("No book indexes found. Please create some indexes first.")
                return
                
            # Check if query is targeting a specific book
            active_indexes = available_indexes.copy()
            for index in available_indexes:
                book_name = index.replace("classic-", "")
                if book_name.lower() in args.query.lower():
                    active_indexes = [index]
                    print(f"Query specifically mentions {book_name}, focusing on that index.")
                    break
        else:
            active_indexes = [index_name]
            
        all_contexts = []
        for index in active_indexes:
            try:
                results = search_index(args.query, index, k=3)
                if results:
                    all_contexts.extend(results)
            except Exception as e:
                logger.error(f"Error searching index {index}: {str(e)}")
                
        if not all_contexts:
            print("No relevant information found for your query.")
            return
            
        response = generate_rag_response(args.query, all_contexts)
        print(response)
    else:
        # Interactive mode
        interactive_cli()

if __name__ == "__main__":
    main() 