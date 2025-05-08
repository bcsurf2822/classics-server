import os
import json
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

def main():
    project = AIProjectClient.from_connection_string(
        conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], 
        credential=DefaultAzureCredential()
    )
    
    embeddings_client = project.inference.get_embeddings_client()
    
    model_name = os.environ["EMBEDDINGS_MODEL"]
    
    # Sample user questions to demonstrate embeddings
    questions = [
        "What is Frankenstein about?",
        "Tell me about Mary Shelley's novel",
        "Who is the monster in Frankenstein?",
        "What happens in Moby Dick?",
        "Tell me about whale hunting in literature"
    ]
    
    print(f"\n{'=' * 50}")
    print(f"EMBEDDINGS MODEL DEMONSTRATION")
    print(f"Using model: {model_name}")
    print(f"{'=' * 50}\n")
    
   
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: '{question}'")
        
    
        embedding_result = embeddings_client.embed(model=model_name, input=question)
        embedding_vector = embedding_result.data[0].embedding
        
        print(f"Vector dimensions: {len(embedding_vector)}")
        
        sample = embedding_vector[:5]
        sample_str = ", ".join([f"{val:.6f}" for val in sample])
        print(f"Vector sample (first 5 dimensions): [{sample_str}, ...]")
        
        
        max_val = max(embedding_vector)
        min_val = min(embedding_vector)
        avg_val = sum(embedding_vector) / len(embedding_vector)
        
        print(f"Vector statistics:")
        print(f"  - Max value: {max_val:.6f}")
        print(f"  - Min value: {min_val:.6f}")
        print(f"  - Average value: {avg_val:.6f}")
        
        if i == 0:  # Save only first question for reference
            with open("sample_embedding.json", "w") as f:
                json.dump({
                    "question": question,
                    "model": model_name,
                    "embedding": embedding_vector,
                    "dimensions": len(embedding_vector)
                }, f, indent=2)
            print(f"Full embedding saved to 'sample_embedding.json'")
        
        print(f"{'-' * 40}")
    
    print("\nEmbedding Demo Complete!")
    print("This demo shows how the embedding model converts text questions into")
    print("vector representations that can be used for semantic search.")

if __name__ == "__main__":
    main() 