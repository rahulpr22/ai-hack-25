from pinecone import Pinecone
from typing import List, Dict
import os
from dotenv import load_dotenv
import openai

class VectorStore:
    def __init__(self):
        load_dotenv()
        
        # Initialize OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )
        
        # Connect to index
        self.index = self.pc.Index("imagine-dragons")

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI"""
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def store_document(self, text: str, metadata: Dict) -> None:
        """Store document in Pinecone using OpenAI embeddings"""
        vector_id = f"{metadata['source']}_{metadata['chunk_id']}"
        
        # Generate embeddings using OpenAI
        embedding = self.get_embedding(text)
        
        # Store in Pinecone
        self.index.upsert(
            vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }]
        ) 