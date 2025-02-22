from typing import Dict, List, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import logging
from ..config.config import get_settings
import uuid

settings = get_settings()

class VectorStore:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        try:
            # Create index if it doesn't exist
            if settings.PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
                pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
        except Exception as e:
            # Ignore index already exists error
            self.logger.info(f"Index already exists or error creating: {str(e)}")
        
        # Connect to index
        self.index = pc.Index(settings.PINECONE_INDEX_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )

    def _prepare_car_data(self, car_data: Dict[str, List[str]], car_model: str) -> List[Dict]:
        """
        Prepare car data for vectorization
        """
        documents = []
        
        for section, items in car_data.items():
            # Convert list items to text
            section_text = "\n".join(items)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(section_text)
            
            # Prepare documents with metadata
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "car_model": car_model,
                        "section": section,
                        "chunk_id": i
                    }
                })
        
        return documents

    async def upsert_car_data(self, car_data: Dict[str, List[str]], car_model: str, images: List[Dict] = None):
        """
        Upsert car data and optional images into Pinecone
        """
        try:
            # Prepare text documents
            documents = self._prepare_car_data(car_data, car_model)
            
            # Generate embeddings and upsert text data
            for i, doc in enumerate(documents):
                # Generate embedding
                embedding = await self.embeddings.aembed_query(doc["text"])
                
                # Create vector ID
                vector_id = f"{car_model}_{doc['metadata']['section']}_{doc['metadata']['chunk_id']}"
                
                # Upsert to Pinecone
                self.index.upsert(
                    vectors=[(
                        vector_id,
                        embedding,
                        {
                            "text": doc["text"],
                            **doc["metadata"],
                            "type": "text"
                        }
                    )]
                )
            
            # If images are provided, store them with text description embeddings
            if images:
                for image in images:
                    # Generate embedding from image description
                    description = f"{image['metadata']['type']} view of {car_model}: {image['metadata'].get('description', '')}"
                    embedding = await self.embeddings.aembed_query(description)
                    
                    vector_id = f"{car_model}_image_{image['metadata']['type']}_{uuid.uuid4()}"
                    self.index.upsert(
                        vectors=[(
                            vector_id,
                            embedding,
                            {
                                "image_data": image["image_data"],
                                **image["metadata"],
                                "car_model": car_model,
                                "description": description,
                                "type": "image"
                            }
                        )]
                    )
            
            self.logger.info(f"Successfully upserted data for {car_model}")
            
        except Exception as e:
            self.logger.error(f"Error upserting vectors: {str(e)}")
            raise

    async def search_car_data(
        self,
        query: str,
        car_model: Optional[str] = None,
        section: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search car data in Pinecone
        """
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Prepare filter
            filter_dict = {}
            if car_model:
                filter_dict["car_model"] = car_model
            if section:
                filter_dict["section"] = section
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "text": match.metadata["text"],
                    "car_model": match.metadata["car_model"],
                    "section": match.metadata["section"],
                    "score": match.score
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching vectors: {str(e)}")
            raise

    async def delete_car_data(self, car_model: str):
        """
        Delete all vectors for a specific car model
        """
        try:
            # Delete vectors with matching metadata
            self.index.delete(
                filter={
                    "car_model": car_model
                }
            )
            
            self.logger.info(f"Successfully deleted vectors for {car_model}")
            
        except Exception as e:
            self.logger.error(f"Error deleting vectors: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            self.logger.info(f"Index stats: {stats}")
            return {
                "total_vector_count": stats.total_vector_count,
                "namespaces": stats.namespaces,
                "dimension": stats.dimension
            }
        except Exception as e:
            self.logger.error(f"Error getting index stats: {str(e)}")
            raise

    async def search_images(
        self,
        query: str,
        car_model: Optional[str] = None,
        image_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search images using text description
        """
        try:
            # Generate query embedding from text
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Prepare filter
            filter_dict = {"type": "image"}
            if car_model:
                filter_dict["car_model"] = car_model
            if image_type:
                filter_dict["image_type"] = image_type
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "image_data": match.metadata["image_data"],
                    "car_model": match.metadata["car_model"],
                    "type": match.metadata["type"],
                    "description": match.metadata.get("description", ""),
                    "score": match.score
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching images: {str(e)}")
            raise 