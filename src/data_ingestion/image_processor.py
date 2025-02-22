from typing import Dict, BinaryIO
import io
from PIL import Image
import base64

class ImageProcessor:
    def __init__(self):
        self.max_size = (800, 800)  # Maximum image dimensions
    
    def process_image(self, image_file: BinaryIO, metadata: Dict) -> Dict:
        """Process image and prepare for storage"""
        try:
            # Read and preprocess image
            image = Image.open(image_file).convert('RGB')
            
            # Resize if image is too large while maintaining aspect ratio
            image.thumbnail(self.max_size)
            
            # Convert image to base64 for storage
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Convert dimensions tuple to string
            width, height = image.size
            dimensions = f"{width}x{height}"
            
            return {
                "image_data": img_str,
                "metadata": {
                    **metadata,
                    "dimensions": dimensions,  # Store as string instead of tuple
                    "type": "image"
                }
            }
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def encode_image_query(self, text_query: str) -> list:
        """Encode text query for image search"""
        inputs = self.processor(text=text_query, return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs)
        return text_features.detach().numpy().tolist()[0] 