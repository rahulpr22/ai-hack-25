import base64
import logging
import os
import tempfile
from io import BytesIO
from typing import List, Dict

import aiohttp
import fal_client
from PIL import Image


class VideoProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Disable noisy HTTP client logs
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)

        # Ensure the videos directory exists in the root
        self.videos_dir = os.path.join(os.getcwd(), "videos")  # Root directory
        os.makedirs(self.videos_dir, exist_ok=True)

    def _resize_image_if_needed(self, image_data: str) -> BytesIO:
        """
        Resizes the image if its minimum dimension is less than 300px.
        Returns a BytesIO object of the resized image.
        """
        try:
            img_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_bytes))

            width, height = img.size
            if min(width, height) < 300:
                # Resize while maintaining aspect ratio
                scale_factor = 300 / min(width, height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                img = img.resize(new_size, Image.LANCZOS)
                self.logger.info(f"Resized image to {new_size}")

            # Convert image back to bytes
            img_buffer = BytesIO()
            img.save(img_buffer, format="JPEG")
            img_buffer.seek(0)

            return img_buffer

        except Exception as e:
            self.logger.error(f"Error resizing image: {str(e)}")
            raise

    def _upload_image_to_fal(self, image_data: str) -> str:
        """Upload base64 image to fal.ai and get URL"""
        try:
            img_buffer = self._resize_image_if_needed(image_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(img_buffer.getvalue())
                temp_file_path = temp_file.name
            url = fal_client.upload_file(temp_file_path)
            os.remove(temp_file_path)
            if not url:
                raise Exception("Failed to get URL from fal.ai upload")
            return url
        except Exception as e:
            self.logger.error(f"Error uploading image to fal: {str(e)}")
            raise

    async def _generate_transition_video(self, start_url: str, end_url: str, description: str,
                                         product_name: str) -> str:
        handler = None
        try:
            prompt = f"A smooth transition showcasing {description}, moving naturally between views"
            handler = await fal_client.submit_async(
                "fal-ai/minimax/video-01/image-to-video",
                arguments={"prompt": prompt, "image_url": start_url, "target_image_url": end_url},
            )

            if not handler:
                raise Exception("Failed to submit async request to fal.ai")

            video_url = None
            async for event in handler.iter_events(with_logs=True):
                self.logger.info(event)  # Log all events
                if isinstance(event, dict):
                    if event.get('error'):
                        raise Exception(f"Error from fal.ai: {event['error']}")
                    # Check for video URL in the event
                    if 'video' in event and event['video'].get('url'):
                        video_url = event['video']['url']
                        break

            if not video_url:
                # Try getting result one more time
                result = await handler.get()
                if result and 'video' in result and result['video'].get('url'):
                    video_url = result['video']['url']
                else:
                    raise Exception("No video URL received from fal.ai")

            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download video, HTTP {response.status}")
                    video_content = await response.read()

            video_path = os.path.join(self.videos_dir, f"{product_name.replace(' ', '_')}.mp4")
            with open(video_path, "wb") as video_file:
                video_file.write(video_content)

            self.logger.info(f"✅ Video saved to {video_path}")
            return base64.b64encode(video_content).decode()

        except Exception as e:
            self.logger.error(f"❌ Error generating transition video: {str(e)}")
            raise

    async def create_video_from_images(self, images: List[Dict], product_name: str, duration_per_image: int = 3) -> str:
        try:
            if not images:
                raise ValueError("No images provided")
            if len(images) == 1:
                image_url = self._upload_image_to_fal(images[0]["image_data"])
                return await self._generate_transition_video(image_url, image_url,
                                                             images[0].get("metadata", {}).get("description",
                                                                                               "A cinematic video"),
                                                             product_name)
            image_urls = []
            for img in images:
                image_urls.append(self._upload_image_to_fal(img["image_data"]))
            transition_videos = []
            for i in range(len(image_urls) - 1):
                description = images[i].get("metadata", {}).get("description", "car showcase")
                video_data = await self._generate_transition_video(image_urls[i], image_urls[i + 1], description,
                                                                   product_name)
                transition_videos.append(video_data)
            final_transition = await self._generate_transition_video(
                image_urls[-1], image_urls[0], images[-1].get("metadata", {}).get("description", "car showcase"),
                product_name)
            transition_videos.append(final_transition)
            if len(transition_videos) > 1:
                self.logger.warning(
                    "Multiple video concatenation not implemented yet - returning first transition only")
            return transition_videos[0]
        except Exception as e:
            self.logger.error(f"Error creating video sequence: {str(e)}")
            raise
