import base64
import logging

from PIL import Image
import io

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def encode_image(image_file):
    """
    Encode uploaded image to base64 format for OpenAI API.
    """
    try:
        logging.info("Starting image conversion.")
        image = Image.open(image_file)
        logging.info(f"Image opened successfully: {image.format}, Size: {image.size}, Mode: {image.mode}")
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=image.format)
        base64_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        
        logging.info("Image successfully converted to base64.")
        return base64_image
    
    except Exception as e:
        logging.error(f"Error processing image: {e}", exc_info=True)
        return None