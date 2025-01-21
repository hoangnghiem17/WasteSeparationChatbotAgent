import base64
import logging

from PIL import Image
import io

from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@tool
def encode_image(image_file):
    """
    Encode uploaded image to base64 format.
    
    Args:
        image_file: uploaded file from a Flask request
        
    Returns:
        str or None: A base64-encoded string representation of the image
    """
    try:
        logging.info("Starting image conversion.")
        image = Image.open(image_file)
        logging.info(f"Image opened successfully: {image.format}, Size: {image.size}, Mode: {image.mode}")
        
        # Saves image to in-memory buffer and encodes raw byte image data to base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=image.format)
        base64_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        
        logging.info("Image successfully converted to base64.")
        return base64_image
    
    except Exception as e:
        logging.error(f"Error processing image: {e}", exc_info=True)
        return None