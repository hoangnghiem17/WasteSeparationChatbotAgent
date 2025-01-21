from langchain_core.tools import tool
from services.image import encode_image

@tool
def process_image_tool(image_file: bytes) -> str:
    """
    Processes an uploaded image and returns a base64-encoded string.

    Args:
        image_file (bytes): The raw image file data uploaded by the user.

    Returns:
        str: Base64-encoded string of the image or an error message.
    """
    if not image_file:
        return "No image provided. Please upload an image."
    
    try:
        base64_image = encode_image(image_file)
        if base64_image:
            return f"Image successfully processed. Base64: {base64_image[:100]}..."  # Log partial base64 for debugging
        else:
            return "Failed to process the image. Please try again with a valid image."
    except Exception as e:
        return f"An error occurred while processing the image: {str(e)}"
