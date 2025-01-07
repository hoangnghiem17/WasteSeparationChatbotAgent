from pydantic import BaseModel, Field
from typing import List, Union

# Validates text input for OpenAI API
class TextPayload(BaseModel):
    type: str = Field("text", description="Sets text type of API payload as text")
    text: str = Field(description="User text input")
    
# Validates images for OpenAI API
class ImagePayload(BaseModel):
    type: str = Field("image_url", description="Sets text type of API payload as image")
    image_url: dict = Field(description="Defines the URL to image in dictionary", example={"url": "data:image/jpeg;base64,<encoded_string>"})
   
# Validates OpenAI API call
class OpenAIRequest(BaseModel):
    model: str = Field("gpt-4o-mini", description="GPT-model used for API call")
    messages: List[Union[TextPayload, ImagePayload]] = Field(description="Combined API payload containing of text and image (optional)")

# Validates OpenAI API response
class OpenAIResponse(BaseModel):
    response: str = Field(description="Response of OpenAI API")
