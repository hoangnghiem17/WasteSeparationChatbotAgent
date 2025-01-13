import os
import logging
from werkzeug.datastructures import FileStorage

import openai
from pydantic import ValidationError
from dotenv import load_dotenv

from services.image import encode_image
from models.models import ImagePayload, OpenAIPayload, OpenAIRequest, OpenAIResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(conversation_history: list, image_file: FileStorage =None, system_prompt: str = None, context: str = "") -> OpenAIResponse:
    """
    Makes API call to OpenAI multimodal LLM with optional image input.
    
    Args:
        conversation_history (List[dict]): List of dictionaries with role-content pairs of conversation history 
        image_file (FileStorage, optional): Uploaded image as FileStorage object
        system_prompt (str, optional): Custom system prompt to use for the API call.
        context (str, optional): Retrieved context to include in the system prompt.
    Returns:
        OpenAIResponse: Response from OpenAI LLM as validated response model.
    """
    try:
        logging.info("call_llm() invoked with conversation history.")
               
        # Combine context with the system prompt
        if context:
            system_prompt = f"{context}\n\n{system_prompt}" if system_prompt else context
            
        # Encode image if provided
        base64_image = None
        if image_file:
            base64_image = encode_image(image_file)

        # Connverts conversation history into list of OpenAIPayload objects
        payload = [
            OpenAIPayload(role=msg['role'], content=msg['content']).model_dump()
            for msg in conversation_history
        ]

        # Prepend system prompt to the payload
        if system_prompt:
            payload.insert(0, OpenAIPayload(role="system", content=system_prompt).model_dump())
        
        # Append image payload if image is uploaded
        if base64_image:
            image_payload = ImagePayload(
                image_url={"url": f"data:image/jpeg;base64,{base64_image}"}
            )
            payload.append(
                OpenAIPayload(
                    role="user",
                    content=[  # Wrap the image payload in a list
                        {
                            "type": "image_url",
                            "image_url": image_payload.image_url
                        }
                    ]
                ).model_dump()
            )
            logging.info("Base64 image appended as ImagePayload.")
                
        # Send request to OpenAI API
        openai_request = OpenAIRequest(
            model="gpt-4o-mini",
            messages=payload
        )
        logging.info("Sending request Payload to OpenAI LLM: %s", payload)
        
        # Extract and return response
        response = client.chat.completions.create(**openai_request.model_dump())
        logging.info("OpenAI API call successful. Extracting response.")
        
        result = response.choices[0].message.content
        logging.info("LLM Response: %s", result)
        
        return OpenAIResponse(response=result)
        
    except ValidationError as ve:
        logging.error("Validation error: %s", ve.errors())
        return OpenAIResponse(response=f"Validation Error: {ve.errors()}")
    except Exception as e:
        logging.error("Error in call_llm(): %s", str(e), exc_info=True)
        return OpenAIResponse(response=f"Error: {e}")