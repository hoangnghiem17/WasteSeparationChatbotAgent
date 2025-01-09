import os
import logging
from werkzeug.datastructures import FileStorage

import openai
from pydantic import ValidationError
from dotenv import load_dotenv

from services.image import encode_image
from models.models import TextPayload, ImagePayload, OpenAIPayload, OpenAIRequest, OpenAIResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(conversation_history: list, image_file: FileStorage =None) -> OpenAIResponse:
    """
    Makes API call to OpenAI multimodal LLM with optional image input.
    
    Args:
        conversation_history (List[dict]): List of dictionaries with role-content pairs of conversation history 
        image_file (FileStorage, optional): Uploaded image as FileStorage object
        
    Returns:
        OpenAIResponse: Response from OpenAI LLM as validated response model.
    """
    try:
        logging.info("call_llm() invoked with conversation history.")

        system_prompt = f"""
        Du bist ein Assistent für Mülltrennung in der Stadt Frankfurt am Main. 
        Nutze dein Wissen über die lokalen Vorschriften zur Mülltrennung und das Recycling in Frankfurt am Main, 
        um präzise, klare und praxisnahe Antworten auf Deutsch zu geben. 

        Der Benutzer kann Text und Bilder hochladen. 
        Wenn ein Bild bereitgestellt wird, versuche zu klassifizieren, welches Objekt darauf zu sehen ist. 
        Nutze die Benutzeranfrage (falls vorhanden), um das Bild und den Text zu kombinieren und eine Antwort zu geben. 
        Falls keine Benutzeranfrage vorhanden ist, beschreibe den Inhalt des Bildes und erkläre die richtige Entsorgungsmethode. 

        Falls die angeforderte Information unklar oder nicht verfügbar ist, 
        weise den Nutzer darauf hin, sich auf der offiziellen Webseite der Abfallwirtschaft Frankfurt oder bei FES Frankfurt zu informieren. 

        Bleibe höflich, professionell und proaktiv. 
        Gib zusätzliche Tipps zur Mülltrennung oder zum Recycling, wenn es angebracht ist.
        """
        
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
