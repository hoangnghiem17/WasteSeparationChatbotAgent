import os
import logging
from werkzeug.datastructures import FileStorage

import openai
from pydantic import ValidationError
from dotenv import load_dotenv

from services.image import encode_image
from models.models import TextPayload, ImagePayload, OpenAIRequest, OpenAIResponse

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(user_query: str, image_file: FileStorage =None) -> OpenAIResponse:
    """
    Makes API call to OpenAI multimodal LLM with optional image input.
    
    Args:
        user_query (str): User's text input 
        image_file (FileStorage, optional): Uploaded image as FileStorage object
        
    Returns:
        OpenAIResponse: Response from OpenAI LLM as validated response model.
    """
    try:
        logging.info("call_llm() invoked with prompt: %s", user_query)
        
        # Encode image if provided
        base64_image = None
        if image_file:
            base64_image = encode_image(image_file)

        # Construct text payload
        payload = [TextPayload(text=user_query).model_dump()]
        logging.debug("User prompt added to messages.")
        
        # Append image payload if available
        if base64_image:
            image_payload = ImagePayload(
                image_url={"url": f"data:image/jpeg;base64,{base64_image}"}
            ).model_dump()
            payload.append(image_payload)
            logging.debug("Base64 image appended to messages.")
        
        # Construct full OpenAI request
        request = OpenAIRequest(messages=payload)
        logging.debug("Sending request Payload to OpenAI LLM: %s", payload)
        
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

        # Send request to OpenAI
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.messages}
            ]
        )
        
        logging.info("OpenAI API call successful. Extracting response.")
        result = response.choices[0].message.content
        logging.debug("LLM Response: %s", result)
        
        return OpenAIResponse(response=result)
        
    except ValidationError as ve:
        logging.error("Validation error: %s", ve.errors())
        return OpenAIResponse(response=f"Validation Error: {ve.errors()}")
    except Exception as e:
        logging.error("Error in call_llm(): %s", str(e), exc_info=True)
        return OpenAIResponse(response=f"Error: {e}")
