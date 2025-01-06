import os
import logging
import openai
from dotenv import load_dotenv

from image import encode_image

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt, image_file=None):
    """
    Makes API call to OpenAI multimodal LLM with optional image input.
    
    Args:
        prompt (str): user's text input 
        image_file (file-like object, optional): image file uploaded by user
        
    Returns:
        str: response from OpenAI LLM as text string   
    """
    try:
        logging.info("call_llm() invoked with prompt: %s", prompt)
        
        # Encode image if provided
        base64_image = None
        if image_file:
            base64_image = encode_image(image_file)

        # Construct message content
        message_content = [{"type": "text", "text": prompt}]
        logging.debug("User prompt added to messages.")
        
        if base64_image:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            )
            logging.debug("Base64 image appended to messages.")
        
        # Log the constructed payload
        logging.debug("Sending request Payload to OpenAI LLM: %s", message_content)
        
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content}
            ]
        )
        
        logging.info("OpenAI API call successful. Extracting response.")
        result = response.choices[0].message.content
        logging.debug("LLM Response: %s", result)
        
        return result
        
    except Exception as e:
        logging.error("Error in call_llm(): %s", str(e), exc_info=True)
        return f"Error: {e}"
