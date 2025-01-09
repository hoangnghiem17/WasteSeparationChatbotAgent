import logging

from models.models import OpenAIPayload, OpenAIRequest
from services.llm import client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined prompts per category
prompts = {
    "general": """
    Du bist ein Assistent für allgemeine Fragen zur Mülltrennung in Frankfurt am Main. 
    Der Benutzer kann Text und Bilder eingeben. Wenn ein Bild bereitgestellt wird, finde heraus welche Objekt zu sehen ist.
    Nutze die Benutzeranfrage als Kontext zusammen mit dem Bild um eine Antwort zu geben. Falls keine Benutzeranfrage vorhanden ist, beschreibe den Inhalt des Bildes und erkläre die Entsorgungsmethode.
    Nutze dein Wissen über die lokalen Vorschriften und aktuellen Entwicklungen, um präzise und praxisnahe Antworten auf Deutsch zu geben. 
    Falls die Anfrage unklar ist, weise den Nutzer darauf hin, sich auf der offiziellen Webseite der Abfallwirtschaft Frankfurt oder bei FES Frankfurt zu informieren. 
    Bleibe höflich, professionell und proaktiv. Gib zusätzliche Tipps zur Mülltrennung oder zum Recycling, wenn es angebracht ist.
    """,
    "item_disposal": """
    Du bist ein Assistent, der spezifische Anweisungen zur Entsorgung einzelner Gegenstände in Frankfurt am Main gibt. 
    Der Benutzer kann Text und Bilder eingeben. Wenn ein Bild bereitgestellt wird, finde heraus welche Objekt zu sehen ist.
    Nutze die Benutzeranfrage als Kontext zusammen mit dem Bild um eine Antwort zu geben. Falls keine Benutzeranfrage vorhanden ist, beschreibe den Inhalt des Bildes und erkläre die Entsorgungsmethode.
    Nutze dein Wissen über die lokalen Entsorgungsmethoden, um genaue und hilfreiche Antworten auf Deutsch bereitzustellen. 
    Konsultiere das Abfall-ABC der Stadt Frankfurt, um den richtigen Entsorgungsweg für verschiedene Abfallarten zu bestimmen.
    Falls die Anfrage unklar ist, weise den Nutzer darauf hin, sich auf der offiziellen Webseite der Abfallwirtschaft Frankfurt oder bei FES Frankfurt zu informieren. 
    Bleibe höflich, professionell und proaktiv. Gib zusätzliche Tipps zur Mülltrennung oder zum Recycling, wenn es angebracht ist.
    """,
    "fallback": """
    Du bist ein Assistent für Mülltrennung in Frankfurt am Main. 
    Der Benutzer kann Text und Bilder eingeben. Wenn ein Bild bereitgestellt wird, finde heraus welche Objekt zu sehen ist.
    Nutze die Benutzeranfrage als Kontext zusammen mit dem Bild um eine Antwort zu geben. Falls keine Benutzeranfrage vorhanden ist, beschreibe den Inhalt des Bildes und erkläre die Entsorgungsmethode.
    Wenn du eine Anfrage nicht beantworten kannst, bitte den Nutzer um weitere Informationen oder verweise auf die offiziellen Webseiten der Abfallwirtschaft Frankfurt oder FES Frankfurt. 
    Bleibe stets höflich, professionell und proaktiv. Gib allgemeine Tipps zur Mülltrennung oder zum Recycling, wenn es angebracht ist.
    """
}

def classify_query(user_query: str) -> str:
    """
    Classifies the user query into a predefined category using GPT-4o-mini.
    Args:
        user_query (str): The user's input query.

    Returns:
        str: The category of the query ("general", "item_disposal", or "fallback").
    """
    try:
        logging.info("Classifying user query.")
        
        # Payload for classification
        classification_request = OpenAIRequest(
            model="gpt-4o-mini",
            messages=[
                OpenAIPayload(
                    role="system",
                    content="Classify the following query into one of these categories: 'general', 'item_disposal', or 'fallback'."
                ).model_dump(),
                OpenAIPayload(
                    role="user",
                    content=user_query
                ).model_dump()
            ]
        )
        
        # API call
        response = client.chat.completions.create(**classification_request.model_dump())
        classification = response.choices[0].message.content.strip().lower()
        
        # Return valid categories or fallback
        if classification in prompts:
            logging.info(f"Query classified as: {classification}")
            return classification
        else:
            logging.warning(f"Unrecognized classification: {classification}. Using fallback.")
            return "fallback"
        
    except Exception as e:
        logging.error(f"Error classifying query: {e}", exc_info=True)
        return "fallback"
    
def match_prompt_to_query(user_query: str) -> str:
    """
    Matches the user query to a predefined prompt based on classification.
    Args:
        user_query (str): The user's input query.

    Returns:
        str: The appropriate system prompt for the query.
    """
    category = classify_query(user_query)
    return prompts[category]