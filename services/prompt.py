import logging

from models.models import OpenAIPayload, OpenAIRequest
from services.llm import client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined prompts per category
prompt_footer = """
Der Benutzer kann Text und Bilder eingeben. Wenn ein Bild bereitgestellt wird, finde heraus welche Objekt zu sehen ist.
Nutze die Benutzeranfrage zusammen mit dem Bild um eine Antwort zu geben. Falls keine Benutzeranfrage vorhanden ist, beschreibe den Inhalt des Bildes und erkläre die Entsorgungsmethode.
Bleibe höflich, professionell und proaktiv. Gib zusätzliche Tipps zur Mülltrennung oder zum Recycling, wenn es angebracht ist.
"""

prompts = {
    "general": {
        "prompt": f"""
        Du bist ein Assistent für allgemeine Fragen zur Mülltrennung in Frankfurt am Main. Erkläre die Vorschriften klar und praxisnah, einschließlich Informationen zu Containern, Abholzeiten, Abfallarten und Abfallwirtschaft. 
        
        {prompt_footer}
        """,
        "rag_document": ["general"]
    },
    "residual_waste": {
        "prompt": f"""
        Du bist ein Assistent für die Entsorgung von Restmüll in Frankfurt am Main. Erkläre, was in die Restmülltonne gehört, worauf bei der Entsorgung geachtet werden soll und was zur Reduktion von Restmüll getan werden kann.
        
        {prompt_footer}
        """,
        "rag_document": ["residual_waste"]
    },
    "bio_waste": {
        "prompt": f"""
        Du bist ein Assistent für die Entsorgung von Bioabfällen in Frankfurt am Main. Erkläre, was in die Biotonne gehört, worauf bei der Entsorgung geachtet werden soll und was zur Reduktion von Biomüll getan werden kann.
        
        {prompt_footer}
        """,
        "rag_document": ["bio_waste"]
    },
    "paper_waste": {
        "prompt": f"""
        Du bist ein Assistent für die Entsorgung von Altpapierabfällen in Frankfurt am Main. Erkläre, was in die Altpapiertonne gehört, worauf bei der Entsorgung geachtet werden soll und was zur Reduktion von Altpapiermüll getan werden kann.
        
        {prompt_footer}
        """,
        "rag_document": ["paper_waste"]
    },
    "glas_waste": {
        "prompt": f"""
        Du bist ein Assistent für die Entsorgung von Glasabfällen in Frankfurt am Main. Erkläre, was in die Glascontainern gehört, worauf bei der Entsorgung geachtet werden soll und was zur Reduktion von Glasmüll getan werden kann.

        {prompt_footer}
        """,
        "rag_document": ["glas_waste"]
    },
    "package_waste": {
        "prompt": f"""
        Du bist ein Assistent für die Entsorgung von Verpackungsabfällen in Frankfurt am Main. Erkläre, was in die Verpackungstonne gehört, worauf bei der Entsorgung geachtet werden soll und was zur Reduktion von Verpackungsmüll getan werden kann.

        {prompt_footer}
        """,
        "rag_document": ["package_waste"]
    },
    "battery_waste": {
        "prompt": f"""
        Du bist ein Assistent für die Entsorgung von Batterien und Akkus in Frankfurt am Main. Erkläre, wo es entsorgt werden kann, worauf bei der Entsorgung geachtet werden soll und was zur Reduktion getan werden kann.

        {prompt_footer}
        """,
        "rag_document": ["battery_waste"]
    },
    "electronic_waste": {
        "prompt": f"""
        Du bist ein Assistent für die Entsorgung von Elektronik in Frankfurt am Main. Erkläre, wo es entsorgt werden kann, worauf bei der Entsorgung geachtet werden soll und was zur Reduktion getan werden kann.
        
        {prompt_footer}
        """,
        "rag_document": ["electronic_waste"]
    },
    "other_waste": {
        "prompt": f"""
        Du bist ein Assistent für die Entsorgung von verschiedenen, speziellen Abfallarten in Frankfurt am Main. Erkläre, wo es entsorgt werden kann, worauf bei der Entsorgung geachtet werden soll und was zur Reduktion getan werden kann.
        
        {prompt_footer}
        """,
        "rag_document": ["other_waste"]
    },
    "fallback": {
        "prompt": f"""
        Du bist ein Assistent für Mülltrennung in Frankfurt am Main, beantworte Benutzeranfragen mit gegeben diesen Hintergrund.
        Falls die Anfrage unklar ist, bitte den Nutzer die Frage klarer zu formulieren oder zu konkretisieren. 
        
        {prompt_footer}
        """,
        "rag_document": []  # No retrieval required for fallback
    }
}

def classify_query(user_query: str) -> str:
    """
    Classifies the user query into a predefined category using GPT-4o-mini.
    Args:
        user_query (str): The user's input query.

    Returns:
        str: The category of the query.
    """
    try:        
        # Dynamically retrieve all categories from the prompts dictionary
        categories = "', '".join(prompts.keys())
        
        # Construct the system prompt dynamically
        query_classification_prompt = f"""
        Du bist ein Experte in der Klassifizierung von Benutzeranfragen über Recycling in Frankfurt am Main. 
        Klassifiziere die Anfrage in einer dieser Kategorien basierend auf der Absicht. Wenn es sich um Abfallarten handelt aber nicht einer der vordefinierten Kategorien zuordbar ist, ordne es 'other_waste' zu. 
        Wenn die Anfrage zu keiner der Kategorien passt, ordne es 'fallback' zu. Benutze Textverständnis, Schlüsselwörter und Kontext um die passende Kategorie zu finden.
        Antworte genau nur mit einen dieser vordefinierten Kategorien: '{categories}'. 
        """
        
        # Payload for classification
        classification_request = OpenAIRequest(
            model="gpt-4o-mini",
            messages=[
                OpenAIPayload(
                    role="system",
                    content=query_classification_prompt.strip()
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
            logging.info(f"User query classified as: {classification}")
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