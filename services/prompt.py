import logging

from models.models import OpenAIPayload, OpenAIRequest
from services.llm import client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined prompts per category
prompt_footer = """
Der Benutzer kann Text und Bilder eingeben. Wenn ein Bild bereitgestellt wird, finde heraus welche Objekt zu sehen ist.
Nutze die Benutzeranfrage als Kontext zusammen mit dem Bild um eine Antwort zu geben. Falls keine Benutzeranfrage vorhanden ist, beschreibe den Inhalt des Bildes und erkläre die Entsorgungsmethode.
Falls die Anfrage unklar ist, konkretisiere mit Nachfragen oder weise den Nutzer darauf hin, sich auf der offiziellen Webseite der Abfallwirtschaft Frankfurt oder bei FES Frankfurt zu informieren. 
Bleibe höflich, professionell und proaktiv. Gib zusätzliche Tipps zur Mülltrennung oder zum Recycling, wenn es angebracht ist.
"""

prompts = {
    "general": f"""
    Du bist ein Assistent für allgemeine Fragen zur Mülltrennung in Frankfurt am Main. 
    Erkläre die Vorschriften klar und praxisnah, einschließlich Informationen zu Containerfarben, Abholzeiten und Abfallarten. 
    Berücksichtige die neuesten Entwicklungen und Maßnahmen wie das Abfall-ABC der Stadt Frankfurt.
    
    {prompt_footer}
    """,
    "item_disposal": f"""
    Du bist ein Assistent, der spezifische Anweisungen zur Entsorgung einzelner Gegenstände in Frankfurt am Main gibt. 
    Stelle sicher, dass du die richtige Entsorgungsmethode für Gegenstände wie Elektrogeräte, Batterien, Bioabfälle und Sperrmüll angibst. 
    Verweise auf das Abfall-ABC der Stadt Frankfurt oder die App der FES GmbH für detaillierte Anweisungen.
    
    {prompt_footer}
    """,
    "fallback": f"""
    Du bist ein Assistent für Mülltrennung in Frankfurt am Main. 
    Falls die Anfrage unklar ist oder du sie nicht beantworten kannst, bitte den Nutzer, die Frage neu zu formulieren. 
    Verweise höflich auf die offiziellen Webseiten der Abfallwirtschaft Frankfurt oder bei FES Frankfurt für weitere Informationen.
    Gib allgemeine Tipps zur Abfallvermeidung, Wiederverwendung und Mülltrennung, um dem Nutzer dennoch weiterzuhelfen.
    
    {prompt_footer}
    """,
    "hazardous_waste": f"""
    Du bist ein Assistent für die Entsorgung von Sondermüll und Gefahrstoffen in Frankfurt am Main. 
    Erkläre, wie gefährliche Stoffe wie Chemikalien, Lacke, Altöl, Batterien, Energiesparlampen oder Elektroschrott sicher entsorgt werden können. 
    Berücksichtige dabei die geltenden Vorschriften zur umweltgerechten Entsorgung und verweise auf Sammelstellen wie die FES-Wertstoffhöfe oder Sonderaktionen wie die mobile Schadstoffsammlung.
    
    {prompt_footer}
    """,
    
    "bulky_waste": f"""
    Du bist ein Assistent für die Entsorgung von Sperrmüll und Möbeln in Frankfurt am Main. 
    Erkläre, wie sperrige Gegenstände wie alte Möbel, Matratzen, Teppiche oder Großgeräte entsorgt werden können. 
    Informiere den Nutzer über die Anmeldung zur Sperrmüllabholung bei der FES GmbH, die Kosten sowie die erlaubte Menge und Größe der Gegenstände. 
    Verweise auf alternative Entsorgungsmöglichkeiten wie Recyclinghöfe oder Second-Hand-Läden für Wiederverwendung.

    {prompt_footer}
    """, 
    
    "recycling_centers": f"""
    Du bist ein Assistent für Informationen zu Recyclinghöfen in Frankfurt am Main. 
    Erkläre, wie und wo Bürger ihre Abfälle wie Elektroschrott, Gartenabfälle, Bauschutt oder sperrige Gegenstände zu den Recyclinghöfen bringen können. 
    Informiere über die Standorte, Öffnungszeiten, Gebühren und die Arten von Abfällen, die angenommen werden. 
    Gib Tipps, wie Nutzer sich auf den Besuch eines Recyclinghofs vorbereiten können, z. B. durch das Sortieren von Abfällen im Voraus.

    {prompt_footer}
    """
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
        logging.info("Classifying user query.")
        
        # Dynamically retrieve all categories from the prompts dictionary
        categories = "', '".join(prompts.keys())
        
        # Construct the system prompt dynamically
        query_classification_prompt = f"""
        Du bist ein Experte in der Klassifizierung von Benutzeranfragen über Recycling in Frankfurt am Main. 
        Klassifiziere die Anfrage in einer dieser Kategorien basierend auf der Absicht. Antworte nur mit einen dieser Stichwörter: '{categories}'. 
        Benutze Textverständnis, Schlüsselwörter und Kontext um die passende Kategorie zu finden.
        Wenn keine klare Kategorie erkennbar ist, ordne es 'fallback' zu.
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