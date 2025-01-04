import os

import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def process_text_query(user_query):
    """
    Process a text query using OpenAI's language model.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": user_query}]
        )
        answer = response["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"
