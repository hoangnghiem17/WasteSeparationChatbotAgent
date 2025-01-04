import openai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

def test_llm_call():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the latest GPT model
            messages=[
                {"role": "system", "content": "You are an assistant that provides recycling tips."},
                {"role": "user", "content": "Explain recycling for plastic bottles."}
            ]
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm_call()
