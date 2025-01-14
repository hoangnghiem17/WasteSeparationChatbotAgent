import logging
import os
from werkzeug.utils import secure_filename
from datetime import timedelta
from flask import Flask, request, jsonify, render_template, session
from pydantic import ValidationError

from services.llm import call_llm
from services.history import get_conversation, add_message
from services.prompt import match_prompt_to_query
from services.retrieval import retrieve_similar_chunks
from models.models import TextPayload, OpenAIResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Flask session configuration
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")
app.config['SESSION_COOKIE_NAME'] = 'wastechat_session'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True for HTTPS
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Prevents cross-site session retention
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Timeout after 30 min inactivity

# Define directory for uploaded pictures
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.before_request
def manage_session():
    session.permanent = False  # Session lasts until the browser/tab closes


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/reset', methods=['POST'])
def reset_conversation():
    session.pop('conversation', None)  # Clear conversation only
    return jsonify({"response": "Conversation reset."})


@app.route('/process_query', methods=['POST'])
def process_query() -> OpenAIResponse:
    try:
        user_query = request.form.get("query", "")
        logging.info(f"Received user query: '{user_query}'")

        user_query = TextPayload(text=user_query).model_dump()
        image_file = request.files.get("image")
       
        # Initialize conversation if not present
        if 'conversation' not in session:
            session['conversation'] = []
        
        # Add user message to conversation history
        add_message("user", user_query["text"])
        logging.debug(f"Added user message to conversation history: {user_query['text']}")
        
        # Get the prompt and rag_document
        prompt_data = match_prompt_to_query(user_query["text"])
        system_prompt = prompt_data["prompt"]
        rag_document = prompt_data.get("rag_document", [])
        logging.info(f"Associated rag_document for retrieval: {rag_document}")
        
        # Retrieve context if rag_document is not empty
        context = ""
        if rag_document:
            logging.info("Starting context retrieval.")
            context_chunks = retrieve_similar_chunks(
                query=user_query["text"],
                faiss_store_path="faiss_store",
                rag_document=rag_document
            )
            if context_chunks:
                context = "\n\n".join(context_chunks)
                logging.info(f"Retrieved {len(context_chunks)} chunks for context.")
            else:
                logging.warning(f"No relevant chunks found for rag_document: {rag_document}")
        else:
            logging.info("No documents specified for retrieval. Skipping context retrieval.")
                    
        # Handle image file if provided                    
        if image_file:
            try:
                logging.info(f"Received an image file: {image_file.filename}")
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                logging.info(f"Image file saved at: {image_path}")
            except Exception as e:
                logging.error(f"Failed to process the uploaded image: {e}")
                return jsonify({"error": "Failed to process the image."}), 500
        
        # Call LLM with the retrieved context included
        logging.info("Calling LLM with conversation history and retrieved context.")
        llm_response = call_llm(
            conversation_history=get_conversation(),
            image_file=image_file,
            system_prompt=system_prompt,
            context=context.strip()  # Strip extra spaces or newlines
        )
        logging.info("LLM call successful. Extracting response.")
        
        # Add assistant's response to the conversation history
        add_message('assistant', llm_response.response)
        logging.debug(f"Assistant response added to conversation history: {llm_response.response[:100]}...")  # Log first 100 characters
        
        return jsonify({"response": llm_response.response})
    
    except ValidationError as ve:
        logging.error(f"Validation error: {ve.errors()}")
        return jsonify({"error": ve.errors()}), 422
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)
