import logging
import os
from werkzeug.utils import secure_filename
from datetime import timedelta
from flask import Flask, request, jsonify, render_template, session
from pydantic import ValidationError

from services.llm import call_llm
from services.history import get_conversation, add_message
from services.prompt import match_prompt_to_query
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
        user_query = TextPayload(text=user_query).model_dump()
        image_file = request.files.get("image")
       
       # Initialize conversation if not present
        if 'conversation' not in session:
            session['conversation'] = []
        
        # Add user message to conversation history
        add_message("user", user_query["text"])
        
        # Get the system prompt for the query
        system_prompt = match_prompt_to_query(user_query["text"])
        
        # Handle image file if provided                    
        if image_file:
            try:
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
            except Exception as e:
                logging.error(f"Failed to upload image: {e}")
                return jsonify({"error": "Failed to process the image."}), 500
        
        llm_response = call_llm(get_conversation(), image_file, system_prompt=system_prompt)
        
        # Add assistant's response to the conversation history
        add_message('assistant', llm_response.response)
        
        return jsonify({"response": llm_response.response})
    
    except ValidationError as ve:
        logging.error(f"Validation error: {ve.errors()}")
        return jsonify({"error": ve.errors()}), 422
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500


if __name__ == '__main__':
    app.run(debug=True)
