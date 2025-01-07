import logging
import os
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, render_template
from pydantic import ValidationError

from services.llm import call_llm
from models.models import OpenAIResponse

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Define directory for uploaded pictures
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Root URL
@app.route('/')
def index():
    return render_template("index.html")

# Handles POST requests from frontend when user submits query 
@app.route('/process_query', methods=['POST'])
def process_query() -> OpenAIResponse:
    try:
        user_query = request.form.get("query", "")
        image_file = request.files.get("image")

        # Checks if image was uploadded, if yes its uploaded to folder
        image_path = None
        if image_file:
            try:
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
            except Exception as e:
                logging.error(f"Failed to upload image: {e}")
                return jsonify({"error": "Failed to process the image."}), 500
        
        # Call LLM with validated UserQuery
        llm_response = call_llm(user_query, image_file)
        
        # Validate response with Pydantic
        validated_response = OpenAIResponse(response=llm_response.response)
        
        # Return structured response
        return jsonify(validated_response.model_dump())
    
    except ValidationError as ve:
        logging.error(f"Validation error: {ve.errors()}")
        return jsonify({"error": ve.errors()}), 422
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)
