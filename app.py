import logging
import os
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, render_template

from llm import call_llm

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
def process_query():
    user_query = request.form.get("query", "")
    image_file = request.files.get("image")

    image_path = None

    # Checks if image was uploadded, if yes its uploaded to folder
    if image_file:
        try:
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
        except Exception as e:
            logging.error(f"Failed to upload image: {e}")
            return jsonify({"error": "Failed to process the image."}), 500

    # Call LLM with optional image
    response = call_llm(user_query, image_file)
    return jsonify({"response": response, "image_url": image_path})

if __name__ == '__main__':
    app.run(debug=True)
