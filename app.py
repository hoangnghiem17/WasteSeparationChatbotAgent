import logging

from flask import Flask, request, jsonify, render_template

from llm import call_llm

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process_query', methods=['POST'])
def process_query():
    """
    Endpoint to process a query.
    """
    user_query = request.form.get("query", "")
    image_file = request.files.get("image")

    # Call for LLM processing
    response = call_llm(user_query, image_file)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
