from flask import Flask, request, jsonify

from llm import process_text_query

app = Flask(__name__)

@app.route('/process_text', methods=['POST'])
def process_text():
    """
    Endpoint to process a text query.
    """
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "No text query provided"}), 400

    answer = process_text_query(user_query)
    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True)
