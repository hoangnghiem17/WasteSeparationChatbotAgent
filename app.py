import logging
import os
from werkzeug.utils import secure_filename
from datetime import timedelta
from flask import Flask, request, jsonify, render_template, session

from langchain_core.agents import AgentFinish
from langchain.agents.output_parsers.tools import ToolAgentAction
from agents.waste_agent import run_tool_agent, agent_app

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


@app.route('/agent', methods=['POST'])
def agent_endpoint():
    data = request.form
    user_query = data.get("query")
    image_file = request.files.get("image")  # Handle optional image input
    image_path = None

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

    # Prepare the state for the agent
    agent_state = {
        "input": {"user_query": user_query, "image_path": image_path},
        "chat_history": session.get("conversation", []),
        "agent_outcome": None,
        "intermediate_steps": []
    }

    try:
        # Run the agent workflow
        output_state = agent_app.invoke(agent_state)
        
        # Log the full agent outcome for investigation
        agent_outcome = output_state.get("agent_outcome")
        logging.info(f"Full agent_outcome: {agent_outcome}")
        print(f"RETURN_VALUES: {agent_outcome.return_values.get('output')}")
        
        # Include the full `agent_outcome` in the API response for inspection
        return jsonify({"agent_outcome": agent_outcome})
    except Exception as e:
        logging.error(f"Error in agent workflow: {e}")
        return jsonify({"error": "An error occurred while processing the query."}), 500


if __name__ == '__main__':
    app.run(debug=True)
