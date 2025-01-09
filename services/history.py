from flask import session

def get_conversation():
    """
    Retrieve the current conversation history from the session.
    """
    return session.get('conversation', [])

def add_message(role, content):
    """
    Add a new message to get the conversation history
    """
    # Get current session
    conversation = session.get("conversation", [])
    conversation.append({"role": role, "content": content})
    
    session["conversation"] = conversation
    session.modified = True
