from flask import session

def get_conversation():
    """
    Retrieve the current conversation history from the session.
    
    Args:
        None
        
    Returns:
        list: A list of dictionaries representating conversation history, where each dictionary contains the keys 'role' (str) and 'content' (str)
    """
    return session.get('conversation', [])

def add_message(role, content):
    """
    Add a new message to the conversation history.

    Args:
        role (str): The role of the message sender (e.g., 'user', 'assistant', or 'system').
        content (str): The content of the message to be added.

    Returns:
        None: Updates the session with the new message appended to the conversation history.
    """
    # Get current session
    conversation = session.get("conversation", [])
    conversation.append({"role": role, "content": content})
    
    session["conversation"] = conversation
    session.modified = True
