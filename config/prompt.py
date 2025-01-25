import json

# Prompt construction
with open("config/categories.json", "r", encoding="utf-8") as file:
    categories = json.load(file)
    
formatted_categories = "\n".join([f"- {key}: {value}" for key, value in categories.items()])
keywords_categories = list(categories.keys())

# Agent System Prompt Formatting  
with open("config/agent_prompt.txt", "r", encoding="utf-8") as file:
    prompt_template = file.read()    

system_prompt = prompt_template.format(
    formatted_categories=formatted_categories,
    keywords_categories=keywords_categories
)