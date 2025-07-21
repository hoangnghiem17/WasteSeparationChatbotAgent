# WasteSeparationChatbotAgent

This project describes an agent-based chatbot assistant for waste separation guidance in Frankfurt am Main, Germany. It provides citizens with accurate information about proper waste disposal and helps them locate nearby recycling facilities.

## Description

The chatbot is a multilingual (primarily German/English) conversational AI system that can process both text and image. The system combines natural language processing, computer vision and geographic information to provide comprehensive waste management guidance. Key features include:

- **Multimodal Input Support**: Accepts both text queries and image uploads for waste identification
- **Intelligent Query Classification**: Automatically categorizes waste types into 9 predefined categories
- **Context-Aware Responses**: Uses RAG (Retrieval-Augmented Generation) to provide accurate, up-to-date information
- **Facility Locator**: Finds the nearest recycling facilities with driving directions and opening hours

## Problem/Background

### The Challenge

Proper waste separation is crucial for environmental sustainability and efficient recycling processes on city-level. I researched this in detail in my master thesis for the city of Frankfurt. The results indicate that many citizens face challenges in:

- **Identifying waste categories**: Uncertainty about which bin to use for specific items
- **Finding disposal information**: Lack of knowledge about specialized waste disposal methods
- **Locating facilities**: Difficulty finding nearby recycling centers and their operating hours
- **Language barriers**: Limited access to waste management information in preferred languages

### The Solution

This chatbot addresses these challenges by providing:

- **Instant Guidance**: Real-time answers to waste separation questions
- **Visual Recognition**: Image-based waste identification for items users are unsure about
- **Localized Information**: Frankfurt-specific waste management guidelines and facility locations

## Architecture

### System Overview

The application follows a modular architecture. The web interface forwards incoming traffic to the central flask server that saves static data. Queries are processed by the agent service to return an answer by using a knowledge base and database if required:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│   Web Interface │    │   Flask Server  │    │   Agent Service     │
│   (HTML/CSS/JS) │◄──►│   (app.py)      │◄──►│  (LangChain/Graph)  │
└─────────────────┘    └─────────────────┘    └─────────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   File Storage  │    │   Knowledge     │
                       │   (Images)      │    │   Base (FAISS)  │
                       └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Database      │
                                              │   (SQLite)      │
                                              └─────────────────┘
```

### Core Components

#### 1. **Agent System** (`agent/waste_agent.py`)
- **LangGraph-based workflow** with tool-calling capabilities
- **State management** for conversation context
- **Tool orchestration** for query processing

#### 2. **Query Classification** (`agent/query_classification.py`)
- **Multimodal classification** (text + image)
- **9 waste categories**: battery_waste, bio_waste, electronic_waste, glas_waste, package_waste, paper_waste, residual_waste, other_waste, general_waste
- **Location queries** for facility finding
- **Fallback handling** for non-waste queries

#### 3. **Information Retrieval** (`agent/retrieval.py`)
- **FAISS vector store** for semantic search
- **Category-filtered retrieval** for relevant information
- **OpenAI embeddings** for similarity matching

#### 4. **Geographic Services** (`agent/geocoding.py`, `agent/facility.py`)
- **Address geocoding** using external Nominatim API
- **Facility database** with SQLite storage
- **OSRM routing** for distance calculations

#### 5. **Web Interface** (`templates/`, `static/`)
- **Responsive design** with Frankfurt branding
- **Real-time chat** with message history
- **Image upload** functionality
- **Session management** for conversation continuity

### Data Sources

#### Knowledge Base
- **FAISS vector store** with embedded waste management documents
- **Category-specific documents** for each waste type
- **Frankfurt waste management guidelines** and regulations

#### Facility Database
- **Recycling facilities** in Frankfurt am Main
- **Location data** (coordinates, addresses)
- **Operating hours** and contact information

#### Conversation Logging
- **SQLite database** for conversation tracking
- **Message history** with timestamps
- **User interaction analytics**

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Internet connection for external services

### Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (OpenAI API key)
4. Initialize databases and knowledge base
5. Run the application: `python app.py`

## Acknowledgments

- **City of Frankfurt am Main** - Project support and domain expertise
- **FES** - Waste management knowledge and facility data
- **Frankfurt University of Applied Sciences** - Academic collaboration
- **OpenAI** - AI technology and API services