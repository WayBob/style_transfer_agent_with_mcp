# LangGraph ReAct Agent with GPT-4o

This project implements a ReAct (Reasoning and Acting) agent using LangGraph and OpenAI's GPT-4o. It features a command-line interface, dialogue memory, and a suite of tools for enhanced interaction.

## Core Features

*   Conversational AI with dialogue memory (in-session).
*   Tool utilization: Image OCR, Current Time, Web Search, and Calculator.
*   Powered by LangGraph and GPT-4o.
*   Detailed interaction and debug logging to `agent_interaction.log` (and terminal).

## Prerequisites

*   Python 3.8+
*   OpenAI API Key
*   Tesseract OCR engine (see [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html) and ensure it's in your PATH. For OCR, Chinese (`chi_sim`) and English (`eng`) language data packs are recommended.)

## Quick Start

1.  **Clone (if applicable) & Navigate** to the project directory.

2.  **Set up Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` exists with a list of packages like `python-dotenv`, `Pillow`, `pytesseract`, `langchain`, `langchain-openai`, `langchain-community`, `langgraph`, `duckduckgo-search`)*

4.  **Configure API Key:**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

5.  **Run the Agent:**
    ```bash
    python main.py
    ```
    Interact with the agent in the terminal. Type "退出" or "exit" to quit.

## Logging

Interactions and debug information are logged to `agent_interaction.log` and also appear in the terminal.

## Further Development

This project serves as a foundational ReAct agent. Future enhancements can include:
*   More sophisticated error handling.
*   Persistent memory solutions (e.g., `SQLChatMessageHistory`).
*   Additional tools and capabilities.
*   Web interface or API endpoints.
