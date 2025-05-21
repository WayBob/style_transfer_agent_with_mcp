[简体中文](./documentation/README_zh.md) | English

# LangGraph ReAct Agent with GPT-4o

This project implements a ReAct (Reasoning and Acting) agent using LangGraph and OpenAI's GPT-4o. It features a command-line interface, dialogue memory, and a suite of tools for enhanced interaction.

## Core Features

*   Conversational AI with dialogue memory (in-session).
*   Tool utilization: Image OCR, Current Time, Web Search, Directory File Listing, and Calculator.
*   Powered by LangGraph and GPT-4o.
*   Detailed interaction and debug logging to `agent_interaction.log` (and terminal).

## Prerequisites

*   Python 3.8+
*   [uv](https://github.com/astral-sh/uv) (Python packaging tool). Install on macOS and Linux with:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    For Windows and other installation methods, refer to the [official uv installation guide](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).
*   OpenAI API Key
*   Tesseract OCR engine (see [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html) and ensure it's in your PATH. For OCR, Chinese (`chi_sim`) and English (`eng`) language data packs are recommended.)

## Quick Start

1.  **Clone (if applicable) & Navigate** to the project directory.
    *(If you used `uv init <project_name>` to create the project, you might already be in the directory).* 

2.  **Set up Virtual Environment (using uv):**
    If a virtual environment (`.venv`) doesn't already exist, create it:
    ```bash
    uv venv
    ```
    Activate the virtual environment:
    ```bash
    source .venv/bin/activate  # On Windows use .venv\Scripts\activate
    ```

3.  **Install Dependencies (using uv):**
    Ensure you have a `requirements.txt` file in your project root with the following content (or similar, based on project needs):
    ```txt
    python-dotenv
    Pillow
    pytesseract
    langchain
    langchain-openai
    langchain-community
    langgraph
    duckduckgo-search
    gradio
    ```
    Install dependencies from `requirements.txt`:
    ```bash
    uv pip install -r requirements.txt
    ```
    Alternatively, to add individual packages (like Gradio, if not in `requirements.txt` yet):
    ```bash
    uv add gradio
    # uv add python-dotenv Pillow pytesseract langchain langchain-openai langchain-community langgraph duckduckgo-search
    ```
    *(Using `uv add` will update your `pyproject.toml` if your project was initialized with `uv init` to use one. If you are primarily using `requirements.txt`, ensure it's kept up-to-date and use `uv pip install -r requirements.txt` or `uv sync` after adding packages via `uv add` if your project uses `pyproject.toml` for locking.)*

4.  **Configure API Key:**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

5.  **Run the Applications:**
    *   **Command-Line Agent:**
        ```bash
        python main.py
        ```
    *   **Gradio Web UI:**
        ```bash
        python gradio_app.py
        ```
    Interact with the agent in the terminal or web browser. Type "退出" or "exit" in the CLI to quit.

## Logging

Interactions and debug information are logged to `agent_interaction.log` and also appear in the terminal.

## Project Structure

*   `core_agent.py`: Contains the central logic for the LangGraph agent, including LLM configuration, tool definitions, and the agent creation function.
*   `main.py`: Provides a command-line interface for interacting with the agent.
*   `gradio_app.py`: Implements a Gradio-based web UI for the agent.
*   `.env`: Stores the OpenAI API key (ensure this is in `.gitignore`).
*   `requirements.txt`: Lists Python dependencies.
*   `agent_interaction.log`: Log file for agent interactions and debug output.

## Further Development

This project serves as a foundational ReAct agent. Future enhancements can include:
*   More sophisticated error handling.
*   Persistent memory solutions (e.g., `SQLChatMessageHistory`).
*   Additional tools and capabilities.
*   Configuration options for `debug` mode via CLI or UI.
