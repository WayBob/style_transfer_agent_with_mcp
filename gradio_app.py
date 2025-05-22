# gradio_app.py
# -*- coding: utf-8 -*-

import gradio as gr
import os
from datetime import datetime
from PIL import Image
import pytesseract
import uuid

from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool

# Import Agent creation function and core tools/prompts from the core module (if needed for reference or comparison)
from core_agent import get_agent_runnable_and_checkpointer, get_core_llm, CORE_SYSTEM_PROMPT, CORE_TOOLS_LIST

# --- Environment Setup (Mainly handled by core_agent, but Gradio might need an LLM instance for non-Agent tasks, if any) ---
# llm_gradio = get_core_llm() # Get core LLM instance for Gradio interface specific logic (if needed)
# If Gradio's Agent fully relies on core_agent's get_agent_runnable..., this line might not be needed,
# because the llm instance will be created inside get_agent_runnable_and_checkpointer.
# For clarity, we let the Agent get its own LLM upon creation.

# --- Gradio Specific Tool Definitions ---

# OCR Tool (PIL Image version - Gradio specific)
def perform_ocr_gradio(pil_image: Image.Image) -> str:
    """Perform OCR using pytesseract to recognize Chinese and English characters in a PIL Image object."""
    if pil_image is None:
        return "Error: No image provided for OCR."
    try:
        text = pytesseract.image_to_string(pil_image, lang='chi_sim+eng')
        if not text.strip():
            return "OCR failed to recognize any text, or the image is blank."
        return f"Image OCR recognition results are as follows:\n{text.strip()}"
    except Exception as e:
        return f"OCR processing failed, error message: {str(e)}"

# List of tools used by Gradio Agent
# The definitions of these tools should be consistent with the tool functions in core_agent.py, but the instances are local.
# Alternatively, they can be selectively used from CORE_TOOLS_LIST, but ensure they are suitable for the Gradio environment
# (e.g., file path OCR is not directly applicable to Gradio's Image component)

def get_gradio_tools():
    """Build a list of tools specific to the Gradio application, selected from CORE_TOOLS_LIST."""
    gradio_tools = []
    for core_tool in CORE_TOOLS_LIST:
        # For Gradio, we do not directly use the file path version of the OCR tool (ImageFileOCR)
        # because OCR is preprocessed by perform_ocr_gradio.
        # The Agent receives text, not a file path for it to perform OCR.
        if core_tool.name == "ImageFileOCR":
            continue # Skip file path OCR tool
        gradio_tools.append(core_tool) # Other tools directly use the core version
    return gradio_tools

tools_gradio_list = get_gradio_tools()
print(f"[Gradio App] Tools loaded for Gradio Agent: {[tool.name for tool in tools_gradio_list]}")

# --- LangGraph Agent Setup (Gradio version, using core module) ---
# Gradio uses the core system prompt because it includes instructions for preprocessed OCR text and ListDirectoryFiles
system_prompt_gradio = CORE_SYSTEM_PROMPT 

# Use the core module's function to create the Agent, but pass Gradio-specific tool list and optional specific prompt
agent_runnable_gradio, checkpointer_gradio = get_agent_runnable_and_checkpointer(
    custom_tools=tools_gradio_list,
    custom_prompt=system_prompt_gradio 
    # debug is True by default in core_agent
)

# --- Gradio Interaction Logic (Basically unchanged) ---
def agent_chat_interface(user_message: str, history: list, image_upload: Image.Image, session_id: str):
    final_user_input = user_message
    if image_upload is not None:
        ocr_result_text = perform_ocr_gradio(image_upload)
        if user_message.strip():
            final_user_input = f"{user_message}\n\n[Attached Image OCR Content]:\n{ocr_result_text}"
        else:
            final_user_input = f"[Image OCR Content]:\n{ocr_result_text}"
    
    if not final_user_input.strip() and image_upload is None:
        history.append((user_message, "Please enter your question or upload an image."))
        return history, session_id

    messages_input = [HumanMessage(content=final_user_input)]
    # Only thread_id is needed in config, as checkpointer is already configured in agent_runnable
    config = {"configurable": {"thread_id": session_id}}
    
    bot_response_content = "Sorry, some issues occurred while processing your request."
    try:
        # print(f"[Gradio DEBUG] Calling agent with input: {final_user_input[:100]}... Tools: {[t.name for t in tools_gradio_list]}")
        result_state = agent_runnable_gradio.invoke({"messages": messages_input}, config=config)
        if result_state and "messages" in result_state and result_state["messages"]:
            ai_message = result_state["messages"][-1]
            if hasattr(ai_message, 'content') and ai_message.content:
                bot_response_content = ai_message.content
    except Exception as e:
        bot_response_content = f"Error calling Agent: {str(e)}"
        print(f"[Gradio ERROR] Error invoking agent for session {session_id}: {e}")
        import traceback
        traceback.print_exc()

    history.append((user_message if not image_upload else f"{user_message} (with image)", bot_response_content))
    return history, session_id

# --- Gradio UI Construction (Unchanged) ---
with gr.Blocks(theme=gr.themes.Soft(), title="LangGraph ReAct Agent") as demo:
    gr.Markdown("## 🧠 LangGraph ReAct Agent (GPT-4o) - Web UI")
    gr.Markdown(
        "Interact with an intelligent assistant built with LangGraph and GPT-4o."
        "You can ask questions, perform calculations, query the time, search the web, list project files, or upload images for text recognition."
    )
    session_id_state = gr.State(lambda: str(uuid.uuid4()))
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat History",
                bubble_full_width=False,
                avatar_images=(None, "https://raw.githubusercontent.com/gradio-app/gradio/main/gradio/components/chat_interface/processing_done.png")
            )
            user_input_textbox = gr.Textbox(
                label="Your Message",
                placeholder="Please enter your question, or ask here after uploading an image...",
                lines=3
            )
            with gr.Row():
                image_input_component = gr.Image(type="pil", label="Upload Image (Optional, for OCR)", sources=['upload', 'clipboard'])
            with gr.Row():
                send_button = gr.Button("Send / Process", variant="primary")
                clear_button = gr.Button("Clear Chat History")

    def handle_submit(user_msg, chat_history, img_upload, sess_id):
        if not user_msg.strip() and img_upload is None:
            pass 
        updated_history, updated_sess_id = agent_chat_interface(user_msg, chat_history, img_upload, sess_id)
        return updated_history, updated_sess_id, "", None

    user_input_textbox.submit(
        handle_submit,
        [user_input_textbox, chatbot, image_input_component, session_id_state],
        [chatbot, session_id_state, user_input_textbox, image_input_component]
    )
    send_button.click(
        handle_submit,
        [user_input_textbox, chatbot, image_input_component, session_id_state],
        [chatbot, session_id_state, user_input_textbox, image_input_component]
    )

    def clear_chat_and_session(current_sess_id):
        new_sess_id = str(uuid.uuid4())
        # print(f"[Gradio DEBUG] Chat cleared. Old session: {current_sess_id}, New session: {new_sess_id}")
        return [], new_sess_id, "", None

    clear_button.click(
        clear_chat_and_session, 
        [session_id_state], 
        [chatbot, session_id_state, user_input_textbox, image_input_component]
    )

# --- Launch Gradio Application (Unchanged) ---
if __name__ == "__main__":
    print("Launching Gradio application (Agent core from core_agent.py)...")
    demo.launch(server_name="0.0.0.0") 