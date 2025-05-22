# core_agent.py
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from PIL import Image
import pytesseract

from dotenv import load_dotenv
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

# --- Environment and Model Initialization ---
load_dotenv() # Ensure environment variables are loaded

OPENAI_API_KEY_CORE = os.getenv("OPENAI_API_KEY")

def get_core_llm():
    """Returns a configured core ChatOpenAI LLM instance."""
    if not OPENAI_API_KEY_CORE:
        raise ValueError("OpenAI API Key is not set in the .env file. Please configure it before starting.")
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY_CORE
    )

# --- Core Tool Definitions ---

# OCR Tool (File path version - for main.py or scenarios requiring file paths)
def perform_ocr_filepath(image_path: str) -> str:
    """Perform OCR using pytesseract to recognize Chinese and English characters in the image at the specified path."""    
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        if not text.strip():
            return "OCR failed to recognize any text, or the image is blank."
        return f"Image ({image_path}) OCR recognition results are as follows:\n{text.strip()}"
    except FileNotFoundError:
        return f"OCR recognition failed: Image file {image_path} not found."
    except Exception as e:
        return f"Failed to process image {image_path} with OCR, error message: {str(e)}"

ocr_tool_filepath = Tool.from_function(
    func=perform_ocr_filepath,
    name="ImageFileOCR",
    description="Use this tool when the user provides an image file path and asks to recognize text in the image. For example: 'Please recognize the text in ./example.png'. Input should be a valid local file path string for the image."
)

# Time Tool
def get_current_time_core(_: str = "") -> str:
    now = datetime.now()
    return f"The current time is: {now.strftime('%Y-%m-%d %H:%M:%S')}"

time_tool_core = Tool.from_function(
    func=get_current_time_core,
    name="GetCurrentTime",
    description="Use this tool to get the current system date and time when the user asks for the current time or date. This tool does not require any input."
)

# Search Tool
search_tool_instance_core = DuckDuckGoSearchRun()
search_tool_core = Tool.from_function(
    func=search_tool_instance_core.run,
    name="WebSearch",
    description="Use this tool when you need to answer questions about news, weather, events, people, places, or anything that requires up-to-date information from the internet. The input should be a clear search query."
)

# Calculator Tool
calculator_tool_instance_core = PythonREPLTool()
calculator_tool_core = Tool.from_function(
    func=calculator_tool_instance_core.run,
    name="Calculator",
    description="Use this tool when the user asks to perform mathematical calculations or solve math problems. For example: 'Calculate 123 * (5 + 6)'. Input should be a valid Python mathematical expression."
)

# New: Tool to list directory contents
def list_directory_files_core(_: str = "") -> str:
    """List image, Python script, and Markdown files in the current working directory."""
    try:
        files = os.listdir('.') # Get all files and folders in the current directory
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        py_extensions = ['.py']
        md_extensions = ['.md']
        
        image_files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in image_extensions]
        py_files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in py_extensions]
        md_files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in md_extensions]
        
        if not image_files and not py_files and not md_files:
            return "No specified image, Python, or Markdown files found in the current directory."
        
        response_lines = ["The following files were found in the current working directory:"]
        if image_files:
            response_lines.append("\nImage files:")
            response_lines.extend([f"  - {f}" for f in image_files])
        if py_files:
            response_lines.append("\nPython scripts (.py):")
            response_lines.extend([f"  - {f}" for f in py_files])
        if md_files:
            response_lines.append("\nMarkdown files (.md):")
            response_lines.extend([f"  - {f}" for f in md_files])
            
        return "\n".join(response_lines)
    except Exception as e:
        return f"Error listing directory files: {str(e)}"

list_files_tool_core = Tool.from_function(
    func=list_directory_files_core,
    name="ListDirectoryFiles",
    description="Use this tool when the user asks what image, Python script (.py), or Markdown document (.md) files are in the current project or working directory. This tool does not require any input."
)
# --- Core Tool Definitions End ---

CORE_TOOLS_LIST = [
    search_tool_core,
    calculator_tool_core,
    time_tool_core,
    ocr_tool_filepath,
    list_files_tool_core, # Add new tool to the list
]

# --- Core System Prompt ---
CORE_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Please understand the user's questions and do your best to answer them clearly.\n"
    "When you need to answer questions about news, weather, real-time information about specific locations, or any information that requires the latest updates from the internet, please proactively use the WebSearch tool.\n"
    "For calculation problems, please use the Calculator tool.\n"
    "If you receive an image file path and are asked to recognize the content of the image, please use the ImageFileOCR tool.\n"
    "For time queries, please use the GetCurrentTime tool.\n"
    "If you need to know what image, Python script, or Markdown files are in the current project or working directory, please use the ListDirectoryFiles tool.\n"
    "If the user provides the OCR text content of an image (for example, in a Gradio application, the text will be pre-extracted and provided to you), please directly use that text content for understanding and answering, and do not attempt to call the OCR tool again to process the original image.\n"
    "If necessary, please use the tools you have appropriately to gather information."
)

# --- Agent Creation Function ---
def get_agent_runnable_and_checkpointer(custom_tools=None, custom_prompt=None):
    """
    Creates and returns a configured LangGraph ReAct Agent runnable and an InMemorySaver checkpointer.
    Allows overriding the default tool list and system prompt via parameters.
    """
    llm = get_core_llm()
    tools_to_use = custom_tools if custom_tools is not None else CORE_TOOLS_LIST
    prompt_to_use = custom_prompt if custom_prompt is not None else CORE_SYSTEM_PROMPT
    
    checkpointer = InMemorySaver()
    
    agent_runnable = create_react_agent(
        model=llm,
        tools=tools_to_use,
        checkpointer=checkpointer,
        prompt=prompt_to_use,
        debug=True # Debug is enabled by default, caller can disable or configure it through other means as needed
    )
    return agent_runnable, checkpointer

if __name__ == '__main__':
    # This part is used for directly testing the functionality of core_agent.py
    print("Testing core Agent configuration...")
    test_llm = get_core_llm()
    print(f"LLM type: {type(test_llm)}")
    print(f"Default number of tools: {len(CORE_TOOLS_LIST)}")
    for tool in CORE_TOOLS_LIST:
        print(f" - Tool: {tool.name}, Description: {tool.description[:70]}...") # Increase description length
    print(f"Default system prompt (partial): \n{CORE_SYSTEM_PROMPT[:250]}...") # Increase prompt display length
    
    runnable, chkptr = get_agent_runnable_and_checkpointer()
    print(f"Agent Runnable type: {type(runnable)}")
    print(f"Checkpointer type: {type(chkptr)}")
    
    # Test new tool
    print("\nTesting ListDirectoryFiles tool:")
    # First, create some test files
    test_files_created = []
    try:
        with open("test_image.png", "w") as f: f.write("png_content"); test_files_created.append("test_image.png")
        with open("test_script.py", "w") as f: f.write("py_content"); test_files_created.append("test_script.py")
        with open("test_notes.md", "w") as f: f.write("md_content"); test_files_created.append("test_notes.md")
        with open("other_file.txt", "w") as f: f.write("txt_content"); test_files_created.append("other_file.txt")
        print(list_directory_files_core(""))
    finally:
        for tf in test_files_created:
            if os.path.exists(tf):
                os.remove(tf)
        print("Test files have been cleaned up.")

    print("Core Agent configuration test complete.") 