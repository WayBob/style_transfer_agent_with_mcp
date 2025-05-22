# main.py
# -*- coding: utf-8 -*-

# 标准库导入
import logging
import sys

from langchain_core.messages import HumanMessage
# 从新创建的核心模块导入Agent创建函数和相关常量
from core_agent import get_agent_runnable_and_checkpointer, CORE_SYSTEM_PROMPT, CORE_TOOLS_LIST

# --- Logging Setup (This part is kept in main.py) ---
LOG_FILENAME = 'agent_interaction.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    filename=LOG_FILENAME,
    filemode='a'
)
logger = logging.getLogger(__name__)

class StreamToLogger:
    def __init__(self, logger_instance, original_stream, log_level=logging.INFO):
        self.logger = logger_instance
        self.log_level = log_level
        self.original_stream = original_stream
    def write(self, buf):
        self.original_stream.write(buf)
        self.original_stream.flush()
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
    def flush(self):
        self.original_stream.flush()

_original_stdout = sys.stdout
_original_stderr = sys.stderr
_original_input = __builtins__.input

sys.stdout = StreamToLogger(logger, _original_stdout, logging.INFO)
logger.info("--- New Agent Session Started (main.py) ---")
logger.info(f"Standard output will be displayed in the terminal and logged to the log file: {LOG_FILENAME}")

def custom_input_for_logging(prompt_message=""):
    _original_stdout.write(prompt_message)
    _original_stdout.flush()
    logger.info(f"Terminal prompt: {prompt_message.strip()}")
    user_text = _original_input()
    logger.info(f"User input: {user_text}")
    return user_text
__builtins__.input = custom_input_for_logging
logger.info("Built-in input() function has been replaced, interactions will be logged.")
# --- Logging Setup End ---

# --- Agent Initialization (Using core module) ---
# main.py uses the core module's default tools and prompts
# Note: get_agent_runnable_and_checkpointer has debug enabled by default
agent_runnable, checkpointer = get_agent_runnable_and_checkpointer()

logger.info("LangGraph ReAct Agent (from core_agent) initialization complete.")
logger.info(f"Tools used: {[tool.name for tool in CORE_TOOLS_LIST]}")
logger.info(f"System prompt used (partial): {CORE_SYSTEM_PROMPT[:150]}...")
# --- Agent Initialization End ---

# --- Start Command Line Interface (Most logic retained) ---
_original_stdout.write("=" * 40 + "\n")
_original_stdout.write("🧠 Welcome to LangGraph ReAct Agent (GPT-4o) - CLI Mode\n")
_original_stdout.write("   Agent core logic provided by core_agent.py\n")
_original_stdout.write("   With conversation memory and logging features.\n")
_original_stdout.write("Enter your question, for example:\n")
_original_stdout.write(" - What time is it?\n")
_original_stdout.write(" - Please recognize image text: [Image file path] (e.g., ./example.png)\n") # Prompt user for file path
_original_stdout.write(" - Please calculate 123 * (5 + 6)\n")
_original_stdout.write(" - What is the weather like in Tokyo today?\n")
_original_stdout.write("Enter 'exit' to end the program\n")
_original_stdout.write("=" * 40 + "\n")
_original_stdout.flush()

conversation_thread_id = "cli_session_main_thread"
logger.info(f"Current session thread ID: {conversation_thread_id}")

while True:
    try:
        user_input_text = input("\n💬 Your input: ")

        if user_input_text.lower() in ["exit", "quit", "q"]:
            logger.info("User requested to exit the program.")
            _original_stdout.write("👋 Program exited, goodbye!\n")
            _original_stdout.flush()
            break

        messages_input = [HumanMessage(content=user_input_text)]
        config = {"configurable": {"thread_id": conversation_thread_id, "checkpointer": checkpointer}}
        # Note: The checkpointer instance needs to be passed to invoke's config if not automatically handled inside agent_runnable
        # LangGraph's create_react_agent, when passed a checkpointer, automatically handles persistence, usually no need to specify again during invoke
        # But for clarity, you can check the LangGraph documentation. Here we will call it in the standard way first.
        # Correction: create_react_agent returns a runnable already configured with a checkpointer, only thread_id is needed in config
        config = {"configurable": {"thread_id": conversation_thread_id}}


        logger.info(f"Preparing to call Agent, input message: {user_input_text}")
        result_state = agent_runnable.invoke({"messages": messages_input}, config=config)
        logger.info(f"Agent call complete, returned state: {result_state}")

        response_content = "Sorry, I seem to have encountered some trouble and cannot reply to you at the moment."
        if result_state and "messages" in result_state and result_state["messages"]:
            ai_message = result_state["messages"][-1]
            if hasattr(ai_message, 'content'):
                response_content = ai_message.content
            else:
                logger.warning(f"The last message returned by the Agent does not have a content attribute: {ai_message}")
        else:
            logger.warning(f"The state returned by the Agent does not have the expected messages structure: {result_state}")
            
        print(f"\n🤖 Agent Reply:\n{response_content}")

    except KeyboardInterrupt:
        logger.info("User interrupted the program via Ctrl+C.")
        _original_stdout.write("\n👋 Program interrupted, goodbye!\n")
        _original_stdout.flush()
        break
    except Exception as e:
        logger.error(f"Uncaught error in main loop: {str(e)}", exc_info=True)
        print(f"❌ A critical error occurred while processing your request: {str(e)}\n   Please check the log file for details: {LOG_FILENAME}")
# --- Command Line Interface End ---