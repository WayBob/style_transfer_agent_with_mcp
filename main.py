# main.py
# -*- coding: utf-8 -*-

# 标准库导入
import logging
import sys

from langchain_core.messages import HumanMessage
# 从新创建的核心模块导入Agent创建函数和相关常量
from core_agent import get_agent_runnable_and_checkpointer, CORE_SYSTEM_PROMPT, CORE_TOOLS_LIST

# --- 日志记录设置 (这部分保留在main.py) ---
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
logger.info("--- 新的 Agent 会话开始 (main.py) ---")
logger.info(f"标准输出将同时显示在终端并记录到日志文件: {LOG_FILENAME}")

def custom_input_for_logging(prompt_message=""):
    _original_stdout.write(prompt_message)
    _original_stdout.flush()
    logger.info(f"终端提示: {prompt_message.strip()}")
    user_text = _original_input()
    logger.info(f"用户输入: {user_text}")
    return user_text
__builtins__.input = custom_input_for_logging
logger.info("内建 input() 函数已替换，交互将被记录。")
# --- 日志记录设置结束 ---

# --- Agent 初始化 (使用核心模块) ---
# main.py 使用核心模块默认的工具和提示
# 注意: get_agent_runnable_and_checkpointer 默认开启debug
agent_runnable, checkpointer = get_agent_runnable_and_checkpointer()

logger.info("LangGraph ReAct Agent (from core_agent) 初始化完成。")
logger.info(f"使用的工具: {[tool.name for tool in CORE_TOOLS_LIST]}")
logger.info(f"使用的系统提示 (部分): {CORE_SYSTEM_PROMPT[:150]}...")
# --- Agent 初始化结束 ---

# --- 启动命令行交互界面 (大部分逻辑保留) ---
_original_stdout.write("=" * 40 + "\n")
_original_stdout.write("🧠 欢迎使用 LangGraph ReAct Agent (GPT-4o) - CLI 模式\n")
_original_stdout.write("   Agent核心逻辑由 core_agent.py 提供\n")
_original_stdout.write("   具有对话记忆和日志记录功能。\n")
_original_stdout.write("输入你的问题，例如：\n")
_original_stdout.write(" - 现在几点了？\n")
_original_stdout.write(" - 请识别图片文字：[图片文件路径] (例如 ./example.png)\n") # 提示用户输入文件路径
_original_stdout.write(" - 请计算 123 * (5 + 6)\n")
_original_stdout.write(" - 东京今天的天气如何？\n")
_original_stdout.write("输入 '退出' 可结束程序\n")
_original_stdout.write("=" * 40 + "\n")
_original_stdout.flush()

conversation_thread_id = "cli_session_main_thread"
logger.info(f"当前会话线程ID: {conversation_thread_id}")

while True:
    try:
        user_input_text = input("\n💬 你的输入：")

        if user_input_text.lower() in ["exit", "退出", "quit", "q"]:
            logger.info("用户请求退出程序。")
            _original_stdout.write("👋 程序已退出，再见！\n")
            _original_stdout.flush()
            break

        messages_input = [HumanMessage(content=user_input_text)]
        config = {"configurable": {"thread_id": conversation_thread_id, "checkpointer": checkpointer}}
        # 注意：需要将 checkpointer 实例传递给 invoke 的 config 中，如果 agent_runnable 内部没有自动处理
        # LangGraph 的 create_react_agent 当传入 checkpointer 时，会自动处理持久化，通常不需要在invoke时再次指定
        # 但为了明确，可以检查LangGraph文档。这里我们先按标准方式调用。
        # 修正：create_react_agent返回的已经是配置好checkpointer的runnable，config中只需thread_id
        config = {"configurable": {"thread_id": conversation_thread_id}}


        logger.info(f"准备调用 Agent，输入消息: {user_input_text}")
        result_state = agent_runnable.invoke({"messages": messages_input}, config=config)
        logger.info(f"Agent 调用完成，返回状态: {result_state}")

        response_content = "抱歉，我好像遇到了一些麻烦，暂时无法回复您。"
        if result_state and "messages" in result_state and result_state["messages"]:
            ai_message = result_state["messages"][-1]
            if hasattr(ai_message, 'content'):
                response_content = ai_message.content
            else:
                logger.warning(f"Agent返回的最后一条消息没有content属性: {ai_message}")
        else:
            logger.warning(f"Agent返回的状态中没有预期的messages结构: {result_state}")
            
        print(f"\n🤖 Agent 回复：\n{response_content}")

    except KeyboardInterrupt:
        logger.info("用户通过 Ctrl+C 中断程序。")
        _original_stdout.write("\n👋 程序已中断，再见！\n")
        _original_stdout.flush()
        break
    except Exception as e:
        logger.error(f"主循环发生未捕获错误: {str(e)}", exc_info=True)
        print(f"❌ 处理您的请求时出现严重错误：{str(e)}\n   详情请查看日志文件: {LOG_FILENAME}")
# --- 命令行交互界面结束 ---