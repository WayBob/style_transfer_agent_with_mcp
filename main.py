# main.py
# -*- coding: utf-8 -*-

# 标准库导入
import logging
import sys
import os
from datetime import datetime

# 第三方库导入
from dotenv import load_dotenv
from PIL import Image
import pytesseract

from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

# --- 日志记录设置 ---
load_dotenv()  # 从 .env 文件加载环境变量，如 API Key

LOG_FILENAME = 'agent_interaction.log' # 定义日志文件名

# 配置基础日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志记录级别为INFO及以上
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # 定义日志输出格式
    filename=LOG_FILENAME, # 指定日志文件
    filemode='a'  # 日志文件打开模式: 'a'为追加, 'w'为覆盖
)
logger = logging.getLogger(__name__) # 获取当前模块的logger实例

class StreamToLogger:
    """一个自定义流处理器，将写入操作同时输出到原始流（如终端）和日志文件。"""
    def __init__(self, logger_instance, original_stream, log_level=logging.INFO):
        self.logger = logger_instance
        self.log_level = log_level
        self.original_stream = original_stream # 保存原始输出流（例如 sys.stdout）

    def write(self, buf):
        self.original_stream.write(buf) # 首先，将内容写入原始流（显示在终端）
        self.original_stream.flush()
        for line in buf.rstrip().splitlines(): # 然后，逐行写入日志文件
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        self.original_stream.flush() # 确保原始流也被刷新

# 保存 Python 内建的原始标准输出、错误输出和输入函数
_original_stdout = sys.stdout
_original_stderr = sys.stderr
_original_input = __builtins__.input

# 重定向标准输出到 StreamToLogger 实例，实现双重输出
sys.stdout = StreamToLogger(logger, _original_stdout, logging.INFO)
# logger.info("标准错误输出也将被重定向并记录。") # 如需记录错误输出，取消此行和下一行的注释
# sys.stderr = StreamToLogger(logger, _original_stderr, logging.ERROR)

logger.info("--- 新的 Agent 会话开始 ---")
logger.info(f"标准输出将同时显示在终端并记录到日志文件: {LOG_FILENAME}")

def custom_input_for_logging(prompt_message=""):
    """自定义输入函数，在终端显示提示、记录提示及用户输入到日志。"""
    _original_stdout.write(prompt_message) # 在终端显示输入提示
    _original_stdout.flush()
    logger.info(f"终端提示: {prompt_message.strip()}") # 记录提示信息
    user_text = _original_input() # 通过原始input获取用户输入
    logger.info(f"用户输入: {user_text}") # 记录用户输入内容
    return user_text

__builtins__.input = custom_input_for_logging # 全局替换内建input函数
logger.info("内建 input() 函数已替换，交互将被记录。")
# --- 日志记录设置结束 ---

# --- OpenAI API Key 和模型初始化 ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("错误：OPENAI_API_KEY 未在 .env 文件中设置。")
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o", # 指定使用的 GPT 模型
    temperature=0,   # 设置温度参数，0表示更倾向于确定性输出
    openai_api_key=openai_api_key
)
logger.info(f"OpenAI GPT 模型 ({llm.model_name}) 初始化完成。")
# --- 模型初始化结束 ---

# --- 工具定义 ---
# OCR 工具：识别图片中的文字
def perform_ocr(image_path: str) -> str:
    """使用 pytesseract 执行 OCR，识别图片中的中英文字符。"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='chi_sim+eng') # 指定识别中英文
        logger.info(f"OCR 工具成功处理图片: {image_path}")
        return f"OCR识别结果如下：\n{text.strip()}"
    except FileNotFoundError:
        logger.error(f"OCR 错误: 图片文件未找到 {image_path}")
        return f"OCR识别失败：图片文件 {image_path} 未找到。"
    except Exception as e:
        logger.error(f"OCR 工具执行失败 ({image_path}): {str(e)}")
        return f"OCR识别失败，错误信息：{str(e)}"

ocr_tool = Tool.from_function(
    func=perform_ocr,
    name="ImageOCR",
    description="当用户提供图片路径并要求识别图像中的文字时，使用此工具。例如：'请识别 example.png 中的文字'。输入应该是图片的有效路径字符串。"
)

# 时间工具：获取当前系统时间
def get_current_time(_: str = "") -> str:
    """获取并格式化当前的系统日期和时间。"""
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"时间工具被调用，当前时间: {formatted_time}")
    return f"当前时间是：{formatted_time}"

time_tool = Tool.from_function(
    func=get_current_time,
    name="GetCurrentTime",
    description="当用户询问当前时间或日期时，使用此工具获取当前的系统日期和时间。此工具不需要任何输入。"
)

# 网络搜索工具 (DuckDuckGo)
search_tool_instance = DuckDuckGoSearchRun()
search_tool = Tool.from_function(
    func=search_tool_instance.run,
    name="WebSearch",
    description="当你需要回答关于新闻、天气、事件、人物、地点或任何需要从互联网获取最新信息的问题时，使用此工具。输入应为清晰的搜索查询。"
)

# Python 计算器工具 (PythonREPL)
calculator_tool_instance = PythonREPLTool()
calculator_tool = Tool.from_function(
    func=calculator_tool_instance.run,
    name="Calculator",
    description="当用户要求执行数学计算或解答数学问题时，使用此工具。例如：'计算 123 * (5 + 6)'。输入应为有效的 Python 数学表达式。"
)

# 工具列表：汇总所有 Agent 可用的工具
tools = [
    search_tool,
    calculator_tool,
    ocr_tool,
    time_tool,
]
logger.info(f"Agent 工具已定义并汇总: {[tool.name for tool in tools]}")
# --- 工具定义结束 ---

# --- LangGraph Agent 设置 ---
# 初始化对话记忆的 checkpointer (使用内存存储)
checkpointer = InMemorySaver()

# 定义 Agent 的系统提示，指导其行为和回复风格
system_prompt = (
    "你是一个乐于助人的人工智能助手。请理解用户的问题并尽力用中文清晰地回答。"
    "当你需要回答关于新闻、天气、特定地点实时信息或任何需要从互联网获取最新信息的问题时，请主动使用 WebSearch 工具。"
    "对于计算问题，请使用 Calculator 工具。对于图片文字识别，请使用 ImageOCR 工具。对于时间查询，请使用 GetCurrentTime 工具。"
    "如果需要，请恰当地使用你拥有的工具来搜集信息。"
)

# 创建 LangGraph ReAct Agent 实例
agent_runnable = create_react_agent(
    model=llm, # 指定语言模型
    tools=tools, # 提供可用工具列表
    checkpointer=checkpointer, # 设置对话记忆检查点
    prompt=system_prompt, # 设置系统提示
    debug=True # 开启 LangGraph 调试模式，输出详细执行步骤
)
logger.info("LangGraph ReAct Agent 初始化完成，已启用调试模式和对话记忆。")
# --- LangGraph Agent 设置结束 ---

# --- 启动命令行交互界面 ---
# 打印欢迎信息和使用示例
_original_stdout.write("=" * 40 + "\n")
_original_stdout.write("🧠 欢迎使用 LangGraph ReAct Agent (GPT-4o)\n")
_original_stdout.write("   基于 LangGraph 构建，具有对话记忆和日志记录功能。\n")
_original_stdout.write("输入你的问题，例如：\n")
_original_stdout.write(" - 现在几点了？\n")
_original_stdout.write(" - 请识别图片文字：[图片路径]\n")
_original_stdout.write(" - 请计算 123 * (5 + 6)\n")
_original_stdout.write(" - 东京今天的天气如何？\n")
_original_stdout.write(" - 你好，你叫什么名字？\n")
_original_stdout.write("输入 '退出' 可结束程序\n")
_original_stdout.write("=" * 40 + "\n")
_original_stdout.flush()

# 为当前命令行会话设置固定的线程ID，以启用和保持对话记忆
# InMemorySaver 的记忆仅在程序单次运行期间有效
conversation_thread_id = "cli_session_main_thread"
logger.info(f"当前会话线程ID: {conversation_thread_id}")

# 主循环，持续接收用户输入并由 Agent 处理
while True:
    try:
        user_input_text = input("\n💬 你的输入：") # 使用自定义的input函数获取输入

        if user_input_text.lower() in ["exit", "退出", "quit", "q"]:
            logger.info("用户请求退出程序。")
            _original_stdout.write("👋 程序已退出，再见！\n")
            _original_stdout.flush()
            break

        # 将用户输入包装成 LangGraph Agent期望的 HumanMessage 格式
        messages_input = [HumanMessage(content=user_input_text)]
        
        # 构建 Agent 调用所需的配置，主要包含用于对话记忆的 thread_id
        config = {"configurable": {"thread_id": conversation_thread_id}}

        logger.info(f"准备调用 Agent，输入消息: {user_input_text}")
        # 调用 Agent (runnable) 处理输入
        result_state = agent_runnable.invoke({"messages": messages_input}, config=config)
        logger.info(f"Agent 调用完成，返回状态: {result_state}")

        # 从 Agent 返回的状态中提取最新的 AI 回复
        response_content = "抱歉，我好像遇到了一些麻烦，暂时无法回复您。"
        if result_state and "messages" in result_state and result_state["messages"]:
            ai_message = result_state["messages"][-1] # 通常最后一条是AI的回复
            if hasattr(ai_message, 'content'):
                response_content = ai_message.content
            else:
                logger.warning(f"Agent返回的最后一条消息没有content属性: {ai_message}")
        else:
            logger.warning(f"Agent返回的状态中没有预期的messages结构: {result_state}")
            
        # 将 Agent 的回复打印到终端 (会被 StreamToLogger 捕获并记录)
        print(f"\n🤖 Agent 回复：\n{response_content}")

    except KeyboardInterrupt:
        logger.info("用户通过 Ctrl+C 中断程序。")
        _original_stdout.write("\n👋 程序已中断，再见！\n")
        _original_stdout.flush()
        break
    except Exception as e:
        logger.error(f"主循环发生未捕获错误: {str(e)}", exc_info=True) # exc_info=True 会记录堆栈跟踪
        # 仍将错误信息打印到终端，确保用户可见
        # 注意：如果 stderr 也被重定向，这将写入日志；否则，写入原始 stderr (终端)
        print(f"❌ 处理您的请求时出现严重错误：{str(e)}\n   详情请查看日志文件: {LOG_FILENAME}")
# --- 命令行交互界面结束 ---