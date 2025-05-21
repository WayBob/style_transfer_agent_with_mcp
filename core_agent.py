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

# --- 环境和模型初始化 ---
load_dotenv() # 确保环境变量被加载

OPENAI_API_KEY_CORE = os.getenv("OPENAI_API_KEY")

def get_core_llm():
    """返回一个配置好的核心 ChatOpenAI LLM 实例。"""
    if not OPENAI_API_KEY_CORE:
        raise ValueError("OpenAI API Key 未在 .env 文件中设置。请在启动前配置。")
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY_CORE
    )

# --- 核心工具定义 ---

# OCR 工具 (文件路径版 - 用于 main.py 或需要文件路径的场景)
def perform_ocr_filepath(image_path: str) -> str:
    """使用 pytesseract 执行 OCR，识别指定路径图片中的中英文字符。"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        if not text.strip():
            return "OCR 未能识别到任何文字，或图片为空白。"
        return f"图片 ({image_path}) OCR识别结果如下：\n{text.strip()}"
    except FileNotFoundError:
        return f"OCR识别失败：图片文件 {image_path} 未找到。"
    except Exception as e:
        return f"OCR处理图片 {image_path} 失败，错误信息：{str(e)}"

ocr_tool_filepath = Tool.from_function(
    func=perform_ocr_filepath,
    name="ImageFileOCR",
    description="当用户提供图片文件路径并要求识别图像中的文字时，使用此工具。例如：'请识别 ./example.png 中的文字'。输入应该是图片的有效本地文件路径字符串。"
)

# 时间工具
def get_current_time_core(_: str = "") -> str:
    now = datetime.now()
    return f"当前时间是：{now.strftime('%Y-%m-%d %H:%M:%S')}"

time_tool_core = Tool.from_function(
    func=get_current_time_core,
    name="GetCurrentTime",
    description="当用户询问当前时间或日期时，使用此工具获取当前的系统日期和时间。此工具不需要任何输入。"
)

# 搜索工具
search_tool_instance_core = DuckDuckGoSearchRun()
search_tool_core = Tool.from_function(
    func=search_tool_instance_core.run,
    name="WebSearch",
    description="当你需要回答关于新闻、天气、事件、人物、地点或任何需要从互联网获取最新信息的问题时，使用此工具。输入应为清晰的搜索查询。"
)

# 计算器工具
calculator_tool_instance_core = PythonREPLTool()
calculator_tool_core = Tool.from_function(
    func=calculator_tool_instance_core.run,
    name="Calculator",
    description="当用户要求执行数学计算或解答数学问题时，使用此工具。例如：'计算 123 * (5 + 6)'。输入应为有效的 Python 数学表达式。"
)

# 新增：列出目录内容的工具
def list_directory_files_core(_: str = "") -> str:
    """列出当前工作目录下的图片、Python脚本和Markdown文件。"""
    try:
        files = os.listdir('.') # 获取当前目录所有文件和文件夹
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        py_extensions = ['.py']
        md_extensions = ['.md']
        
        image_files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in image_extensions]
        py_files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in py_extensions]
        md_files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in md_extensions]
        
        if not image_files and not py_files and not md_files:
            return "当前目录中未找到指定的图片、Python或Markdown文件。"
        
        response_lines = ["在当前工作目录中找到以下文件："]
        if image_files:
            response_lines.append("\n图片文件:")
            response_lines.extend([f"  - {f}" for f in image_files])
        if py_files:
            response_lines.append("\nPython 脚本 (.py):")
            response_lines.extend([f"  - {f}" for f in py_files])
        if md_files:
            response_lines.append("\nMarkdown 文件 (.md):")
            response_lines.extend([f"  - {f}" for f in md_files])
            
        return "\n".join(response_lines)
    except Exception as e:
        return f"列出目录文件时出错: {str(e)}"

list_files_tool_core = Tool.from_function(
    func=list_directory_files_core,
    name="ListDirectoryFiles",
    description="当用户询问当前项目或工作目录下有哪些图片、Python脚本(.py)或Markdown文档(.md)时使用此工具。此工具不需要任何输入。"
)
# --- 核心工具定义结束 ---

CORE_TOOLS_LIST = [
    search_tool_core,
    calculator_tool_core,
    time_tool_core,
    ocr_tool_filepath,
    list_files_tool_core, # 添加新工具到列表
]

# --- 核心系统提示 ---
CORE_SYSTEM_PROMPT = (
    "你是一个乐于助人的人工智能助手。请理解用户的问题并尽力用中文清晰地回答。"
    "当你需要回答关于新闻、天气、特定地点实时信息或任何需要从互联网获取最新信息的问题时，请主动使用 WebSearch 工具。"
    "对于计算问题，请使用 Calculator 工具。"
    "如果你接收到的是图片文件路径，并且被要求识别图片内容，请使用 ImageFileOCR 工具。"
    "对于时间查询，请使用 GetCurrentTime 工具。"
    "如果你需要知道当前项目或工作目录下有哪些图片、Python脚本或Markdown文件，请使用 ListDirectoryFiles 工具。"
    "如果用户提供了图片OCR的文本内容（例如，在Gradio应用中，文本会预先提取并提供给你），请直接使用该文本内容进行理解和回答，不要尝试再次调用OCR工具处理原始图片。"
    "如果需要，请恰当地使用你拥有的工具来搜集信息。"
)

# --- Agent 创建函数 ---
def get_agent_runnable_and_checkpointer(custom_tools=None, custom_prompt=None):
    """
    创建并返回一个配置好的 LangGraph ReAct Agent runnable 和一个 InMemorySaver checkpointer。
    允许通过参数覆盖默认的工具列表和系统提示。
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
        debug=True # 默认开启debug，调用者可按需关闭或通过其他方式配置
    )
    return agent_runnable, checkpointer

if __name__ == '__main__':
    # 此部分用于直接测试 core_agent.py 的功能
    print("测试核心 Agent 配置...")
    test_llm = get_core_llm()
    print(f"LLM 类型: {type(test_llm)}")
    print(f"默认工具数量: {len(CORE_TOOLS_LIST)}")
    for tool in CORE_TOOLS_LIST:
        print(f" - 工具: {tool.name}, 描述: {tool.description[:70]}...") # 增加描述长度
    print(f"默认系统提示 (部分): \n{CORE_SYSTEM_PROMPT[:250]}...") # 增加提示显示长度
    
    runnable, chkptr = get_agent_runnable_and_checkpointer()
    print(f"Agent Runnable 类型: {type(runnable)}")
    print(f"Checkpointer 类型: {type(chkptr)}")
    
    # 测试新工具
    print("\n测试 ListDirectoryFiles 工具:")
    # 先创建一些测试文件
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
        print("测试文件已清理。")

    print("核心 Agent 配置测试完成。") 