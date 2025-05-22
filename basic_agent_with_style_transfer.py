"""
Basic Agent with Style Transfer Tool
This is an enhanced version of basic_agent.py that includes style transfer functionality
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from urllib import parse

import pytz
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Import the style transfer tool
from style_transfer_tool import style_transfer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redirect stdout to log
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            self.level(message.rstrip())

    def flush(self):
        pass

sys.stdout = LoggerWriter(logger.info)

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# OCR 工具
@tool
def ocr_tool(image_path: str) -> str:
    """对图片进行OCR文字识别，返回识别到的文字内容。"""
    return f"[模拟OCR结果] 这是图片 {image_path} 的文字内容：示例文本"

# 获取时间工具
@tool
def get_current_time() -> str:
    """获取当前的北京时间。"""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(beijing_tz)
    return current_time.strftime("%Y年%m月%d日 %H:%M:%S")

# 网页搜索工具
search = TavilySearchResults(
    max_results=5,
    description="搜索网页获取相关信息。输入应该是一个搜索查询。"
)

# 计算器工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果。输入应该是一个有效的数学表达式，例如：2+2、10*5、100/4等。"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

# 组合所有工具（包括风格转换工具）
tools = [
    ocr_tool,
    get_current_time,
    search,
    calculator,
    style_transfer  # 添加风格转换工具
]

# 创建 ReAct prompt 模板
prompt_template = """你是一个有帮助的AI助手。你可以使用以下工具来帮助回答问题：

{tools}

使用以下格式回答：

Question: 需要回答的问题
Thought: 你应该思考要做什么
Action: 要采取的动作，应该是 [{tool_names}] 中的一个
Action Input: 动作的输入
Observation: 动作的结果
... (这个 Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 原始问题的最终答案

注意：
1. 请用中文回答用户的问题
2. 在使用工具时，确保提供正确的输入格式
3. 如果需要进行风格转换，使用 style_transfer 工具，需要提供内容图片路径和风格图片路径

开始！

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["input", "agent_scratchpad"],
    partial_variables={
        "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        "tool_names": ", ".join([tool.name for tool in tools])
    }
)

# 创建 agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 创建 agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

def main():
    print("AI助手已启动！输入 'quit' 或 'exit' 退出。")
    print("我现在可以进行风格转换了！试试让我将一张图片转换成另一种艺术风格。")
    print("-" * 50)
    
    # 示例对话
    example_prompts = [
        "请将 StyTR-2/demo/image_c/2_10_0_0_512_512.png 转换成 StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png 的艺术风格",
        "现在几点了？",
        "帮我搜索一下最新的AI技术发展",
        "计算 1234 * 5678"
    ]
    
    print("\n示例问题：")
    for i, prompt in enumerate(example_prompts, 1):
        print(f"{i}. {prompt}")
    print("-" * 50)
    
    while True:
        user_input = input("\n请输入您的问题: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        try:
            # 运行 agent
            result = agent_executor.invoke({"input": user_input})
            print(f"\n回答: {result['output']}")
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            print(f"抱歉，处理您的请求时出现错误：{str(e)}")

if __name__ == "__main__":
    main() 