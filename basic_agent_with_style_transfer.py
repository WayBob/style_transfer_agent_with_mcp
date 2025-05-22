"""
Fixed version of Basic Agent with Style Transfer Tool
This version properly handles tool inputs for style transfer
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pytz
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Import the style transfer tool
from style_transfer_tool import style_transfer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,  # 降低温度以获得更一致的输出
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
        # 安全评估数学表达式
        import ast
        import operator as op
        
        # 定义允许的操作符
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }
        
        def eval_expr(expr):
            """
            >>> eval_expr('2^3')
            8
            >>> eval_expr('2+3*4')
            14
            """
            return eval_(ast.parse(expr, mode='eval').body)
        
        def eval_(node):
            if isinstance(node, ast.Num):  # <number>
                return node.n
            elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                return operators[type(node.op)](eval_(node.left), eval_(node.right))
            elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
                return operators[type(node.op)](eval_(node.operand))
            else:
                raise TypeError(node)
        
        result = eval_expr(expression.replace('^', '**'))
        return str(result)
    except Exception as e:
        # 如果安全评估失败，使用原始的 eval（仅用于简单数学表达式）
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误：{str(e)}"

# 组合所有工具
tools = [
    ocr_tool,
    get_current_time,
    search,
    calculator,
    style_transfer
]

# 使用结构化的聊天代理模板
system_prompt = """你是一个有帮助的AI助手。你可以使用以下工具来帮助回答问题：

{tools}

使用 json blob 来指定一个工具，通过提供 action 键（工具名称）和 action_input 键（工具输入）。

有效的 "action" 值：{tool_names} 或 "Final Answer"

每个 JSON_BLOB 只提供一个动作，格式如下：

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

请遵循以下格式：

问题：需要回答的输入问题
思考：考虑之前和后续的步骤
动作：
```
$JSON_BLOB
```
观察：动作结果
...（重复 思考/动作/观察 N 次）
思考：我知道该如何回答了
动作：
```
{{
  "action": "Final Answer",
  "action_input": "对人类的最终回答"
}}
```

开始！记住始终以有效的 json blob 响应单个动作。如有必要使用工具。如果合适，可以直接回答。
格式是 动作:```$JSON_BLOB```然后是 观察

特别注意工具的输入格式：
- style_transfer 工具需要 content_image_path 和 style_image_path 两个参数
- 所有参数都应该是正确的类型（字符串、数字等）
"""

human_prompt = """{input}

{agent_scratchpad}

（提醒：无论如何都要以 JSON blob 格式响应）"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", human_prompt),
])

# 创建结构化聊天代理
agent = create_structured_chat_agent(
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
    max_iterations=5,
    return_intermediate_steps=True,
)

def main():
    print("AI助手已启动！输入 'quit' 或 'exit' 退出。")
    print("我现在可以进行风格转换了！试试让我将一张图片转换成另一种艺术风格。")
    print("-" * 50)
    
    # 示例对话
    example_prompts = [
        "请将 StyTR-2/demo/c_img/2_10_0_0_512_512.png 转换成 StyTR-2/demo/s_img/LevelSequence_Vaihingen.0002.png 的艺术风格",
        "现在几点了？",
        "帮我搜索一下最新的AI技术发展",
        "计算 1234 * 5678"
    ]
    
    print("\n示例问题：")
    for i, prompt_text in enumerate(example_prompts, 1):
        print(f"{i}. {prompt_text}")
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
            
            # 调试：显示中间步骤
            if 'intermediate_steps' in result and result['intermediate_steps']:
                print("\n[调试信息] 中间步骤:")
                for i, step in enumerate(result['intermediate_steps']):
                    action, observation = step
                    print(f"步骤 {i+1}:")
                    print(f"  工具: {action.tool}")
                    print(f"  输入: {action.tool_input}")
                    print(f"  结果: {observation[:100]}..." if len(str(observation)) > 100 else f"  结果: {observation}")
                    
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            print(f"抱歉，处理您的请求时出现错误：{str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 