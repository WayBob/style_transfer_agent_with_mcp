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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,  # Lower temperature for more consistent output
    api_key=os.getenv("OPENAI_API_KEY"),
)

# OCR tool
@tool
def ocr_tool(image_path: str) -> str:
    """Perform OCR on an image and return the recognized text content."""
    return f"[Simulated OCR Result] This is the text content of image {image_path}: Sample Text"

# Get time tool
@tool
def get_current_time() -> str:
    """Get the current Beijing time."""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(beijing_tz)
    return current_time.strftime("%Y-%m-%d %H:%M:%S")

# Web search tool
search = TavilySearchResults(
    max_results=5,
    description="Search the web for relevant information. Input should be a search query."
)

# Calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate the result of a mathematical expression. Input should be a valid mathematical expression, e.g., 2+2, 10*5, 100/4, etc."""
    try:
        # Safely evaluate mathematical expression
        import ast
        import operator as op
        
        # Define allowed operators
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
        # If safe evaluation fails, use original eval (for simple math expressions only)
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Calculation error: {str(e)}"

# Combine all tools
tools = [
    ocr_tool,
    get_current_time,
    search,
    calculator,
    style_transfer
]

# Use structured chat agent template
system_prompt = """You are a helpful AI assistant. You can use the following tools to help answer questions:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: {tool_names} or "Final Answer"

Provide only one action per JSON_BLOB, in the following format:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: The input question to answer
Thought: Consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: The result of the action
... (repeat Thought/Action/Observation N times)
Thought: I know how to answer now
Action:
```
{{
  "action": "Final Answer",
  "action_input": "The final answer to the human"
}}
```

Begin! Remember to always respond with a single action in a valid json blob. Use tools if necessary. If appropriate, answer directly.
The format is Action:```$JSON_BLOB``` followed by Observation

Pay special attention to the input format of the tools:
- The style_transfer tool requires content_image_path and style_image_path parameters
- All parameters should be of the correct type (string, number, etc.)
"""

human_prompt = """{input}

{agent_scratchpad}

(Reminder: Respond in JSON blob format no matter what)"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", human_prompt),
])

# Create structured chat agent
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    return_intermediate_steps=True,
)

def main():
    print("AI assistant started! Type 'quit' or 'exit' to quit.")
    print("I can now perform style transfer! Try asking me to transfer the style of one image to another.")
    print("-" * 50)
    
    # Example conversation
    example_prompts = [
        "Please transfer the style of StyTR-2/demo/c_img/2_10_0_0_512_512.png to StyTR-2/demo/s_img/LevelSequence_Vaihingen.0002.png",
        "What time is it?",
        "Help me search for the latest AI technology developments",
        "Calculate 1234 * 5678"
    ]
    
    print("\nExample questions:")
    for i, prompt_text in enumerate(example_prompts, 1):
        print(f"{i}. {prompt_text}")
    print("-" * 50)
    
    while True:
        user_input = input("\nPlease enter your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Run agent
            result = agent_executor.invoke({"input": user_input})
            print(f"\nAnswer: {result['output']}")
            
            # Debug: Show intermediate steps
            if 'intermediate_steps' in result and result['intermediate_steps']:
                print("\n[Debug Info] Intermediate steps:")
                for i, step in enumerate(result['intermediate_steps']):
                    action, observation = step
                    print(f"Step {i+1}:")
                    print(f"  Tool: {action.tool}")
                    print(f"  Input: {action.tool_input}")
                    print(f"  Result: {str(observation)[:100]}..." if len(str(observation)) > 100 else f"  Result: {observation}")
                    
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            print(f"Sorry, an error occurred while processing your request: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 