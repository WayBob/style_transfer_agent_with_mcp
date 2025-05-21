# gradio_app.py
# -*- coding: utf-8 -*-

import gradio as gr
import os
from datetime import datetime
from PIL import Image
import pytesseract
import uuid # 用于生成独立的会话ID

from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

# --- 环境和模型设置 ---
load_dotenv() # 从 .env 文件加载 API Key

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    # 在Gradio应用中，更友好的错误提示可能是在UI上显示，但这里我们先用异常
    raise ValueError("OpenAI API Key 未在 .env 文件中设置。请在启动前配置。")

llm_gradio = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=openai_api_key
)

# --- Gradio Agent 工具定义 ---

# 调整后的 OCR 工具：直接处理 PIL Image 对象
def perform_ocr_gradio(pil_image: Image.Image) -> str:
    """使用 pytesseract 执行 OCR，识别 PIL Image 对象中的中英文字符。"""
    if pil_image is None:
        return "错误：没有提供图像进行OCR。"
    try:
        # pytesseract可以直接处理PIL Image对象
        text = pytesseract.image_to_string(pil_image, lang='chi_sim+eng')
        if not text.strip():
            return "OCR未能识别到任何文字，或者图片为空白。"
        return f"图像OCR识别结果如下：\n{text.strip()}"
    except Exception as e:
        return f"OCR处理失败，错误信息：{str(e)}"

# 注意：Agent本身不会直接调用这个PIL Image版本的OCR工具，
# Gradio界面会在调用Agent前处理图片，并将文本结果融入输入。
# 但我们仍然可以定义一个基于此函数的工具，以备将来Agent可能需要处理已加载图片的情况。
# 或者，如果Agent的逻辑被设计为接收一个特殊的输入键（例如image_data），那它可以调用此类工具。
# 为简单起见，在此Gradio应用中，OCR主要在Agent调用之前由UI逻辑完成。

# 时间工具
def get_current_time_gradio(_: str = "") -> str:
    now = datetime.now()
    return f"当前时间是：{now.strftime('%Y-%m-%d %H:%M:%S')}"

time_tool_gradio = Tool.from_function(
    func=get_current_time_gradio,
    name="GetCurrentTime",
    description="当用户询问当前时间或日期时，使用此工具获取当前的系统日期和时间。此工具不需要任何输入。"
)

# 搜索工具
search_tool_instance_gradio = DuckDuckGoSearchRun()
search_tool_gradio = Tool.from_function(
    func=search_tool_instance_gradio.run,
    name="WebSearch",
    description="当你需要回答关于新闻、天气、事件、人物、地点或任何需要从互联网获取最新信息的问题时，使用此工具。输入应为清晰的搜索查询。"
)

# 计算器工具
calculator_tool_instance_gradio = PythonREPLTool()
calculator_tool_gradio = Tool.from_function(
    func=calculator_tool_instance_gradio.run,
    name="Calculator",
    description="当用户要求执行数学计算或解答数学问题时，使用此工具。例如：'计算 123 * (5 + 6)'。输入应为有效的 Python 数学表达式。"
)

# Gradio Agent 使用的工具列表 (不直接包含PIL Image OCR工具，因为OCR在外部处理)
tools_gradio = [
    search_tool_gradio,
    calculator_tool_gradio,
    time_tool_gradio,
    # 如果需要Agent直接处理图像（需要更复杂的Agent逻辑），可以添加一个不同的OCR工具定义
]

# --- LangGraph Agent 设置 (Gradio版) ---
checkpointer_gradio = InMemorySaver()

system_prompt_gradio = (
    "你是一个乐于助人的人工智能助手。请理解用户的问题并尽力用中文清晰地回答。"
    "用户可能会提供图片OCR识别的文本内容，请基于这些信息进行理解和回答。"
    "当你需要回答关于新闻、天气、特定地点实时信息或任何需要从互联网获取最新信息的问题时，请主动使用 WebSearch 工具。"
    "对于计算问题，请使用 Calculator 工具。对于时间查询，请使用 GetCurrentTime 工具。"
    "如果需要，请恰当地使用你拥有的工具来搜集信息。"
)

agent_runnable_gradio = create_react_agent(
    model=llm_gradio,
    tools=tools_gradio,
    checkpointer=checkpointer_gradio,
    prompt=system_prompt_gradio,
    debug=True # 在Gradio后端控制台查看Agent执行步骤
)

# --- Gradio 交互逻辑 ---
def agent_chat_interface(user_message: str, history: list, image_upload: Image.Image, session_id: str):
    """
    处理用户输入，与LangGraph Agent交互，并返回更新后的聊天历史。
    history: Gradio的聊天历史，格式为 [[user_msg1, bot_msg1], [user_msg2, bot_msg2], ...]
    session_id: 用于LangGraph Agent会话记忆的唯一ID。
    """
    # print(f"[Gradio DEBUG] Session ID: {session_id}")
    # print(f"[Gradio DEBUG] Received message: {user_message}")
    # print(f"[Gradio DEBUG] History: {history}")
    # print(f"[Gradio DEBUG] Image uploaded: {image_upload is not None}")

    final_user_input = user_message

    # 1. 如果有图片上传，先进行OCR处理
    if image_upload is not None:
        ocr_result_text = perform_ocr_gradio(image_upload)
        # 将OCR结果和用户原始消息结合
        if user_message.strip(): # 如果用户同时输入了文字
            final_user_input = f"{user_message}\n\n[附加图片OCR内容]:\n{ocr_result_text}"
        else: # 如果用户只上传了图片而没有文字
            final_user_input = f"[图片OCR内容]:\n{ocr_result_text}"
        # print(f"[Gradio DEBUG] OCR Result: {ocr_result_text}")
        # print(f"[Gradio DEBUG] Final user input with OCR: {final_user_input}")
    
    if not final_user_input.strip() and image_upload is None:
        # 如果既没有文本输入也没有图片，可以不调用agent或返回提示
        history.append((user_message, "请输入您的问题或上传一张图片。"))
        return history, session_id # 返回原始历史和session_id

    # 2. 调用Agent
    messages_input = [HumanMessage(content=final_user_input)]
    config = {"configurable": {"thread_id": session_id}}
    
    bot_response_content = "抱歉，处理您的请求时出现了一些问题。"
    try:
        result_state = agent_runnable_gradio.invoke({"messages": messages_input}, config=config)
        
        if result_state and "messages" in result_state and result_state["messages"]:
            ai_message = result_state["messages"][-1]
            if hasattr(ai_message, 'content') and ai_message.content:
                bot_response_content = ai_message.content
            # else: print(f"[Gradio DEBUG] Agent's last message no content: {ai_message}")
        # else: print(f"[Gradio DEBUG] Agent state no messages: {result_state}")

    except Exception as e:
        bot_response_content = f"调用Agent时出错: {str(e)}"
        print(f"[Gradio ERROR] Error invoking agent: {e}")
        import traceback
        traceback.print_exc()

    history.append((user_message if not image_upload else f"{user_message} (附带图片)", bot_response_content))
    # print(f"[Gradio DEBUG] Updated history: {history}")
    return history, session_id # 确保session_id状态被传回并保持

# --- Gradio UI 构建 ---
with gr.Blocks(theme=gr.themes.Soft(), title="LangGraph ReAct Agent") as demo:
    gr.Markdown("## 🧠 LangGraph ReAct Agent (GPT-4o) - Web UI")
    gr.Markdown(
        "与基于LangGraph和GPT-4o构建的智能助手进行交互。" 
        "您可以提问、进行计算、查询时间、搜索网页，或上传图片进行文字识别。"
    )

    # 用于存储每个用户会话的唯一ID (LangGraph thread_id)
    # Gradio的State在每个用户的浏览器会话中是独立的
    session_id_state = gr.State(lambda: str(uuid.uuid4()))

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="聊天记录", 
                bubble_full_width=False,
                avatar_images=(None, "https://raw.githubusercontent.com/gradio-app/gradio/main/gradio/components/chat_interface/processing_done.png") # (user, bot)
            )
            user_input_textbox = gr.Textbox(
                label="你的消息", 
                placeholder="请输入您的问题，或上传图片后在此提问...", 
                lines=3
            )
            with gr.Row():
                image_input_component = gr.Image(type="pil", label="上传图片 (可选，用于OCR)", sources=['upload', 'clipboard'])
            with gr.Row():
                send_button = gr.Button("发送 / 处理", variant="primary")
                clear_button = gr.Button("清除聊天记录")
        
        # 考虑在这里添加一个区域显示工具的使用情况或Agent的思考过程，如果需要的话
        # 例如，一个 gr.Textbox(label="Agent 思考过程", lines=10, interactive=False)
        # 但这需要从 agent_runnable_gradio 的 debug 输出中捕获信息，比较复杂

    # 绑定交互逻辑
    def handle_submit(user_msg, chat_history, img_upload, sess_id):
        # 如果用户只打了回车但没有实际内容（且没有图片），则不处理或返回提示
        if not user_msg.strip() and img_upload is None:
            # gr.Warning("请输入消息或上传图片！") # Gradio 警告似乎不直接适用此场景
            # 直接返回，不改变聊天记录，或者可以添加一个提示消息
            # chat_history.append((user_msg, "请输入您的问题或上传一张图片。"))
            # return chat_history, sess_id, "", None # 清空输入框和图片框
            # 为避免界面行为复杂，让 agent_chat_interface 处理空输入
            pass 
        
        # 调用核心处理函数
        updated_history, updated_sess_id = agent_chat_interface(user_msg, chat_history, img_upload, sess_id)
        return updated_history, updated_sess_id, "", None # 返回更新后的历史，session_id，并清空输入文本框和图片框

    # 交互触发器
    # 1. 用户在文本框中按Enter键
    user_input_textbox.submit(
        handle_submit,
        [user_input_textbox, chatbot, image_input_component, session_id_state],
        [chatbot, session_id_state, user_input_textbox, image_input_component]
    )
    # 2. 用户点击发送按钮
    send_button.click(
        handle_submit,
        [user_input_textbox, chatbot, image_input_component, session_id_state],
        [chatbot, session_id_state, user_input_textbox, image_input_component]
    )

    # 清除按钮逻辑
    def clear_chat_and_session(current_sess_id):
        # 生成一个新的session_id，相当于重置了agent的记忆
        new_sess_id = str(uuid.uuid4())
        # print(f"[Gradio DEBUG] Chat cleared. Old session: {current_sess_id}, New session: {new_sess_id}")
        return [], new_sess_id, "", None # 清空聊天记录, 更新session_id, 清空输入文本框和图片框

    clear_button.click(
        clear_chat_and_session, 
        [session_id_state], 
        [chatbot, session_id_state, user_input_textbox, image_input_component]
    )

# --- 启动 Gradio 应用 ---
if __name__ == "__main__":
    print("正在启动 Gradio 应用...")
    # 注意：如果你的main.py中的日志设置重定向了stdout，这里的print可能不会显示在终端
    # Gradio应用通常在终端输出自己的服务器信息
    demo.launch(server_name="0.0.0.0") # server_name="0.0.0.0" 允许局域网访问 