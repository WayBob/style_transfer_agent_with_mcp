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

# 从核心模块导入 Agent 创建函数和核心工具/提示（如果需要引用或比较）
from core_agent import get_agent_runnable_and_checkpointer, get_core_llm, CORE_SYSTEM_PROMPT, CORE_TOOLS_LIST

# --- 环境设置 (主要由core_agent处理，但Gradio可能需要LLM实例用于非Agent任务，如果存在的话) ---
# llm_gradio = get_core_llm() # 获取核心LLM实例，供Gradio界面特定逻辑使用 (如果需要)
# 如果Gradio的Agent完全依赖core_agent的get_agent_runnable...，则此行可能不需要，
# 因为llm实例会在get_agent_runnable_and_checkpointer内部创建。
# 为了清晰，我们让Agent创建时获取自己的LLM。

# --- Gradio 特定工具定义 ---

# OCR 工具 (PIL Image版 - Gradio专用)
def perform_ocr_gradio(pil_image: Image.Image) -> str:
    """使用 pytesseract 执行 OCR，识别 PIL Image 对象中的中英文字符。"""
    if pil_image is None:
        return "错误：没有提供图像进行OCR。"
    try:
        text = pytesseract.image_to_string(pil_image, lang='chi_sim+eng')
        if not text.strip():
            return "OCR未能识别到任何文字，或者图片为空白。"
        return f"图像OCR识别结果如下：\n{text.strip()}"
    except Exception as e:
        return f"OCR处理失败，错误信息：{str(e)}"

# Gradio Agent 使用的工具列表
# 这些工具的定义应与 core_agent.py 中的工具功能一致，但实例是本地的。
# 或者，可以从 CORE_TOOLS_LIST 中选择性地使用，但要确保它们适用于Gradio环境
# (例如，文件路径OCR不直接适用Gradio的Image组件)

def get_gradio_tools():
    """构建Gradio应用专用的工具列表，从CORE_TOOLS_LIST中选取。"""
    gradio_tools = []
    for core_tool in CORE_TOOLS_LIST:
        # 对于Gradio，我们不直接使用文件路径版的OCR工具（ImageFileOCR）
        # 因为OCR是通过perform_ocr_gradio预处理的。
        # Agent接收的是文本，而不是文件路径让它去OCR。
        if core_tool.name == "ImageFileOCR":
            continue # 跳过文件路径OCR工具
        gradio_tools.append(core_tool) # 其他工具直接使用核心版本
    return gradio_tools

tools_gradio_list = get_gradio_tools()
print(f"[Gradio App] Tools loaded for Gradio Agent: {[tool.name for tool in tools_gradio_list]}")

# --- LangGraph Agent 设置 (Gradio版，使用核心模块) ---
# Gradio 使用核心的系统提示，因为它包含了对预处理OCR文本和ListDirectoryFiles的说明
system_prompt_gradio = CORE_SYSTEM_PROMPT 

# 使用核心模块的函数创建Agent，但传入Gradio特定的工具列表和可选的特定提示
agent_runnable_gradio, checkpointer_gradio = get_agent_runnable_and_checkpointer(
    custom_tools=tools_gradio_list,
    custom_prompt=system_prompt_gradio 
    # debug 默认在 core_agent 中为 True
)

# --- Gradio 交互逻辑 (基本保持不变) ---
def agent_chat_interface(user_message: str, history: list, image_upload: Image.Image, session_id: str):
    final_user_input = user_message
    if image_upload is not None:
        ocr_result_text = perform_ocr_gradio(image_upload)
        if user_message.strip():
            final_user_input = f"{user_message}\n\n[附加图片OCR内容]:\n{ocr_result_text}"
        else:
            final_user_input = f"[图片OCR内容]:\n{ocr_result_text}"
    
    if not final_user_input.strip() and image_upload is None:
        history.append((user_message, "请输入您的问题或上传一张图片。"))
        return history, session_id

    messages_input = [HumanMessage(content=final_user_input)]
    # config中只需thread_id，因为checkpointer已在agent_runnable中配置
    config = {"configurable": {"thread_id": session_id}}
    
    bot_response_content = "抱歉，处理您的请求时出现了一些问题。"
    try:
        # print(f"[Gradio DEBUG] Calling agent with input: {final_user_input[:100]}... Tools: {[t.name for t in tools_gradio_list]}")
        result_state = agent_runnable_gradio.invoke({"messages": messages_input}, config=config)
        if result_state and "messages" in result_state and result_state["messages"]:
            ai_message = result_state["messages"][-1]
            if hasattr(ai_message, 'content') and ai_message.content:
                bot_response_content = ai_message.content
    except Exception as e:
        bot_response_content = f"调用Agent时出错: {str(e)}"
        print(f"[Gradio ERROR] Error invoking agent for session {session_id}: {e}")
        import traceback
        traceback.print_exc()

    history.append((user_message if not image_upload else f"{user_message} (附带图片)", bot_response_content))
    return history, session_id

# --- Gradio UI 构建 (保持不变) ---
with gr.Blocks(theme=gr.themes.Soft(), title="LangGraph ReAct Agent") as demo:
    gr.Markdown("## 🧠 LangGraph ReAct Agent (GPT-4o) - Web UI")
    gr.Markdown(
        "与基于LangGraph和GPT-4o构建的智能助手进行交互。"
        "您可以提问、进行计算、查询时间、搜索网页、列出项目文件，或上传图片进行文字识别。"
    )
    session_id_state = gr.State(lambda: str(uuid.uuid4()))
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="聊天记录",
                bubble_full_width=False,
                avatar_images=(None, "https://raw.githubusercontent.com/gradio-app/gradio/main/gradio/components/chat_interface/processing_done.png")
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

# --- 启动 Gradio 应用 (保持不变) ---
if __name__ == "__main__":
    print("正在启动 Gradio 应用 (Agent核心来自core_agent.py)...")
    demo.launch(server_name="0.0.0.0") 