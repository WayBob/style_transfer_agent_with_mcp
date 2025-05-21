English | [简体中文](../README.md)

**注意**: 此为中文版README。英文版位于项目根目录的 `README.md`。

# 基于 LangGraph 和 GPT-4o 的 ReAct Agent

本项目实现了一个基于 LangGraph 框架和 OpenAI GPT-4o 模型构建的 ReAct (推理与行动) Agent。它拥有一个命令行交互界面，具备对话记忆能力，并可以利用一套工具来增强交互和完成任务。

##核心特性

*   **对话式AI**: 支持具备会话内有效对话记忆的多轮对话。
*   **工具使用**: Agent 能够自主调用以下工具：
    *   图像OCR (ImageOCR)：识别图片中的文字（支持中英文）。
    *   获取当前时间 (GetCurrentTime)：提供当前的日期和时间。
    *   网页搜索 (WebSearch)：使用 DuckDuckGo 搜索互联网以获取最新信息。
    *   列出目录文件 (ListDirectoryFiles)：列出当前工作目录下的特定类型文件。
    *   计算器 (Calculator)：通过 Python REPL 执行数学表达式。
*   **LangGraph 和 GPT-4o 驱动**: 基于 LangGraph 的 `create_react_agent` 和 GPT-4o 的强大能力构建。
*   **详细日志**: 所有的交互、Agent 内部执行步骤（当 `main.py` 或 `core_agent.py` 中 `debug=True` 时）以及标准输出都会被记录到 `agent_interaction.log` 文件中，并且同时在终端显示。

## 先决条件

*   Python 3.8 或更高版本。
*   [uv](https://github.com/astral-sh/uv) (Python 包管理工具)。在 macOS 和 Linux 上安装命令如下：
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    对于 Windows 和其他安装方法，请参考 [uv 官方安装指南](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)。
*   OpenAI API 密钥。
*   Tesseract OCR 引擎：确保已在你的系统上安装并配置到系统路径 (PATH)。
    *   关于 Tesseract 的安装，请参考 [Tesseract 安装指南](https://tesseract-ocr.github.io/tessdoc/Installation.html)。
    *   为了更好地进行OCR识别，建议安装中文 (`chi_sim`) 和英文 (`eng`) 的语言数据包。

## 快速上手

1.  **克隆项目 (如果适用) 并进入目录**：
    *(如果你是使用 `uv init <project_name>` 创建的项目，可能已经位于项目目录中。)*

2.  **设置虚拟环境 (使用 uv)**：
    如果 `.venv` 虚拟环境尚不存在，请创建它：
    ```bash
    uv venv
    ```
    激活虚拟环境：
    ```bash
    source .venv/bin/activate  # Windows 用户请使用: .venv\Scripts\activate
    ```

3.  **安装依赖 (使用 uv)**：
    确保项目根目录中有一个 `requirements.txt` 文件，内容如下 (或根据项目实际需求调整)：
    ```txt
    python-dotenv
    Pillow
    pytesseract
    langchain
    langchain-openai
    langchain-community
    langgraph
    duckduckgo-search
    gradio
    ```
    从 `requirements.txt` 文件安装依赖：
    ```bash
    uv pip install -r requirements.txt
    ```
    或者，单独添加包 (例如 Gradio，如果尚未在 `requirements.txt` 中)：
    ```bash
    uv add gradio
    # uv add python-dotenv Pillow pytesseract langchain langchain-openai langchain-community langgraph duckduckgo-search
    ```
    *(注意：如果你的项目是用 `uv init` 初始化的并使用 `pyproject.toml`，`uv add` 会更新该文件。如果你主要使用 `requirements.txt`，请确保其保持最新，并在通过 `uv add` 添加包后（如果项目使用 `pyproject.toml` 进行锁定）考虑运行 `uv pip install -r requirements.txt` 或 `uv sync`。)*

4.  **配置 API 密钥**：
    在项目根目录创建一个名为 `.env` 的文件，并填入你的 OpenAI API 密钥：
    ```env
    OPENAI_API_KEY="你的OpenAI_API密钥"
    ```

5.  **运行应用**：
    *   **命令行 Agent**：
        ```bash
        python main.py
        ```
    *   **Gradio Web UI**：
        ```bash
        python gradio_app.py
        ```
    你可以在终端或 Web 浏览器中与 Agent 交互。在命令行界面中，输入 "退出" 或 "exit" 来结束程序。

## 日志记录

所有的交互和详细的 Agent 执行步骤都会被记录到项目根目录下的 `agent_interaction.log` 文件中（每次运行时追加内容），并且相关信息也会实时显示在终端。

## 项目结构

*   `core_agent.py`: 包含 LangGraph Agent 的核心逻辑，包括 LLM 配置、工具定义和 Agent 创建函数。
*   `main.py`: 提供与 Agent 交互的命令行界面。
*   `gradio_app.py`: 实现了一个基于 Gradio 的 Agent Web UI。
*   `.env`: 用于存储 OpenAI API 密钥 (请确保此文件已添加到 `.gitignore` 中)。
*   `requirements.txt`: 列出项目所需的 Python 依赖包。
*   `agent_interaction.log`: 记录 Agent 交互和调试信息的日志文件。
*   `README.md`: 项目的英文版说明文档。
*   `documentation/README_zh.md`: 项目的中文版说明文档 (即本文档)。

## 未来开发方向

本项目为一个基础的 ReAct Agent 实现。未来的增强方向可以包括：
*   更完善的错误处理机制。
*   持久化的对话记忆方案 (例如使用 `SQLChatMessageHistory`)。
*   添加更多的工具和功能。
*   通过命令行参数或UI界面配置 `debug` 模式。 