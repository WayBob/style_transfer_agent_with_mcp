[简体中文](./documentation/README_zh.md) | English

# LangGraph ReAct Agent with Artistic Style Transfer

A powerful AI agent built with LangGraph and OpenAI's GPT-4o, featuring ReAct (Reasoning and Acting) capabilities, multiple tools, and artistic style transfer functionality powered by StyTR-2.

## 🌟 Features

### Core Capabilities
- **Conversational AI** with dialogue memory (in-session)
- **ReAct Agent** implementation using LangGraph
- **Multiple Interfaces**: Command-line and Gradio web UI
- **Comprehensive Logging** to `agent_interaction.log`

### Available Tools
1. **Style Transfer** 🎨 - Apply artistic styles to images using StyTR-2
2. **Image OCR** - Extract text from images using Tesseract
3. **Current Time** - Get current Beijing time
4. **Web Search** - Search the web for real-time information
5. **Calculator** - Perform mathematical calculations
6. **List Files** - List project files (images, Python scripts, Markdown)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (Python packaging tool)
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- OpenAI API Key
- Tesseract OCR engine ([installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html))
- CUDA GPU (optional, for faster style transfer)

### Quick Installation

Use our setup script for automatic installation:
```bash
chmod +x setup.sh
./setup.sh
```

Or follow manual steps:

1. **Clone & Navigate** to the project directory

2. **Set up Virtual Environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   uv sync
   ```

4. **Configure API Key:**
   Create a `.env` file:
   ```env
   OPENAI_API_KEY="your_openai_api_key_here"
   ```

5. **Download Style Transfer Models**:
   - Download the decoder model from [Google Drive](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing)
   - Place it in `StyTR-2/experiments/decoder_iter_160000.pth`
   - The other required model files should already be in the repository

## 💻 Usage

### 🎨 Style Transfer Agent (Recommended)

The most feature-rich option with all tools including artistic style transfer:

```bash
python basic_agent_with_style_transfer.py
```

**Example Conversations:**

```
You: Please apply the artistic style from StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png to StyTR-2/demo/image_c/2_10_0_0_512_512.png

Agent: [Processes the images and creates a stylized output]
Output: Style transfer completed! Output saved to: stylized_2_10_0_0_512_512_with_LevelSequence_Vaihingen.0000.jpg

You: What time is it now?

Agent: The current time is: 2024年05月23日 15:30:45

You: Search for the latest developments in AI art generation

Agent: [Searches and returns current information about AI art tools]

You: Calculate 1234 * 5678

Agent: 1234 * 5678 = 7,006,652
```

### Command-Line Interface (Basic Tools Only)

For a simpler agent without style transfer:

```bash
python main.py
```

This version includes OCR, time, search, calculator, and file listing tools but not style transfer.

### Gradio Web Interface

For a web-based interface:

```bash
python gradio_app.py
```

Access the web UI at `http://localhost:7860`

**Note**: The Gradio interface doesn't include style transfer by default. It focuses on OCR and other basic tools.

## 🎨 Style Transfer Details

### How It Works

The style transfer feature uses the StyTR-2 (Style Transformer 2) model, which:
- Takes two images as input: a content image and a style image
- Applies the artistic style of the style image to the content image
- Preserves the structure and content while transforming the artistic appearance

### Usage Examples

#### Basic Style Transfer
```
Please convert StyTR-2/demo/image_c/2_10_0_0_512_512.png into the artistic style of StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png
```

#### Custom Style Strength
The style transfer tool supports an `alpha` parameter (0.0-1.0) to control style strength:
- `alpha=1.0` (default): Full style transfer
- `alpha=0.8`: 80% style, 20% original content
- `alpha=0.5`: Balanced mix of style and content

### Direct Tool Usage (Without Agent)

You can also use the style transfer tool programmatically:

```python
from style_transfer_tool import style_transfer

# Basic usage
result = style_transfer.invoke({
    "content_image_path": "path/to/content.jpg",
    "style_image_path": "path/to/style.jpg"
})

# With custom alpha
result = style_transfer.invoke({
    "content_image_path": "path/to/content.jpg",
    "style_image_path": "path/to/style.jpg",
    "alpha": 0.8,  # 80% style strength
    "output_path": "my_stylized_image.jpg"  # Optional custom output path
})
```

### Test the Style Transfer Tool

To verify the style transfer functionality is working correctly:

```bash
python test_style_transfer_tool.py
```

This will run a comprehensive test including:
- Model file verification
- Direct tool testing
- Langchain integration testing

### Programmatic MCP Client for Style Transfer Server

You can also interact with the `style_transfer_mcp_server.py` programmatically using an MCP client. This is useful if you want to integrate the style transfer capability into other Python scripts or workflows directly, bypassing the agent interface.

Create a Python script (e.g., `mcp_style_client.py`):

```python
import asyncio
import logging
import os # Ensure os is imported
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_mcp_style_client():
    # Assume client and server scripts are in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    server_script_path = os.path.join(project_root, "style_transfer_mcp_server.py") 
    
    server_params = StdioServerParameters(
        command="python", 
        args=[server_script_path],
        cwd=project_root # Set server's CWD to project root
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("Successfully connected to the Style Transfer MCP Server.")

                tools_response = await session.list_tools()
                available_tools = {tool.name: tool for tool in tools_response.tools}
                logger.info(f"Available tools: {list(available_tools.keys())}")

                tool_to_call = "apply_style_transfer"
                if tool_to_call not in available_tools:
                    logger.error(f"Error: Tool '{tool_to_call}' not found on server.")
                    return

                logger.info(f"Attempting to call tool: {tool_to_call}")
                
                # Ensure these paths are correct relative to the server's CWD (project_root)
                style_transfer_payload = {
                    "content_image_path": "StyTR-2/demo/image_c/2_10_0_0_512_512.png", 
                    "style_image_path": "StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png", 
                    "alpha": 0.8,
                    "return_base64": False 
                    # "output_path": "custom_output_via_client.jpg" # Optional
                }
                
                arguments_for_call = {"request": style_transfer_payload}
                
                logger.info(f"Calling '{tool_to_call}' with args: {arguments_for_call}")
                result = await session.call_tool(tool_to_call, arguments=arguments_for_call)
                
                if result.isError:
                    error_message = "Unknown error"
                    if result.content and hasattr(result.content[0], 'text'):
                        error_message = result.content[0].text
                    logger.error(f"Tool '{tool_to_call}' failed: {error_message}")
                else:
                    logger.info(f"Tool '{tool_to_call}' executed successfully!")
                    for content_item in result.content:
                        if content_item.type == "text":
                            logger.info(f"  Response: {content_item.text}")
                        # Add more handling for other content types if expected (e.g., resource URI)
    except Exception as e:
        logger.exception(f"An error occurred while running the MCP client: {e}")

if __name__ == "__main__":
    # Ensure the StyTR-2 model (decoder_iter_160000.pth) is in StyTR-2/experiments/
    # Ensure your Python environment for the server has all necessary dependencies.
    # This example assumes the client script is in the project root.
    asyncio.run(run_mcp_style_client())

## 🏗️ Project Structure

```
.
├── core_agent.py                    # Core agent logic and tool definitions
├── main.py                          # Basic command-line interface
├── gradio_app.py                    # Gradio web UI
├── basic_agent_with_style_transfer.py # Complete agent with style transfer
├── style_transfer_tool.py           # Style transfer Langchain tool
├── style_transfer_mcp_server.py     # MCP server for style transfer
├── test_style_transfer_tool.py      # Style transfer test script
├── setup.sh                         # Quick setup script
├── StyTR-2/                         # Style transfer model files
│   ├── demo/                        # Demo images
│   │   ├── image_c/                 # Content images
│   │   └── image_s/                 # Style images
│   └── experiments/                 # Model weights
├── documentation/                   # Additional documentation
│   ├── README_zh.md                 # Chinese documentation
│   ├── style_transfer_guide.md      # Style transfer guide
│   └── STYLE_TRANSFER_INTEGRATION_SUMMARY.md
└── .env                             # API keys (create this file)
```

## 🔧 Technical Details

### Model Architecture

The style transfer feature uses StyTR-2, which employs:
- **Vision Transformer (ViT)** for feature extraction
- **Style-Attentional Network** for style transfer
- **Multi-scale feature matching** for better quality

### Agent Configuration

The project uses LangChain's `create_structured_chat_agent` for agent functionality, which requires specific configuration:

- **Prompt Format**: The agent requires specific JSON format instructions in the prompt template to function correctly.
- **JSON Output Structure**: The LLM must respond with a properly formatted JSON blob containing:
  ```json
  {
    "action": "tool_name_or_Final_Answer",
    "action_input": "parameters_or_final_response"
  }
  ```
- **Human Message Template**: Must include both `{input}` and `{agent_scratchpad}` variables.
- **System Prompt**: Must specify the expected JSON format and provide clear guidance on tool usage.

Example system prompt format:
```
You are a helpful AI assistant. You can use the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: {tool_names} or "Final Answer"

Provide only ONE action per JSON_BLOB, as shown:

{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}

(Additional formatting instructions...)
```

### Performance Considerations

- **GPU Recommended**: Style transfer is much faster on CUDA-enabled GPUs
- **CPU Fallback**: The tool automatically falls back to CPU if no GPU is available
- **Processing Time**: 
  - GPU: 2-5 seconds per image
  - CPU: 20-60 seconds per image
- **Memory Usage**: Approximately 2-4 GB during processing

### Supported Image Formats

- Input: PNG, JPG, JPEG, BMP, GIF
- Output: JPG (default), can be customized
- Recommended size: 512x512 pixels (automatically resized if different)

## 🐛 Troubleshooting

### Common Issues

1. **"Field required" error when using style transfer**
   - Make sure you're providing both content_image_path and style_image_path
   - The agent should automatically parse your request correctly

2. **Model file not found**
   - Ensure you've downloaded the decoder model from the Google Drive link
   - Place it in `StyTR-2/experiments/decoder_iter_160000.pth`
   - Run `test_style_transfer_tool.py` to verify all files are present

3. **"Variable agent_scratchpad should be a list of base messages" error**
   - This indicates a mismatch between the expected format of `agent_scratchpad` in the prompt
   - Ensure your human message template includes `{agent_scratchpad}` as a string variable
   - Check that the prompt structure follows LangChain's expectations for `create_structured_chat_agent`

4. **"Could not parse LLM output" or "Invalid or incomplete response" errors**
   - The LLM is not responding with the expected JSON format
   - Update the system prompt to include explicit JSON formatting instructions
   - Ensure the prompt includes examples of the exact expected response format
   - Add a reminder at the end of the human prompt to respond in JSON format

5. **Out of memory error**
   - Try using smaller images
   - Close other applications to free up RAM/VRAM
   - Use CPU mode if GPU memory is limited

6. **Style transfer output looks wrong**
   - Check that input images are valid and not corrupted
   - Try adjusting the alpha parameter for different style strengths
   - Ensure both images are in supported formats

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-4o
- LangChain and LangGraph teams
- StyTR-2 authors for the style transfer model
- The open-source community

# StyTR-2 图像风格迁移工具 (Image Style Transfer Tool using StyTR-2)

## 简介 (Introduction)

本工具包利用 StyTR-2 模型实现强大的图像风格迁移功能。它被封装为一个 Langchain 工具，并提供了一个 MCP (MCP) 服务器接口，方便集成到各种应用中。此外，还包含一个集成了此工具的 Langchain Agent 示例。

This toolkit leverages the StyTR-2 model to perform powerful image style transfer. It is packaged as a Langchain tool and also provides an MCP (Multi-Content Interaction Protocol) server interface, making it easy to integrate into various applications. Additionally, an example Langchain Agent demonstrating the tool's usage is included.

## 功能特性 (Features)

*   **高质量风格迁移 (High-Quality Style Transfer)**：基于 StyTR-2 模型，能够生成艺术效果出色的风格化图像。
*   **Langchain 工具集成 (Langchain Tool Integration)**：提供 `style_transfer` 工具，可轻松嵌入到 Langchain Agent 或链中。
    *   输入内容图片路径 (content image path) 和风格图片路径 (style image path)。
    *   可选输出路径 (optional output path)；如果未提供，则默认保存到项目根目录下的 `output/` 文件夹。
    *   可调风格强度 `alpha` (adjustable style strength `alpha`)。
*   **MCP 服务器 (MCP Server)**：通过 `style_transfer_mcp_server.py` 启动，提供以下 API 端点：
    *   `apply_style_transfer`：执行风格迁移，可选择返回 base64 编码的图像或保存到文件。
    *   `list_available_styles`：列出 `StyTR-2/demo/s_img/` 目录下的可用风格图片。
    *   `list_content_images`：列出 `StyTR-2/demo/c_img/` 目录下的可用内容图片。
    *   `style-transfer://model-info`：获取模型信息资源。
*   **示例 Agent (Example Agent)**：`basic_agent_with_style_transfer.py` 展示了如何在 Langchain Agent 中使用风格迁移工具及其他常用工具。
*   **标准化输出 (Standardized Output)**：所有通过工具生成的图像默认保存在项目根目录下的 `output/` 文件夹中，该文件夹会自动创建。
*   **清晰的项目结构 (Clear Project Structure)**：易于理解和扩展。

## 项目结构 (Project Structure)

```
.
├── StyTR-2/                    # StyTR-2 模型和相关文件 (Git submodule or directory)
│   ├── demo/
│   │   ├── c_img/              # 示例内容图片 (Sample content images)
│   │   └── s_img/              # 示例风格图片 (Sample style images)
│   └── experiments/            # 预训练模型权重 (Pre-trained model weights)
├── output/                     # 生成的风格化图像默认输出目录 (Default output directory for stylized images)
├── .gitignore                  # Git忽略文件配置
├── style_transfer_tool.py      # Langchain 工具定义
├── style_transfer_mcp_server.py # MCP 服务器实现
├── basic_agent_with_style_transfer.py # 集成工具的示例Agent
├── test_style_transfer_tool.py # 工具测试脚本
└── README.md                   # 本文档 (This document)
```

**重要 (Important):**
*   `StyTR-2/` 目录包含了核心模型代码和必要的模型权重文件。请确保此目录完整且模型文件已按要求下载。
*   `output/` 目录用于存放所有由本工具包生成的图像。此目录会在首次生成图像时自动创建，并已被添加到 `.gitignore`。

## 环境设置与依赖 (Setup and Dependencies)

1.  **Python 版本 (Python Version)**：建议使用 Python 3.8 或更高版本。
    (Python 3.8+ is recommended.)

2.  **克隆仓库 (Clone Repository)**：
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

3.  **StyTR-2 模型 (StyTR-2 Model)**：
    *   如果 `StyTR-2` 是一个 Git 子模块 (Git submodule)，请运行：
        ```bash
        git submodule update --init --recursive
        ```
    *   如果不是子模块，请确保您已将 StyTR-2 的代码和预训练模型放置在项目根目录下的 `StyTR-2` 文件夹中。
    *   **下载模型权重 (Download Model Weights)**：根据 StyTR-2 的原始说明，下载 `vgg_normalised.pth`、`decoder_iter_160000.pth`、`transformer_iter_160000.pth` 和 `embedding_iter_160000.pth`，并将它们放置在 `StyTR-2/experiments/` 目录下。

4.  **安装依赖包 (Install Dependencies)**：
    ```bash
    pip install torch torchvision Pillow langchain langchain-openai pydantic pytz tavily-python mcp-fastmcp
    ```
    *   请确保安装与您的 CUDA 版本兼容的 PyTorch 版本（如果希望使用 GPU）。
        (Ensure you install a PyTorch version compatible with your CUDA version if GPU usage is desired.)
    *   `mcp-fastmcp` 用于 MCP 服务器。

5.  **OpenAI API 密钥 (OpenAI API Key)**：
    对于 `basic_agent_with_style_transfer.py` 中的 Agent，需要设置 OpenAI API 密钥：
    (For the Agent in `basic_agent_with_style_transfer.py`, set your OpenAI API key:)
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```

## 使用方法 (Usage)

**请确保从项目根目录运行以下脚本，以保证相对路径的正确解析。**
**(Please ensure you run the following scripts from the project root directory for correct relative path resolution.)**

### 1. Langchain 工具 (`style_transfer_tool.py`)

可以直接在 Python 脚本中导入和使用 `style_transfer` 工具。

```python
from style_transfer_tool import style_transfer

# 示例调用 (Example call)
result = style_transfer.invoke({
    "content_image_path": "StyTR-2/demo/c_img/2_10_0_0_512_512.png",
    "style_image_path": "StyTR-2/demo/s_img/LevelSequence_Vaihingen.0002.png",
    "output_path": "output/my_stylized_image.jpg", # 可选 (Optional)
    "alpha": 0.8 # 可选，默认 1.0 (Optional, default 1.0)
})
print(result)
# Output: Style transfer completed! Output saved to: output/my_stylized_image.jpg
```
如果未提供 `output_path`，图像将保存到 `output/stylized_<content_name>_with_<style_name>.jpg`。

(If `output_path` is not provided, the image will be saved to `output/stylized_<content_name>_with_<style_name>.jpg`.)

### 2. MCP 服务器 (`style_transfer_mcp_server.py`)

启动服务器：
(Start the server:)
```bash
python style_transfer_mcp_server.py
```
服务器将在默认端口 (通常是 8000) 上运行。您可以使用任何 MCP 客户端与服务器交互，调用 `apply_style_transfer`, `list_available_styles`, `list_content_images` 等工具。

### 3. 示例 Agent (`basic_agent_with_style_transfer.py`)

此脚本演示了如何将 `style_transfer` 工具与其他工具（如 OCR、时间获取、网页搜索、计算器）集成到一个 Langchain Agent 中。

运行示例 Agent：
(Run the example Agent:)
```bash
python basic_agent_with_style_transfer.py
```
然后您可以与 Agent 进行交互，例如：
(Then you can interact with the Agent, e.g.:)
```
请输入您的问题: 请将 StyTR-2/demo/c_img/2_10_0_0_512_512.png 转换成 StyTR-2/demo/s_img/LevelSequence_Vaihingen.0002.png 的艺术风格
```
生成的图像将保存在 `output/` 目录下。

## 测试 (Testing)

项目包含一个测试脚本 `test_style_transfer_tool.py`，用于验证 Langchain 工具的基本功能和模型文件的完整性。

运行测试：
(Run tests:)
```bash
python test_style_transfer_tool.py
```
测试脚本会：
1.  检查所有必需的 StyTR-2 模型文件是否存在于 `StyTR-2/experiments/`。
2.  使用 Langchain 工具对 `StyTR-2/demo/` 中的示例图像进行风格迁移。
3.  检查输出图像是否成功创建在 `output/` 目录中。

## 注意事项 (Notes)

*   **执行路径 (Execution Path)**：强烈建议所有脚本都从项目的根目录执行，以确保所有相对路径（特别是针对 `StyTR-2` 目录和 `output` 目录的路径）都能被正确解析。
*   **GPU 支持 (GPU Support)**：默认情况下，代码会尝试使用 CUDA GPU（如果可用）。如果 GPU 不可用或未正确配置，则会回退到 CPU。在 CPU 上进行风格迁移可能会非常慢。
*   **模型文件 (Model Files)**：确保 `StyTR-2/experiments/` 目录下的模型文件 (`.pth`) 未被意外添加到 Git 跟踪中（已通过 `.gitignore` 配置忽略）。
*   **错误处理 (Error Handling)**：脚本中包含基本的错误处理，但可能需要根据具体部署环境进行增强。

## 未来展望 (Future Work) (Optional)

*   [ ] 更细致的参数控制（例如，内容权重、风格权重等）。
        (Finer-grained parameter control, e.g., content weight, style weight.)
*   [ ] 支持通过 URL 输入图像。
        (Support for image input via URLs.)
*   [ ] 构建一个简单的 Web UI 界面。
        (Build a simple Web UI interface.)

---

希望这份 README 对您有所帮助！如果您有任何其他建议或需要修改的地方，请告诉我。
(Hope this README is helpful! Please let me know if you have any other suggestions or need modifications.)
