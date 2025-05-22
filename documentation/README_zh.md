简体中文 | [English](../README.md)

# LangGraph ReAct 智能体与艺术风格转换

基于 LangGraph 和 OpenAI GPT-4o 构建的强大 AI 智能体，具备 ReAct（推理与行动）能力、多种工具支持，以及由 StyTR-2 提供的艺术风格转换功能。

## 🌟 功能特性

### 核心能力
- **对话式 AI**，带有对话记忆（会话内）
- **ReAct 智能体**实现，使用 LangGraph
- **多种界面**：命令行和 Gradio 网页界面
- **全面的日志记录**到 `agent_interaction.log`

### 可用工具
1. **风格转换** 🎨 - 使用 StyTR-2 将艺术风格应用到图像
2. **图像 OCR** - 使用 Tesseract 从图像中提取文本
3. **获取当前时间** - 获取北京时间
4. **网络搜索** - 搜索网络获取实时信息
5. **计算器** - 进行数学计算
6. **文件列表** - 列出项目文件（图片、Python脚本、Markdown）

## 🚀 快速开始

### 前置要求
- Python 3.8+
- [uv](https://github.com/astral-sh/uv)（Python 包管理工具）
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- OpenAI API 密钥
- Tesseract OCR 引擎（[安装指南](https://tesseract-ocr.github.io/tessdoc/Installation.html)）
- CUDA GPU（可选，用于加速风格转换）

### 快速安装

使用我们的安装脚本自动安装：
```bash
chmod +x setup.sh
./setup.sh
```

或按照以下手动步骤：

1. **克隆并进入**项目目录

2. **设置虚拟环境：**
   ```bash
   uv venv
   source .venv/bin/activate  # Windows 使用: .venv\Scripts\activate
   ```

3. **安装依赖：**
   ```bash
   uv sync
   ```

4. **配置 API 密钥：**
   创建 `.env` 文件：
   ```env
   OPENAI_API_KEY="your_openai_api_key_here"
   ```

5. **下载风格转换模型**：
   - 从 [Google Drive](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing) 下载 decoder 模型
   - 放置到 `StyTR-2/experiments/decoder_iter_160000.pth`
   - 其他所需的模型文件应该已经在仓库中

## 💻 使用方法

### 🎨 风格转换智能体（推荐）

包含所有工具（包括艺术风格转换）的最全功能版本：

```bash
python basic_agent_with_style_transfer.py
```

**对话示例：**

```
你: 请将 StyTR-2/demo/image_c/2_10_0_0_512_512.png 转换成 StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png 的艺术风格

智能体: [处理图像并创建风格化输出]
输出: 风格转换完成！已保存到: stylized_2_10_0_0_512_512_with_LevelSequence_Vaihingen.0000.jpg

你: 现在几点了？

智能体: 当前时间是：2024年05月23日 15:30:45

你: 搜索一下最新的AI艺术生成技术发展

智能体: [搜索并返回关于AI艺术工具的最新信息]

你: 计算 1234 * 5678

智能体: 1234 * 5678 = 7,006,652
```

### 命令行界面（仅基础工具）

不包含风格转换的简化版本：

```bash
python main.py
```

这个版本包含 OCR、时间、搜索、计算器和文件列表工具，但不包含风格转换。

### Gradio 网页界面

基于网页的交互界面：

```bash
python gradio_app.py
```

在浏览器中访问 `http://localhost:7860`

**注意**：Gradio 界面默认不包含风格转换功能，主要专注于 OCR 和其他基础工具。

## 🎨 风格转换详细说明

### 工作原理

风格转换功能使用 StyTR-2（风格变换器2）模型，它：
- 接收两张图片作为输入：内容图片和风格图片
- 将风格图片的艺术风格应用到内容图片上
- 保留结构和内容的同时转换艺术外观

### 使用示例

#### 基础风格转换
```
请将 StyTR-2/demo/image_c/2_10_0_0_512_512.png 转换成 StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png 的艺术风格
```

#### 自定义风格强度
风格转换工具支持 `alpha` 参数（0.0-1.0）来控制风格强度：
- `alpha=1.0`（默认）：完全风格转换
- `alpha=0.8`：80% 风格，20% 原始内容
- `alpha=0.5`：风格和内容的平衡混合

### 直接使用工具（不经过智能体）

您也可以编程方式使用风格转换工具：

```python
from style_transfer_tool import style_transfer

# 基础用法
result = style_transfer.invoke({
    "content_image_path": "path/to/content.jpg",
    "style_image_path": "path/to/style.jpg"
})

# 使用自定义 alpha
result = style_transfer.invoke({
    "content_image_path": "path/to/content.jpg",
    "style_image_path": "path/to/style.jpg",
    "alpha": 0.8,  # 80% 风格强度
    "output_path": "my_stylized_image.jpg"  # 可选的自定义输出路径
})
```

### 测试风格转换工具

验证风格转换功能是否正常工作：

```bash
python test_style_transfer_tool.py
```

这将运行全面的测试，包括：
- 模型文件验证
- 直接工具测试
- Langchain 集成测试

## 🏗️ 项目结构

```
.
├── core_agent.py                    # 核心智能体逻辑和工具定义
├── main.py                          # 基础命令行界面
├── gradio_app.py                    # Gradio 网页界面
├── basic_agent_with_style_transfer.py # 带风格转换的完整智能体
├── style_transfer_tool.py           # 风格转换 Langchain 工具
├── style_transfer_mcp_server.py     # MCP 风格转换服务器
├── test_style_transfer_tool.py      # 风格转换测试脚本
├── setup.sh                         # 快速安装脚本
├── StyTR-2/                         # 风格转换模型文件
│   ├── demo/                        # 演示图片
│   │   ├── image_c/                 # 内容图片
│   │   └── image_s/                 # 风格图片
│   └── experiments/                 # 模型权重
├── documentation/                   # 附加文档
│   ├── README_zh.md                 # 中文文档
│   ├── style_transfer_guide.md      # 风格转换指南
│   └── STYLE_TRANSFER_INTEGRATION_SUMMARY.md
└── .env                             # API 密钥（需创建此文件）
```

## 🔧 技术细节

### 模型架构

风格转换功能使用 StyTR-2，采用了：
- **视觉变换器（ViT）**用于特征提取
- **风格注意力网络**用于风格转换
- **多尺度特征匹配**以获得更好的质量

### 智能体配置

本项目使用 LangChain 的 `create_structured_chat_agent` 来实现智能体功能，这需要特定的配置：

- **提示词格式**：智能体需要在提示模板中包含特定的 JSON 格式指令才能正常运行。
- **JSON 输出结构**：LLM 必须以正确格式的 JSON 块响应，包含：
  ```json
  {
    "action": "工具名称或Final_Answer",
    "action_input": "参数或最终回答"
  }
  ```
- **人类消息模板**：必须同时包含 `{input}` 和 `{agent_scratchpad}` 变量。
- **系统提示词**：必须指定预期的 JSON 格式并提供清晰的工具使用指导。

系统提示词格式示例：
```
你是一个有帮助的AI助手。你可以使用以下工具来帮助回答问题：

{tools}

使用 json blob 来指定一个工具，通过提供 action 键（工具名称）和 action_input 键（工具输入）。

有效的 "action" 值：{tool_names} 或 "Final Answer"

每个 JSON_BLOB 只提供一个动作，格式如下：

{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}

（其他格式说明...）
```

### 性能考虑

- **推荐使用 GPU**：在支持 CUDA 的 GPU 上风格转换速度更快
- **CPU 后备**：如果没有可用的 GPU，工具会自动切换到 CPU
- **处理时间**：
  - GPU：每张图片 2-5 秒
  - CPU：每张图片 20-60 秒
- **内存使用**：处理过程中大约需要 2-4 GB

### 支持的图片格式

- 输入：PNG, JPG, JPEG, BMP, GIF
- 输出：JPG（默认），可自定义
- 推荐尺寸：512x512 像素（如果不同会自动调整大小）

## 🐛 故障排除

### 常见问题

1. **使用风格转换时出现"Field required"错误**
   - 确保同时提供了 content_image_path 和 style_image_path
   - 智能体应该会自动正确解析您的请求

2. **找不到模型文件**
   - 确保已从 Google Drive 链接下载了 decoder 模型
   - 将其放置在 `StyTR-2/experiments/decoder_iter_160000.pth`
   - 运行 `test_style_transfer_tool.py` 验证所有文件是否存在

3. **"Variable agent_scratchpad should be a list of base messages" 错误**
   - 这表明提示词中 `agent_scratchpad` 的预期格式不匹配
   - 确保您的人类消息模板将 `{agent_scratchpad}` 作为字符串变量包含
   - 检查提示词结构是否符合 LangChain 对 `create_structured_chat_agent` 的要求

4. **"Could not parse LLM output" 或 "Invalid or incomplete response" 错误**
   - LLM 没有以预期的 JSON 格式响应
   - 更新系统提示词，包含明确的 JSON 格式说明
   - 确保提示词包含精确的预期响应格式示例
   - 在人类提示词的末尾添加以 JSON 格式响应的提醒

5. **内存不足错误**
   - 尝试使用更小的图片
   - 关闭其他应用程序以释放 RAM/VRAM
   - 如果 GPU 内存有限，使用 CPU 模式

6. **风格转换输出看起来不对**
   - 检查输入图片是否有效且未损坏
   - 尝试调整 alpha 参数以获得不同的风格强度
   - 确保两张图片都是支持的格式

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

## 📄 许可证

本项目是开源的，采用 MIT 许可证。

## 🙏 致谢

- OpenAI 提供的 GPT-4o
- LangChain 和 LangGraph 团队
- StyTR-2 作者提供的风格转换模型
- 开源社区 

### 编程方式使用 MCP 客户端调用风格迁移服务

您也可以通过编写 MCP 客户端程序来直接与 `style_transfer_mcp_server.py` 交互。如果您希望将风格迁移功能直接集成到其他 Python 脚本或工作流中，而不是通过代理界面，这将非常有用。

创建一个 Python 脚本 (例如, `mcp_style_client.py`):

```python
import asyncio
import logging
import os # 确保导入 os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_mcp_style_client():
    # 假设客户端脚本和服务器脚本都在项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    server_script_path = os.path.join(project_root, "style_transfer_mcp_server.py")
    
    server_params = StdioServerParameters(
        command="python", 
        args=[server_script_path],
        cwd=project_root # 设置服务器的工作目录为项目根目录
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("成功连接到风格迁移 MCP 服务器。")

                tools_response = await session.list_tools()
                available_tools = {tool.name: tool for tool in tools_response.tools}
                logger.info(f"可用的工具: {list(available_tools.keys())}")

                tool_to_call = "apply_style_transfer"
                if tool_to_call not in available_tools:
                    logger.error(f"错误：服务器未提供名为 '{tool_to_call}' 的工具。")
                    return

                logger.info(f"尝试调用工具: {tool_to_call}")
                
                # 确保这些路径相对于服务器的工作目录 (cwd) 是正确的
                style_transfer_payload = {
                    "content_image_path": "StyTR-2/demo/image_c/2_10_0_0_512_512.png", 
                    "style_image_path": "StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png", 
                    "alpha": 0.8,
                    "return_base64": False
                    # "output_path": "custom_output_via_client.jpg" # 可选：指定输出路径
                }
                
                arguments_for_call = {"request": style_transfer_payload}
                
                logger.info(f"调用工具 '{tool_to_call}' 的参数: {arguments_for_call}")
                result = await session.call_tool(tool_to_call, arguments=arguments_for_call)
                
                if result.isError:
                    error_message = "未知错误"
                    if result.content and hasattr(result.content[0], 'text'):
                        error_message = result.content[0].text
                    logger.error(f"工具 '{tool_to_call}' 调用失败: {error_message}")
                else:
                    logger.info(f"工具 '{tool_to_call}' 执行成功！")
                    for content_item in result.content:
                        if content_item.type == "text":
                            logger.info(f"  响应: {content_item.text}")
                        # 如果期望其他类型的内容（如资源URI），可以添加更多处理逻辑
    except Exception as e:
        logger.exception(f"运行 MCP 客户端时发生错误: {e}")

if __name__ == "__main__":
    # 确保 StyTR-2 模型 (decoder_iter_160000.pth) 已放置在 StyTR-2/experiments/ 目录下
    # 确保服务器的 Python 环境已安装所有必要依赖。
    # 此示例假设客户端脚本位于项目根目录，并设置服务器CWD为项目根目录。
    # 如果您的文件结构不同，请相应调整 server_script_path 和图片路径。
    asyncio.run(run_mcp_style_client())
```

**运行此客户端：**
1. 将以上代码保存为 `mcp_style_client.py` (或其他名称) 到您的项目根目录下。
2. 确保您的 `style_transfer_mcp_server.py` 也在项目根目录下，或者在脚本中提供正确的相对/绝对路径。
3. 调整 `style_transfer_payload` 中的 `content_image_path` 和 `style_image_path`，确保它们是服务器可以访问到的有效图片路径（通常是相对于服务器工作目录的路径，或绝对路径）。
4. 运行客户端脚本: `python mcp_style_client.py` (如果需要，请确保使用项目虚拟环境中的 Python 解释器)。

此脚本将会连接到您本地的 MCP 服务器，调用 `apply_style_transfer` 工具，并打印返回结果。 