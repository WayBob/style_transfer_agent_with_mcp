简体中文 | [English](../README.md)

# LangGraph ReAct Agent 与风格转换

一个基于 LangGraph 和 OpenAI GPT-4o 构建的强大 AI 智能体，具备 ReAct（推理与行动）能力、多种工具和艺术风格转换功能。

## 🌟 功能特性

### 核心能力
- **对话式 AI**，带有对话记忆（会话内）
- **ReAct 智能体**实现，使用 LangGraph
- **多种界面**：命令行和 Gradio 网页界面
- **全面的日志记录**到 `agent_interaction.log`

### 可用工具
1. **图像 OCR** - 使用 Tesseract 从图像中提取文本
2. **获取当前时间** - 获取北京时间
3. **网络搜索** - 搜索网络获取实时信息
4. **计算器** - 进行数学计算
5. **风格转换** - 使用 StyTR-2 将艺术风格应用到图像

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

5. **下载风格转换模型**（如果使用风格转换功能）：
   - 从 [Google Drive](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing) 下载 decoder 模型
   - 放置到 `StyTR-2/experiments/decoder_iter_160000.pth`

## 💻 使用方法

### 🎯 推荐：带风格转换的完整智能体

使用包含所有功能（包括风格转换）的完整智能体：

```bash
python basic_agent_with_style_transfer.py
```

**对话示例：**
```
你: 请将 StyTR-2/demo/image_c/2_10_0_0_512_512.png 转换成 StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png 的艺术风格

智能体: [自动处理并创建风格化图片]

你: 现在几点了？

智能体: [返回当前北京时间]

你: 搜索一下最新的AI技术发展

智能体: [搜索并返回结果]
```

### 其他使用选项

#### 1. 基础命令行界面（不含风格转换）
```bash
python main.py
```
与基础工具（OCR、时间、搜索、计算器）进行交互聊天。

#### 2. Gradio 网页界面
```bash
python gradio_app.py
```
在浏览器中访问 `http://localhost:7860`

#### 3. 测试风格转换功能
```bash
python test_style_transfer_tool.py
```
验证风格转换是否正常工作。

## 🎨 风格转换集成指南

### 应该使用哪个文件？

| 文件 | 用途 | 何时使用 |
|------|------|----------|
| `basic_agent_with_style_transfer.py` | **完整的可运行智能体** | 想要立即使用 |
| `style_transfer_tool.py` | Langchain 工具模块 | 集成到自己的代码 |
| `style_transfer_mcp_server.py` | MCP 服务器 | 需要跨应用使用 |
| `test_style_transfer_tool.py` | 测试脚本 | 验证功能是否正常 |

### 集成选项

#### 选项 1：使用完整智能体（推荐）
```bash
python basic_agent_with_style_transfer.py
```

#### 选项 2：集成到您自己的智能体
```python
from style_transfer_tool import style_transfer

# 添加到您的工具列表
tools = [
    # ... 其他工具
    style_transfer
]

# 在您的智能体中使用
result = style_transfer.invoke({
    "content_image_path": "path/to/content.jpg",
    "style_image_path": "path/to/style.jpg",
    "alpha": 0.8  # 风格强度 (0.0-1.0)
})
```

#### 选项 3：作为 MCP 服务器使用
```bash
# 启动服务器
python style_transfer_mcp_server.py

# 配置您的 MCP 客户端连接到服务器
```

## 🏗️ 项目结构

```
.
├── core_agent.py                    # 核心智能体逻辑和工具定义
├── main.py                          # 基础命令行界面
├── gradio_app.py                    # Gradio 网页界面
├── basic_agent_with_style_transfer.py # ⭐ 带风格转换的完整智能体
├── style_transfer_tool.py           # Langchain 风格转换工具
├── style_transfer_mcp_server.py     # MCP 风格转换服务器
├── test_style_transfer_tool.py      # 风格转换测试脚本
├── setup.sh                         # 快速安装脚本
├── StyTR-2/                         # 风格转换模型文件
├── documentation/                   # 附加文档
│   ├── README_zh.md                 # 中文文档
│   ├── style_transfer_guide.md      # 风格转换指南
│   └── STYLE_TRANSFER_INTEGRATION_SUMMARY.md
└── .env                             # API 密钥（需创建此文件）
```

## 🔧 高级配置

### 自定义工具
通过修改 `core_agent.py` 添加自定义工具：

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(input: str) -> str:
    """工具描述"""
    return "工具输出"
```

### 模型配置
智能体默认使用 GPT-4o。在 `core_agent.py` 中修改：

```python
def get_core_llm():
    return ChatOpenAI(
        model="gpt-4o",  # 在此更改模型
        temperature=0.0,
        streaming=True
    )
```

### 风格转换参数
- `alpha`: 风格强度 (0.0-1.0)
  - 0.0 = 原始内容
  - 1.0 = 最大风格转换
  - 0.8 = 推荐的平衡值

## 📚 文档

- [English Documentation](../README.md)
- [风格转换指南](./style_transfer_guide.md)
- [集成总结](./STYLE_TRANSFER_INTEGRATION_SUMMARY.md)

## 🐛 故障排除

### 常见问题

1. **torch._six 导入错误**
   - 我们的实现中已经修复
   - 使用兼容层支持不同的 PyTorch 版本

2. **缺少 decoder 模型**
   - 从上面的 Google Drive 链接下载
   - 确保放置在正确的目录中

3. **GPU 不可用**
   - 工具会自动回退到 CPU
   - 处理速度会慢一些但仍然可以工作

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

## 📄 许可证

本项目是开源的，采用 MIT 许可证。

## 🙏 致谢

- OpenAI 提供的 GPT-4o
- LangChain 和 LangGraph 团队
- StyTR-2 作者提供的风格转换模型 