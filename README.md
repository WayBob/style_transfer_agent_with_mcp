[简体中文](./documentation/README_zh.md) | English

# LangGraph ReAct Agent with Style Transfer

A powerful AI agent built with LangGraph and OpenAI's GPT-4o, featuring ReAct (Reasoning and Acting) capabilities, multiple tools, and artistic style transfer functionality.

## 🌟 Features

### Core Capabilities
- **Conversational AI** with dialogue memory (in-session)
- **ReAct Agent** implementation using LangGraph
- **Multiple Interfaces**: Command-line and Gradio web UI
- **Comprehensive Logging** to `agent_interaction.log`

### Available Tools
1. **Image OCR** - Extract text from images using Tesseract
2. **Current Time** - Get current Beijing time
3. **Web Search** - Search the web for real-time information
4. **Calculator** - Perform mathematical calculations
5. **Style Transfer** - Apply artistic styles to images using StyTR-2

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

5. **Download Style Transfer Models** (if using style transfer):
   - Download the decoder model from [Google Drive](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing)
   - Place it in `StyTR-2/experiments/decoder_iter_160000.pth`

## 💻 Usage

### 🎯 Recommended: Full Agent with Style Transfer

For a complete, ready-to-use agent with all features including style transfer:

```bash
python basic_agent_with_style_transfer.py
```

**Example conversations:**
```
You: Please apply the artistic style from StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png to StyTR-2/demo/image_c/2_10_0_0_512_512.png

Agent: [Automatically processes and creates stylized image]

You: What time is it now?

Agent: [Returns current Beijing time]

You: Search for the latest AI news

Agent: [Searches and returns results]
```

### Other Usage Options

#### 1. Basic Command-Line Interface (without style transfer)
```bash
python main.py
```
Interactive chat with basic tools (OCR, time, search, calculator).

#### 2. Gradio Web Interface
```bash
python gradio_app.py
```
Access the web UI at `http://localhost:7860`

#### 3. Test Style Transfer Functionality
```bash
python test_style_transfer_tool.py
```
Verify that style transfer is working correctly.

## 🎨 Style Transfer Integration Guide

### Which File to Use?

| File | Purpose | When to Use |
|------|---------|-------------|
| `basic_agent_with_style_transfer.py` | **Complete working agent** | Want to use immediately |
| `style_transfer_tool.py` | Langchain tool module | Integrate into your own code |
| `style_transfer_mcp_server.py` | MCP server | Need cross-application usage |
| `test_style_transfer_tool.py` | Test script | Verify functionality |

### Integration Options

#### Option 1: Use the Complete Agent (Recommended)
```bash
python basic_agent_with_style_transfer.py
```

#### Option 2: Integrate into Your Own Agent
```python
from style_transfer_tool import style_transfer

# Add to your tools list
tools = [
    # ... other tools
    style_transfer
]

# Use in your agent
result = style_transfer.invoke({
    "content_image_path": "path/to/content.jpg",
    "style_image_path": "path/to/style.jpg",
    "alpha": 0.8  # Style strength (0.0-1.0)
})
```

#### Option 3: Use as MCP Server
```bash
# Start the server
python style_transfer_mcp_server.py

# Configure your MCP client to connect to the server
```

## 🏗️ Project Structure

```
.
├── core_agent.py                    # Core agent logic and tool definitions
├── main.py                          # Basic command-line interface
├── gradio_app.py                    # Gradio web UI
├── basic_agent_with_style_transfer.py # ⭐ Complete agent with style transfer
├── style_transfer_tool.py           # Langchain tool for style transfer
├── style_transfer_mcp_server.py     # MCP server for style transfer
├── test_style_transfer_tool.py      # Style transfer test script
├── setup.sh                         # Quick setup script
├── StyTR-2/                         # Style transfer model files
├── documentation/                   # Additional documentation
│   ├── README_zh.md                 # Chinese documentation
│   ├── style_transfer_guide.md      # Style transfer guide
│   └── STYLE_TRANSFER_INTEGRATION_SUMMARY.md
└── .env                             # API keys (create this file)
```

## 🔧 Advanced Configuration

### Custom Tools
Add custom tools by modifying `core_agent.py`:

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(input: str) -> str:
    """Description of your tool"""
    return "Tool output"
```

### Model Configuration
The agent uses GPT-4o by default. Modify in `core_agent.py`:

```python
def get_core_llm():
    return ChatOpenAI(
        model="gpt-4o",  # Change model here
        temperature=0.0,
        streaming=True
    )
```

### Style Transfer Parameters
- `alpha`: Style strength (0.0-1.0)
  - 0.0 = Original content
  - 1.0 = Maximum style transfer
  - 0.8 = Recommended balance

## 📚 Documentation

- [中文文档](./documentation/README_zh.md)
- [Style Transfer Guide](./documentation/style_transfer_guide.md)
- [Integration Summary](./documentation/STYLE_TRANSFER_INTEGRATION_SUMMARY.md)

## 🐛 Troubleshooting

### Common Issues

1. **ImportError with torch._six**
   - Already fixed in our implementation
   - Uses compatibility layer for different PyTorch versions

2. **Missing decoder model**
   - Download from the Google Drive link above
   - Ensure it's placed in the correct directory

3. **GPU not available**
   - The tool will automatically fall back to CPU
   - Processing will be slower but still functional

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-4o
- LangChain and LangGraph teams
- StyTR-2 authors for the style transfer model
