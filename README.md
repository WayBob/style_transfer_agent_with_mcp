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

3. **Out of memory error**
   - Try using smaller images
   - Close other applications to free up RAM/VRAM
   - Use CPU mode if GPU memory is limited

4. **Style transfer output looks wrong**
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
