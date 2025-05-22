#!/bin/bash
# Quick setup script for LangGraph ReAct Agent with Style Transfer

echo "🚀 Setting up LangGraph ReAct Agent with Style Transfer..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed successfully"
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating template..."
    echo 'OPENAI_API_KEY="your_openai_api_key_here"' > .env
    echo "📝 Please edit .env file and add your OpenAI API key"
fi

# Check for style transfer model
if [ ! -f "StyTR-2/experiments/decoder_iter_160000.pth" ]; then
    echo "⚠️  Style transfer decoder model not found."
    echo "📥 Please download from: https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing"
    echo "📁 And place it in: StyTR-2/experiments/decoder_iter_160000.pth"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "1. Edit .env file with your OpenAI API key"
echo "2. Download style transfer model if needed"
echo "3. Run: python main.py (for CLI) or python gradio_app.py (for Web UI)"
echo ""
echo "Happy coding! 🎨" 