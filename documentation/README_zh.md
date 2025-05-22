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

3. **内存不足错误**
   - 尝试使用更小的图片
   - 关闭其他应用程序以释放 RAM/VRAM
   - 如果 GPU 内存有限，使用 CPU 模式

4. **风格转换输出看起来不对**
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