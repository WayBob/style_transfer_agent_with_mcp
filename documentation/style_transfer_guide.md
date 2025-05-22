# 风格转换工具集成指南

本文档说明如何将 StyTR-2 风格转换功能集成到您的 LLM Agent 中。

## 方案一：Langchain 工具（推荐用于快速集成）

### 安装依赖

```bash
# 确保已安装所需的包
pip install langchain langchain-openai
```

### 使用方法

1. **导入工具**：
```python
from style_transfer_tool import style_transfer
```

2. **添加到工具列表**：
```python
tools = [
    # ... 其他工具
    style_transfer
]
```

3. **运行示例**：
```bash
python basic_agent_with_style_transfer.py
```

### 工具参数

- `content_image_path`: 内容图片路径（必需）
- `style_image_path`: 风格图片路径（必需）
- `output_path`: 输出路径（可选，默认自动生成）
- `alpha`: 风格强度 0.0-1.0（可选，默认 1.0）

### 使用示例

```python
# 在您的 agent 代码中
result = style_transfer(
    content_image_path="path/to/content.jpg",
    style_image_path="path/to/style.jpg",
    alpha=0.8
)
```

## 方案二：MCP Server（推荐用于标准化集成）

### 安装依赖

```bash
# 安装 FastMCP
pip install fastmcp
```

### 启动 MCP Server

```bash
python style_transfer_mcp_server.py
```

### MCP 配置

在您的 MCP 客户端配置文件中添加：

```json
{
  "mcpServers": {
    "style-transfer": {
      "command": "python",
      "args": ["path/to/style_transfer_mcp_server.py"],
      "env": {}
    }
  }
}
```

### 可用工具

1. **apply_style_transfer**
   - 执行风格转换
   - 参数：content_image_path, style_image_path, output_path, alpha, return_base64

2. **list_available_styles**
   - 列出可用的风格图片

3. **list_content_images**
   - 列出可用的内容图片

### 可用资源

- `style-transfer://model-info`: 获取模型信息

## 性能优化建议

1. **GPU 加速**：如果有 CUDA GPU，模型会自动使用 GPU 加速
2. **批处理**：可以修改代码支持批量处理多张图片
3. **缓存**：模型只在首次调用时加载，后续调用会重用

## 错误处理

两种方案都包含了错误处理：
- 文件不存在时会返回错误信息
- 模型加载失败时会记录日志
- 处理失败时会返回详细错误信息

## 扩展功能

可以轻松添加以下功能：
1. 支持更多图片格式
2. 调整输出图片尺寸
3. 批量处理
4. 风格混合（多个风格图片）
5. 区域风格转换

## 注意事项

1. 确保已下载所有必需的模型文件到 `StyTR-2/experiments/` 目录
2. 输入图片会被调整为 512x512 尺寸
3. 建议使用高质量的风格图片以获得更好的效果
4. alpha 参数控制风格强度，通常 0.6-1.0 效果较好 