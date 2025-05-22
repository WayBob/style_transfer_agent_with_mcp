# 假设这是您的客户端代码
import asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import logging # 导入日志模块

# 设置日志级别，以便看到更多MCP客户端的内部信息（可选）
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO) # 或者 INFO 级别，根据需要调整
logger = logging.getLogger(__name__)


async def run_style_transfer_client():
    server_script_path = "/home/ohya-bob/Documents/mcp/agent/style_transfer_mcp_server.py"
    
    server_params = StdioServerParameters(
        command="python", 
        args=[server_script_path],
        # cwd="/home/ohya-bob/Documents/mcp/agent" # 通常最好指定工作目录
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("成功连接到风格迁移 MCP 服务器。")

                tools_response = await session.list_tools()
                logger.info("服务器提供的工具:")
                available_tools = {}
                for tool in tools_response.tools:
                    logger.info(f" - 名称: {tool.name}, 描述: {tool.description}")
                    available_tools[tool.name] = tool
                
                # --- 调用 apply_style_transfer 工具 ---
                tool_to_call = "apply_style_transfer"
                if tool_to_call not in available_tools:
                    logger.error(f"错误：服务器未提供名为 '{tool_to_call}' 的工具。")
                    return

                logger.info(f"将使用工具: {tool_to_call}")
                
                style_transfer_payload = {
                    "content_image_path": "/home/ohya-bob/Documents/mcp/agent/StyTR-2/demo/image_c/2_10_0_0_512_512.png",
                    "style_image_path": "/home/ohya-bob/Documents/mcp/agent/StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png",
                    "alpha": 0.8,
                    "return_base64": False # 您可以根据需要设置为 True
                    # "output_path": "my_custom_output.jpg" # 如果需要指定输出路径
                }
                
                # MCP call_tool 需要的参数结构，顶层键为 'request'
                arguments_for_call = {
                    "request": style_transfer_payload
                }
                
                logger.info(f"调用工具 '{tool_to_call}' 的参数: {arguments_for_call}")
                result = await session.call_tool(tool_to_call, arguments=arguments_for_call) # 使用包装后的参数
                
                if result.isError:
                    error_message = "未知错误"
                    if result.content and isinstance(result.content, list) and len(result.content) > 0 and hasattr(result.content[0], 'text'):
                        error_message = result.content[0].text
                    logger.error(f"工具 '{tool_to_call}' 调用出错: {error_message}")
                else:
                    logger.info(f"工具 '{tool_to_call}' 调用成功！")
                    for content_item in result.content:
                        if content_item.type == "text":
                            logger.info(f"  结果: {content_item.text}")
                        elif content_item.type == "resource" and hasattr(content_item, 'resource'):
                             logger.info(f"  资源 URI: {content_item.resource.uri}")
                             if hasattr(content_item.resource, 'name') and content_item.resource.name:
                                 logger.info(f"    资源名称: {content_item.resource.name}")
                             if hasattr(content_item.resource, 'text') and content_item.resource.text:
                                 logger.info(f"    资源文本预览: {content_item.resource.text[:200]}...") # 预览部分文本
                             # 如果还想打印图片，需要进一步处理 blob
                        # 您可以根据 apply_style_transfer 工具实际返回的内容类型添加更多处理逻辑

                # --- （可选）调用 list_available_styles 工具 ---
                tool_to_call_list_styles = "list_available_styles"
                if tool_to_call_list_styles in available_tools:
                    logger.info(f"\\n将使用工具: {tool_to_call_list_styles}")
                    # list_available_styles 通常不需要参数，或者您可以传递一个空字典
                    list_styles_args = {} 
                    logger.info(f"调用工具 '{tool_to_call_list_styles}' 的参数: {list_styles_args}")
                    list_result = await session.call_tool(tool_to_call_list_styles, arguments=list_styles_args)

                    if list_result.isError:
                        error_message = "未知错误"
                        if list_result.content and isinstance(list_result.content, list) and len(list_result.content) > 0 and hasattr(list_result.content[0], 'text'):
                             error_message = list_result.content[0].text
                        logger.error(f"工具 '{tool_to_call_list_styles}' 调用出错: {error_message}")
                    else:
                        logger.info(f"工具 '{tool_to_call_list_styles}' 调用成功！")
                        for content_item in list_result.content:
                            if content_item.type == "text":
                                logger.info(f"  结果: {content_item.text}")
                else:
                    logger.warning(f"警告：服务器未提供名为 '{tool_to_call_list_styles}' 的工具。")


    except Exception as e:
        logger.exception(f"连接或调用 MCP 服务器时发生严重错误: {e}")

if __name__ == "__main__":
    # 重要: 确保您的 StyTR-2/experiments/decoder_iter_160000.pth 模型文件已下载并放置正确
    # 同时，确保 Tesseract OCR (如果服务器内部间接用到了相关工具) 也已配置好
    asyncio.run(run_style_transfer_client())