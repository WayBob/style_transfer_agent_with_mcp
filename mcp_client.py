# Assume this is your client code
import asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import logging # Import logging module
import os

# Set log level to see more internal information of MCP client (optional)
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO) # Or INFO level, adjust as needed
logger = logging.getLogger(__name__)


async def run_style_transfer_client():
    server_script_path = "/home/ohya-bob/Documents/mcp/agent/style_transfer_mcp_server.py"
    
    server_params = StdioServerParameters(
        command="python", 
        args=[server_script_path],
        # cwd="/home/ohya-bob/Documents/mcp/agent" # Usually it's best to specify the working directory
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("Successfully connected to Style Transfer MCP server.")

                tools_response = await session.list_tools()
                logger.info("Tools provided by the server:")
                available_tools = {}
                for tool in tools_response.tools:
                    logger.info(f" - Name: {tool.name}, Description: {tool.description}")
                    available_tools[tool.name] = tool
                
                # --- Call apply_style_transfer tool ---
                tool_to_call = "apply_style_transfer"
                if tool_to_call not in available_tools:
                    logger.error(f"Error: Server does not provide a tool named '{tool_to_call}'.")
                    return

                logger.info(f"Will use tool: {tool_to_call}")
                
                content_image_to_use = "/home/ohya-bob/Documents/mcp/agent/StyTR-2/demo/c_img/2_10_0_0_512_512.png"
                style_images_to_use = [
                    "/home/ohya-bob/Documents/mcp/agent/StyTR-2/demo/s_img/LevelSequence_Vaihingen.0011.png",
                    "/home/ohya-bob/Documents/mcp/agent/StyTR-2/demo/s_img/LevelSequence_Vaihingen.0004.png",
                    "/home/ohya-bob/Documents/mcp/agent/StyTR-2/demo/s_img/LevelSequence_Vaihingen.0002.png"
                ]

                for style_image_path in style_images_to_use:
                    logger.info(f"--- Processing style image: {style_image_path} ---")
                    style_transfer_payload = {
                        "content_image_path": content_image_to_use,
                        "style_image_path": style_image_path,
                        "alpha": 0.8,
                        "return_base64": False,
                        # "output_path": f"output/stylized_with_{os.path.basename(style_image_path)}" # Example of custom output path
                    }
                    
                    arguments_for_call = {
                        "request": style_transfer_payload
                    }
                    
                    logger.info(f"Parameters for calling tool '{tool_to_call}': {arguments_for_call}")
                    result = await session.call_tool(tool_to_call, arguments=arguments_for_call)
                    
                    if result.isError:
                        error_message = "Unknown error"
                        if result.content and isinstance(result.content, list) and len(result.content) > 0 and hasattr(result.content[0], 'text'):
                            error_message = result.content[0].text
                        logger.error(f"Error calling tool '{tool_to_call}' for style {style_image_path}: {error_message}")
                    else:
                        logger.info(f"Tool '{tool_to_call}' called successfully for style {style_image_path}!")
                        for content_item in result.content:
                            if content_item.type == "text":
                                logger.info(f"  Result: {content_item.text}")
                            elif content_item.type == "resource" and hasattr(content_item, 'resource'):
                                logger.info(f"  Resource URI: {content_item.resource.uri}")
                                if hasattr(content_item.resource, 'name') and content_item.resource.name:
                                    logger.info(f"    Resource Name: {content_item.resource.name}")
                                if hasattr(content_item.resource, 'text') and content_item.resource.text:
                                    logger.info(f"    Resource Text Preview: {content_item.resource.text[:200]}...")
                    logger.info(f"--- Finished processing style image: {style_image_path} ---\\n")

                # --- (Optional) Call list_available_styles tool ---
                tool_to_call_list_styles = "list_available_styles"
                if tool_to_call_list_styles in available_tools:
                    logger.info(f"\\nWill use tool: {tool_to_call_list_styles}")
                    # list_available_styles usually does not require parameters, or you can pass an empty dictionary
                    list_styles_args = {} 
                    logger.info(f"Parameters for calling tool '{tool_to_call_list_styles}': {list_styles_args}")
                    list_result = await session.call_tool(tool_to_call_list_styles, arguments=list_styles_args)

                    if list_result.isError:
                        error_message = "Unknown error"
                        if list_result.content and isinstance(list_result.content, list) and len(list_result.content) > 0 and hasattr(list_result.content[0], 'text'):
                             error_message = list_result.content[0].text
                        logger.error(f"Error calling tool '{tool_to_call_list_styles}': {error_message}")
                    else:
                        logger.info(f"Tool '{tool_to_call_list_styles}' called successfully!")
                        for content_item in list_result.content:
                            if content_item.type == "text":
                                logger.info(f"  Result: {content_item.text}")
                else:
                    logger.warning(f"Warning: Server does not provide a tool named '{tool_to_call_list_styles}'.")


    except Exception as e:
        logger.exception(f"A critical error occurred while connecting to or calling the MCP server: {e}")

if __name__ == "__main__":
    # Important: Ensure your StyTR-2/experiments/decoder_iter_160000.pth model file has been downloaded and placed correctly
    # Also, ensure Tesseract OCR (if related tools are used indirectly inside the server) is also configured correctly
    asyncio.run(run_style_transfer_client())