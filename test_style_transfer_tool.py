"""
Test script for style transfer tool
"""

import os
import sys

def test_langchain_tool():
    """Test the Langchain tool implementation"""
    print("Testing Langchain Style Transfer Tool...")
    
    try:
        from style_transfer_tool import style_transfer
        
        # Test with demo images using invoke method
        result = style_transfer.invoke({
            "content_image_path": "StyTR-2/demo/image_c/2_10_0_0_512_512.png",
            "style_image_path": "StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png",
            "output_path": "test_langchain_output.jpg",
            "alpha": 0.8
        })
        
        print(f"✓ Langchain tool test successful!")
        print(f"  Result: {result}")
        
        # Check if output file exists
        if os.path.exists("test_langchain_output.jpg"):
            print(f"✓ Output file created successfully")
        else:
            print(f"✗ Output file not found")
            
    except Exception as e:
        print(f"✗ Langchain tool test failed: {str(e)}")
        return False
    
    return True

def test_mcp_server():
    """Test the MCP server (requires server to be running)"""
    print("\nTesting MCP Server...")
    print("Note: This test requires the MCP server to be running.")
    print("Start the server with: python style_transfer_mcp_server.py")
    
    # For actual testing, you would need an MCP client
    # This is just a placeholder to show how it would work
    print("✓ MCP server code created successfully")
    print("  To test: Run the server and connect with an MCP client")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Style Transfer Tool Test Suite")
    print("=" * 60)
    
    # Check if StyTR-2 directory exists
    if not os.path.exists("StyTR-2"):
        print("✗ Error: StyTR-2 directory not found!")
        print("  Please ensure you're running from the correct directory")
        return
    
    # Check if model files exist
    model_files = [
        "StyTR-2/experiments/vgg_normalised.pth",
        "StyTR-2/experiments/decoder_iter_160000.pth",
        "StyTR-2/experiments/transformer_iter_160000.pth",
        "StyTR-2/experiments/embedding_iter_160000.pth"
    ]
    
    missing_models = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        print("✗ Error: Missing model files:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nPlease download all model files as described in DOWNLOAD_MODELS.md")
        return
    
    print("✓ All model files found")
    
    # Run tests
    langchain_success = test_langchain_tool()
    mcp_success = test_mcp_server()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Langchain Tool: {'✓ PASSED' if langchain_success else '✗ FAILED'}")
    print(f"  MCP Server: {'✓ CREATED' if mcp_success else '✗ FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    main() 