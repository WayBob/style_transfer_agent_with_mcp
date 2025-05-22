"""
Style Transfer MCP Server
This MCP server provides style transfer functionality using StyTR-2
"""

import os
import sys
import logging
from typing import Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from torchvision import transforms
from torchvision.utils import save_image

# Add StyTR-2 to path
STYTR2_PATH = os.path.join(os.path.dirname(__file__), 'StyTR-2')
sys.path.insert(0, STYTR2_PATH)

# Import StyTR-2 modules
import models.StyTR as StyTR
import models.transformer as transformer
from collections import OrderedDict
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("Style Transfer Server")

# Define test_transform function (from StyTR-2 test.py)
def test_transform(size, crop=False):
    transform_list = []
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

class StyleTransferRequest(BaseModel):
    """Request model for style transfer"""
    content_image_path: str = Field(description="Path to the content image")
    style_image_path: str = Field(description="Path to the style image")
    output_path: Optional[str] = Field(default=None, description="Path for output image")
    alpha: float = Field(default=1.0, description="Style weight (0-1)")
    return_base64: bool = Field(default=False, description="Return result as base64 encoded image")

class StyleTransferResponse(BaseModel):
    """Response model for style transfer"""
    output_path: Optional[str] = Field(description="Path to output image if saved")
    base64_image: Optional[str] = Field(description="Base64 encoded image if requested")
    message: str = Field(description="Status message")

class StyleTransferModel:
    """Singleton class to manage the style transfer model"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model paths
        model_dir = os.path.join(STYTR2_PATH, 'experiments')
        self.vgg_path = os.path.join(model_dir, 'vgg_normalised.pth')
        self.decoder_path = os.path.join(model_dir, 'decoder_iter_160000.pth')
        self.trans_path = os.path.join(model_dir, 'transformer_iter_160000.pth')
        self.embed_path = os.path.join(model_dir, 'embedding_iter_160000.pth')
        
        self._load_models()
        self._initialized = True
        
    def _load_models(self):
        """Load all required models"""
        # Create args namespace with required attributes
        class Args:
            position_embedding = 'sine'
            hidden_dim = 512
        args = Args()
        
        # Load VGG
        self.vgg = StyTR.vgg
        self.vgg.load_state_dict(torch.load(self.vgg_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:44])
        
        # Load decoder
        self.decoder = StyTR.decoder
        new_state_dict = OrderedDict()
        state_dict = torch.load(self.decoder_path)
        for k, v in state_dict.items():
            new_state_dict[k] = v
        self.decoder.load_state_dict(new_state_dict)
        
        # Load transformer
        self.Trans = transformer.Transformer()
        new_state_dict = OrderedDict()
        state_dict = torch.load(self.trans_path)
        for k, v in state_dict.items():
            new_state_dict[k] = v
        self.Trans.load_state_dict(new_state_dict)
        
        # Load embeddings
        self.embedding = StyTR.PatchEmbed()
        new_state_dict = OrderedDict()
        state_dict = torch.load(self.embed_path)
        for k, v in state_dict.items():
            new_state_dict[k] = v
        self.embedding.load_state_dict(new_state_dict)
        
        # Create the network
        self.network = StyTR.StyTrans(self.vgg, self.decoder, self.embedding, self.Trans, args)
        self.network.eval()
        self.network.to(self.device)
        
    def transfer_style(self, content_path: str, style_path: str, output_path: str, alpha: float = 1.0, return_base64: bool = False):
        """Perform style transfer"""
        try:
            # Load and preprocess images
            content_tf = test_transform(size=512, crop=False)
            style_tf = test_transform(size=512, crop=False)
            
            content = content_tf(Image.open(content_path).convert('RGB'))
            style = style_tf(Image.open(style_path).convert('RGB'))
            
            style = style.to(self.device).unsqueeze(0)
            content = content.to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                # Use the network directly
                output = self.network(content, style)
            
            # Save output
            output = output.cpu()
            
            # Apply alpha blending if needed
            if alpha < 1.0:
                output = output * alpha + content.cpu() * (1.0 - alpha)
            
            base64_str = None
            if return_base64:
                # Save to buffer first using save_image to ensure proper formatting
                from io import BytesIO
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    save_image(output, tmp.name)
                    tmp_path = tmp.name
                
                # Read back and convert to base64
                with open(tmp_path, 'rb') as f:
                    base64_str = base64.b64encode(f.read()).decode()
                os.unlink(tmp_path)
            
            if output_path:
                # Ensure output directory exists before saving
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                save_image(output, output_path)
            
            return output_path, base64_str
            
        except Exception as e:
            logger.error(f"Style transfer failed: {str(e)}")
            raise

# Initialize model
model = StyleTransferModel()

@mcp.tool()
async def apply_style_transfer(request: StyleTransferRequest) -> StyleTransferResponse:
    """
    Apply artistic style transfer to an image using StyTR-2.
    
    This tool takes a content image and applies the artistic style from a style image.
    The result preserves the content but renders it in the specified artistic style.
    """
    try:
        # Generate output path if not provided
        if request.output_path is None and not request.return_base64:
            content_name = os.path.splitext(os.path.basename(request.content_image_path))[0]
            style_name = os.path.splitext(os.path.basename(request.style_image_path))[0]
            # Ensure output directory exists
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            request.output_path = os.path.join(output_dir, f"stylized_{content_name}_with_{style_name}.jpg")
        
        # Perform style transfer
        output_path, base64_image = model.transfer_style(
            request.content_image_path,
            request.style_image_path,
            request.output_path,
            request.alpha,
            request.return_base64
        )
        
        return StyleTransferResponse(
            output_path=output_path,
            base64_image=base64_image,
            message=f"Style transfer completed successfully!"
        )
        
    except Exception as e:
        return StyleTransferResponse(
            output_path=None,
            base64_image=None,
            message=f"Style transfer failed: {str(e)}"
        )

@mcp.tool()
async def list_available_styles() -> dict:
    """List available style images in the demo directory"""
    style_dir = os.path.join(STYTR2_PATH, 'demo', 'image_s')
    if os.path.exists(style_dir):
        styles = [f for f in os.listdir(style_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return {
            "styles": styles,
            "path": style_dir,
            "message": f"Found {len(styles)} style images"
        }
    return {
        "styles": [],
        "path": style_dir,
        "message": "Style directory not found"
    }

@mcp.tool()
async def list_content_images() -> dict:
    """List available content images in the demo directory"""
    content_dir = os.path.join(STYTR2_PATH, 'demo', 'image_c')
    if os.path.exists(content_dir):
        contents = [f for f in os.listdir(content_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return {
            "contents": contents,
            "path": content_dir,
            "message": f"Found {len(contents)} content images"
        }
    return {
        "contents": [],
        "path": content_dir,
        "message": "Content directory not found"
    }

# Resource for model information
@mcp.resource("style-transfer://model-info")
async def get_model_info() -> str:
    """Get information about the style transfer model"""
    return f"""
# StyTR-2 Style Transfer Model

## Model Information
- Framework: PyTorch
- Device: {model.device}
- Model Components:
  - VGG Feature Extractor
  - Transformer Module
  - Patch Embedding
  - Decoder

## Capabilities
- Artistic style transfer between images
- Adjustable style strength (alpha parameter)
- Support for various image formats

## Usage
Use the `apply_style_transfer` tool with:
- content_image_path: Path to the image you want to transform
- style_image_path: Path to the image whose style you want to apply
- alpha: Style strength (0.0-1.0)
- output_path: Where to save the result (optional)
- return_base64: Return result as base64 string (optional)

## Demo Images
Use `list_available_styles` and `list_content_images` to see available demo images.
"""

if __name__ == "__main__":
    # Run the server
    mcp.run() 