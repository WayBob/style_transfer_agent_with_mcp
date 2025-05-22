"""
Style Transfer Tool for Langchain
This tool wraps the StyTR-2 style transfer functionality
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from langchain.tools import tool
from pydantic import BaseModel, Field  # Updated to use pydantic directly
import logging
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
logger = logging.getLogger(__name__)

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

class StyleTransferInput(BaseModel):
    """Input schema for style transfer tool"""
    content_image_path: str = Field(description="Path to the content image")
    style_image_path: str = Field(description="Path to the style image")
    output_path: Optional[str] = Field(default=None, description="Path for output image. If not provided, will generate based on input names")
    alpha: float = Field(default=1.0, description="Style weight (0-1), higher means stronger style")
    
class StyleTransferTool:
    def __init__(self, model_dir: str = None):
        """Initialize the style transfer model"""
        if model_dir is None:
            model_dir = os.path.join(STYTR2_PATH, 'experiments')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.vgg_path = os.path.join(model_dir, 'vgg_normalised.pth')
        self.decoder_path = os.path.join(model_dir, 'decoder_iter_160000.pth')
        self.trans_path = os.path.join(model_dir, 'transformer_iter_160000.pth')
        self.embed_path = os.path.join(model_dir, 'embedding_iter_160000.pth')
        
        self._load_models()
        
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
        
    def transfer_style(self, content_path: str, style_path: str, output_path: str, alpha: float = 1.0) -> str:
        """
        Perform style transfer
        
        Args:
            content_path: Path to content image
            style_path: Path to style image
            output_path: Path for output image
            alpha: Style weight (0-1)
            
        Returns:
            Path to the output image
        """
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
            
            save_image(output, output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Style transfer failed: {str(e)}")
            raise

# Global instance
_tool_instance = None

def get_tool_instance():
    """Get or create tool instance"""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = StyleTransferTool()
    return _tool_instance

@tool("style_transfer", args_schema=StyleTransferInput, return_direct=False)
def style_transfer(content_image_path: str, style_image_path: str, output_path: Optional[str] = None, alpha: float = 1.0) -> str:
    """
    Apply artistic style transfer to an image using StyTR-2.
    
    This tool takes a content image and applies the artistic style from a style image to it.
    The result is a new image that preserves the content but renders it in the artistic style.
    
    Args:
        content_image_path: Path to the content image (the image you want to transform)
        style_image_path: Path to the style image (the artistic style to apply)
        output_path: Optional path for the output image. If not provided, will auto-generate
        alpha: Style strength (0.0-1.0). Higher values mean stronger style application
        
    Returns:
        Path to the generated stylized image
    """
    # Generate output path if not provided
    if output_path is None:
        content_name = os.path.splitext(os.path.basename(content_image_path))[0]
        style_name = os.path.splitext(os.path.basename(style_image_path))[0]
        output_path = f"stylized_{content_name}_with_{style_name}.jpg"
    
    # Get tool instance and perform transfer
    tool = get_tool_instance()
    result_path = tool.transfer_style(content_image_path, style_image_path, output_path, alpha)
    
    return f"Style transfer completed! Output saved to: {result_path}"

# For testing
if __name__ == "__main__":
    # Test the tool
    result = style_transfer(
        content_image_path="StyTR-2/demo/image_c/2_10_0_0_512_512.png",
        style_image_path="StyTR-2/demo/image_s/LevelSequence_Vaihingen.0000.png",
        output_path="test_output.jpg"
    )
    print(result) 