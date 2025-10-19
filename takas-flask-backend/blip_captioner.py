import os
import logging
from typing import Dict, Optional
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BLIPCaptioner:
    def __init__(self):
        """Initialize BLIP model for image captioning"""
        self.model_id = "Salesforce/blip-image-captioning-base"
        self.processor = None
        self.model = None
        self.device = "cpu"  # Force CPU usage
        self._model_loaded = False
        # Don't load model immediately - use lazy loading
    
    def _load_model(self):
        """Load BLIP model and processor with optimizations"""
        try:
            logger.info(f"Loading BLIP model: {self.model_id}")
            
            # Load processor and model with optimizations
            self.processor = BlipProcessor.from_pretrained(
                self.model_id,
                cache_dir="./cache",  # Local cache
                local_files_only=False,
                force_download=False  # Don't re-download if cached
            )
            
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir="./cache",  # Local cache
                local_files_only=False,
                torch_dtype=torch.float16,  # Use half precision for speed
                low_cpu_mem_usage=True,  # Reduce memory usage
                force_download=False  # Don't re-download if cached
            )
            
            # Move to CPU and optimize
            self.model.to(self.device)
            self.model.eval()
            
            # Disable gradient computation for inference
            for param in self.model.parameters():
                param.requires_grad = False
            
            self._model_loaded = True
            logger.info("BLIP model loaded successfully with optimizations!")
            
        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}")
            self.processor = None
            self.model = None
            self._model_loaded = False
    
    def generate_caption(self, image_path: str, prompt: str = "a photo of") -> Dict:
        """Generate caption for an image using BLIP"""
        try:
            # Lazy load model only when needed
            if not self._model_loaded:
                logger.info("Loading BLIP model on demand...")
                self._load_model()
            
            if self.model is None or self.processor is None:
                return {
                    'success': False,
                    'error': 'BLIP model not loaded',
                    'caption': 'Model not available',
                    'model_used': 'None'
                }
            
            # Load image
            if image_path.startswith('http'):
                # Load from URL
                response = requests.get(image_path, stream=True)
                raw_image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # Load from local file
                raw_image = Image.open(image_path).convert("RGB")
            
            # Process image and generate caption with optimizations
            inputs = self.processor(raw_image, prompt, return_tensors="pt")
            
            with torch.no_grad():
                # Ultra-fast generation parameters
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=20,  # Further reduced for speed
                    num_beams=1,  # Greedy decoding for speed
                    do_sample=False,  # Deterministic for speed
                    early_stopping=True,  # Stop early if possible
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,  # Enable caching
                    output_attentions=False,  # Disable attention output
                    output_hidden_states=False  # Disable hidden states output
                )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Clean up caption (remove prompt if it's at the beginning)
            if caption.lower().startswith(prompt.lower()):
                caption = caption[len(prompt):].strip()
            
            return {
                'success': True,
                'caption': caption,
                'model_used': 'BLIP',
                'prompt_used': prompt
            }
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {
                'success': False,
                'error': str(e),
                'caption': 'Error generating caption',
                'model_used': 'Error'
            }
    
    # Fallback function removed - no default values

# Global instance
blip_captioner = BLIPCaptioner()

def generate_image_caption(image_path: str, prompt: str = "a photo of") -> Dict:
    """Public function to generate image caption using BLIP"""
    global blip_captioner
    return blip_captioner.generate_caption(image_path, prompt)
