from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import torch
import random
import numpy as np
import einops
from torch import nn
from PIL import Image

class ControlNetSampler:
    def __init__(self, model_config_path, model_checkpoint_path):
        self.model = create_model(model_config_path).cpu()
        self.model.load_state_dict(load_state_dict(model_checkpoint_path, location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
    
    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, 
                image_resolution, ddim_steps, guess_mode, strength, scale, 
                seed, eta):
        """
        Process an input image with ControlNet without using Canny edge detection
        
        Args:
            input_image: Input image array (HWC format)
            prompt: Main prompt text
            a_prompt: Additional prompt text to append
            n_prompt: Negative prompt text
            num_samples: Number of samples to generate
            image_resolution: Target resolution for processing
            ddim_steps: Number of DDIM steps
            guess_mode: Whether to use guess mode
            strength: Control strength
            scale: Guidance scale
            seed: Random seed (-1 for random)
            eta: DDIM eta parameter
            
        Returns:
            List of generated samples
        """
        with torch.no_grad():
            # Resize input image
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape
            
            # Convert input image to control signal (no Canny detection)
            # Normalize and prepare control tensor
            control = torch.from_numpy(img.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            
            # Seed handling
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)
            
            # Memory optimization if enabled
            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            
            # Prepare conditioning
            cond = {
                "c_concat": [control],
                "c_crossattn": [self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)]
            }
            un_cond = {
                "c_concat": None if guess_mode else [control],
                "c_crossattn": [self.model.get_learned_conditioning(
                    [n_prompt] * num_samples)]
            }
            shape = (4, H // 8, W // 8)
            
            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)
            
            # Set control scales
            self.model.control_scales = (
                [strength * (0.825 ** float(12 - i)) for i in range(13)] 
                if guess_mode else ([strength] * 13)
            )
            
            # Sample with DDIM
            samples, intermediates = self.ddim_sampler.sample(
                ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond
            )
            
            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            
            # Decode samples
            x_samples = self.model.decode_first_stage(samples)
            x_samples = (
                einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
            ).cpu().numpy().clip(0, 255).astype(np.uint8)
            
            results = [x_samples[i] for i in range(num_samples)]
            
        return results[0]

sampler = ControlNetSampler('./models/cldm_v15.yaml', 'checkpoints/last.ckpt')

image_path = "training/fill50k/source/1.png"
input_image = np.array(Image.open(image_path).convert('RGB'))
prompt = "light coral circle with white background"
result = sampler.process(
    input_image, 
    prompt, 
    "high quality, detailed", 
    "blurry, low quality", 
    1, 512, 20, False, 1.0, 9.0, -1, 0.0
)
output_image = Image.fromarray(result)
output_image.save("examples/output1.png") 