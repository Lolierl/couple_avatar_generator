from share import *
import config
import os
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
from PIL import Image, ImageOps

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
def process(image_array, flip):
    h, w, _ = image_array.shape
    longest = max(h, w)
    delta_w = longest - w
    delta_h = longest - h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    padded = cv2.copyMakeBorder(image_array, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))    
    if flip:
        padded = cv2.flip(padded, 1)
    resized = cv2.resize(padded, (512, 512), interpolation=cv2.INTER_CUBIC)
    return resized

if __name__ == "__main__":
    sampler = ControlNetSampler('./models/cldm_v15.yaml', '/root/autodl-tmp/ControlNet/checkpoints/last.ckpt')

    input_dir = "/root/autodl-tmp/ControlNet/test"
    output_dir = "raw_examples"
    os.makedirs(output_dir, exist_ok=True)
    import os
    import json
    from PIL import Image
    import numpy as np

    # 加载 prompt.json
    json_path = "/root/autodl-tmp/ControlNet/test/prompt.json"
    with open(json_path, 'r') as f:
        prompt_data = json.load(f)

    # 设置目录
    base_dir = os.path.dirname(json_path)
    output_dir = "/root/autodl-tmp/ControlNet/results"
    input_save_dir = os.path.join(output_dir, "inputs")
    output_save_dir = os.path.join(output_dir, "outputs")

    os.makedirs(input_save_dir, exist_ok=True)
    os.makedirs(output_save_dir, exist_ok=True)

    # 处理前 100 条
    for idx, item in enumerate(prompt_data[:100]):
        source_path = os.path.join(base_dir, item["source"])
        prompt = item["prompt"]
        print(f"Processing {source_path}...")

        # 打开源图像
        raw_image = Image.open(source_path).convert('RGB')
        raw_image_np = np.array(raw_image)

        condition_image = process(raw_image_np, flip = True)  

        # 设置提示词
        positive_prompt = "solo, cute, beautiful face, high quality, detailed, best quality, masterpiece"
        negative_prompt = "blurry, low quality, twisted, ugly, deformed, distorted, bad anatomy, bad face, text, error, missing fingers, extra digit, fewer digits"

        # 调用采样器
        result = sampler.process(
            condition_image,  # 条件图像
            prompt,
            positive_prompt,
            negative_prompt,
            1, 512, 50, False, 0.9, 11.0, -1, 0.0
        )

        output_image = Image.fromarray(result)

        input_save_path = os.path.join(input_save_dir, f"input_{idx:03d}.png")
        output_save_path = os.path.join(output_save_dir, f"output_{idx:03d}.png")
        
        raw_image.save(input_save_path)
        output_image.save(output_save_path)

        print(f"Saved: {input_save_path} and {output_save_path}")