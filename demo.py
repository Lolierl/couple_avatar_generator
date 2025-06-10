import os
import numpy as np
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
import tempfile

def generate_prompt(image_path):
    prompt_obj = get_prompt(image_path, openai_api_key, openai_base_url)
    return prompt_obj.message.content if hasattr(prompt_obj, "message") else prompt_obj

def run_sampler(input_image, prompt, cfg_scale=9, image_strength=0.7):
    img = Image.fromarray(np.array(input_image)).convert('RGB')
    img = np.array(process(np.array(img), flip=True))

    a_prompt = "solo, cute, beautiful face, high quality, detailed, best quality, masterpiece, beautiful eyes"
    n_prompt = "blurry, low quality, twisted, ugly, deformed, distorted, bad anatomy, bad face, text, error, missing fingers, extra digit, fewer digits"
    print(prompt)
    result = sampler.process(
        img, prompt, a_prompt, n_prompt,
        1, 512, 50, False, image_strength, cfg_scale, -1, 0.0
    )
    
    return Image.fromarray(result)


def run_adain(content_img, style_img, alpha=0.7):
    import uuid
    import shutil

    current_dir = os.getcwd()
    os.chdir("pytorch-AdaIN")

    # 保存临时文件
    content_path = f"content_{uuid.uuid4().hex}.png"
    style_path = f"style_{uuid.uuid4().hex}.png"
    content_img.save(content_path)
    style_img.save(style_path)

    cmd = (
        f"python test.py --content {content_path} --style {style_path}"
        f" --alpha {alpha} --content_size 512 --style_size 512 --output . --save_ext .png"
    )
    os.system(cmd)

    stylized_name = f"{os.path.splitext(content_path)[0]}_stylized_{os.path.splitext(style_path)[0]}.png"
    result = None
    if os.path.exists(stylized_name):
        result = Image.open(stylized_name).convert('RGB')
        os.remove(stylized_name)

    # 清理临时图片
    os.remove(content_path)
    os.remove(style_path)
    os.chdir(current_dir)

    return result


def full_pipeline(input_image, input_prompt = None, alpha=0.7, cfg_scale=9, image_strength=0.7):
    # Step 1: 生成 Prompt
    if input_image is None:
        raise gr.Error("请先上传图片")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        input_image.save(tmp.name)
        input_path = tmp.name
    
    prompt = generate_prompt(input_image)
    if input_prompt:
        prompt = prompt + ", " + input_prompt
    # Step 2: 生成配对头像
    sampled_img = run_sampler(input_image, prompt, cfg_scale=cfg_scale, image_strength=image_strength)
    # Step 3: 风格迁移
    stylized_img = run_adain(sampled_img, input_image, alpha)
    return stylized_img

demo = gr.Interface(
    fn=full_pipeline,
    inputs=[
        gr.Image(type="pil", label="上传一张头像"),
        gr.Textbox(lines=2, placeholder="可选：输入你想要的情侣头像特征(In English tags, e.g. girl, facing left, pink dress)", label="自定义 Prompt"), 
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="风格化程度 (alpha)"), 
        gr.Slider(minimum=0, maximum=15, value = 9, label = "文本提示强度 (CFG Scale)", step=1),
        gr.Slider(minimum=0, maximum=1, value = 0.7, label = "图像提示强度 (Image Strength)")
    ],
    outputs=[
        gr.Image(type="pil", label="你的情侣头像❤️")
    ],
    title="🎨 CUPID: AI情侣头像生成器",
    description=(
        "✨ 上传一张你的头像，CUPID 自动帮你生成专属的情侣头像。\n\n"
        "📌 可选输入你想要的对方形象关键词（如：glasses, smiling）来调整内容。\n\n"
        "🎨 通过风格迁移技术，让新头像与原图风格匹配。\n\n"
        "💕 快来试试，生成属于你们的专属情侣头像吧！\n\n"
        "Reminder: 推荐上传human头像，效果最佳哦！"
    )
)

if __name__ == "__main__":
    load_dotenv()
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    # 设置 API 参数
    openai_api_key = os.getenv("INFINI_API_KEY")
    openai_base_url = os.getenv("INFINI_BASE_URL")
    # 模型与工具导入
    from get_gpt4_prompt import get_prompt
    from sampler import ControlNetSampler, process

    model_config = './models/cldm_v15.yaml'
    model_ckpt = 'checkpoints/last.ckpt'
    sampler = ControlNetSampler(model_config, model_ckpt)
    demo.launch(server_name='0.0.0.0', server_port=7860)