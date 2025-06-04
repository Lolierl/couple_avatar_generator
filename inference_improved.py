import os
from dotenv import load_dotenv
load_dotenv()
# 修复 MKL 线程层冲突
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# 直接设置API配置
openai_api_key = "sk-dahw6xzrbtarbh4w"

openai_base_url = "https://cloud.infini-ai.com/maas/v1"

print("OpenAI API Key:", openai_api_key)
print("OpenAI Base URL:", openai_base_url)
# 先导入 numpy 避免冲突
import numpy as np

from PIL import Image

# 1. 生成prompt
from get_gpt4_prompt import get_prompt

def generate_prompt(image_path, openai_api_key, openai_base_url):
    prompt_obj = get_prompt(image_path, openai_api_key, openai_base_url)
    # 你可能需要根据实际返回结构调整
    return prompt_obj.message.content if hasattr(prompt_obj, "message") else prompt_obj

# 2. 调用sampler生成B
def run_sampler(input_image, prompt, output_path):
    # 这里假设sampler.py支持import调用
    from sampler import ControlNetSampler, process
    # 路径根据你实际模型和配置调整
    model_config = './models/cldm_v15.yaml'
    model_ckpt = 'checkpoints/last.ckpt'  
    print("加载”）"
    sampler = ControlNetSampler(model_config, model_ckpt)
    # 预处理
    img = Image.open(input_image).convert('RGB')
    print("原始图片大小:", img.size)
    img = np.array(process(np.array(img), flip=True))  # 你可以根据需要调整flip
    # prompt参数
    a_prompt = "solo, cute, beautiful face, high quality, detailed, best quality, masterpiece"
    n_prompt = "blurry, low quality, twisted, ugly, deformed, distorted, bad anatomy, bad face, text, error, missing fingers, extra digit, fewer digits"
    result = sampler.process(
        img, prompt, a_prompt, n_prompt,
        1, 512, 50, False, 0.7, 15.0, -1, 0.0
    )
    Image.fromarray(result).save(output_path)

# 3. 用AdaIN做风格迁移
def run_adain(content_path, style_path, output_path, alpha=1.0, 
              preserve_color=False, content_size=512, style_size=512, crop=False):
    """
    运行 AdaIN 风格迁移
    
    参数:
        content_path: 内容图片路径（ControlNet 生成的配对头像）
        style_path: 风格图片路径（原始输入图片）
        output_path: 输出路径
        alpha: 风格化程度 (0.0-1.0)，1.0为完全风格化，0.0为保持原内容
        preserve_color: 是否保留内容图片的颜色
        content_size: 内容图片大小（0表示保持原始大小）
        style_size: 风格图片大小（0表示保持原始大小）
        crop: 是否中心裁剪成正方形
    """
    # 确保 alpha 在合理范围内
    alpha = max(0.0, min(1.0, alpha))
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    
    # 保存当前目录
    current_dir = os.getcwd()
    
    # 切换到 pytorch-AdaIN 目录执行
    os.chdir('pytorch-AdaIN')
    
    # 获取绝对路径
    content_abs = os.path.abspath(os.path.join(current_dir, content_path))
    style_abs = os.path.abspath(os.path.join(current_dir, style_path))
    
    # 构建命令
    cmd = f"python test.py --content {content_abs} --style {style_abs}"
    cmd += f" --alpha {alpha}"
    cmd += f" --content_size {content_size}"
    cmd += f" --style_size {style_size}"
    if preserve_color:
        cmd += " --preserve_color"
    if crop:
        cmd += " --crop"
    cmd += " --output . --save_ext .png"
    
    print(f"执行命令: {cmd}")
    print(f"参数设置:")
    print(f"  - Alpha (风格化程度): {alpha}")
    print(f"  - 保留颜色: {'是' if preserve_color else '否'}")
    print(f"  - 内容图片大小: {content_size if content_size > 0 else '原始大小'}")
    print(f"  - 风格图片大小: {style_size if style_size > 0 else '原始大小'}")
    print(f"  - 中心裁剪: {'是' if crop else '否'}")
    
    os.system(cmd)
    
    # 计算输出文件名
    content_stem = os.path.splitext(os.path.basename(content_path))[0]
    style_stem = os.path.splitext(os.path.basename(style_path))[0]
    stylized_name = f"{content_stem}_stylized_{style_stem}.png"
    
    # 在 pytorch-AdaIN 目录下检查文件
    if os.path.exists(stylized_name):
        import shutil
        # 构建目标路径
        target_path = os.path.join(current_dir, output_dir, stylized_name)
        # 复制文件到目标位置
        shutil.copy2(stylized_name, target_path)
        # 返回原目录
        os.chdir(current_dir)
        print(f"成功生成风格迁移图片: {os.path.join(output_dir, stylized_name)}")
        return os.path.join(output_dir, stylized_name)
    else:
        # 返回原目录
        os.chdir(current_dir)
        print(f"警告：未找到输出文件 {stylized_name}")
        # 列出 pytorch-AdaIN 目录下的文件帮助调试
        print("pytorch-AdaIN 目录下的文件：")
        for f in os.listdir('pytorch-AdaIN'):
            if f.endswith('.png'):
                print(f"  - {f}")
        return None

if __name__ == "__main__":
    # 输入图片路径（比如一张女性头像）
    input_image = "./test/female/0.png"
    
    # 可调参数
    alpha = 0.7  # 风格化程度：0.0-1.0，建议0.6-0.8
    preserve_color = False  # 是否保留原始颜色
    content_size = 512  # 内容图片大小（0=保持原始）
    style_size = 512  # 风格图片大小（0=保持原始）
    crop = False  # 是否中心裁剪成正方形
    
    print("=== 情侣头像生成流程 ===\n")
    print(f"输入图片: {input_image}")
    print(f"参数设置:")
    print(f"  - Alpha (风格化程度): {alpha}")
    print(f"  - 保留颜色: {'是' if preserve_color else '否'}")
    print(f"  - 图片大小: {content_size}")
    print(f"  - 中心裁剪: {'是' if crop else '否'}")
    
    # 1. 生成prompt（描述配对头像）
    print("\n步骤1: 生成配对头像的描述...")
    prompt = generate_prompt(input_image, openai_api_key, openai_base_url)
    print("生成的prompt:", prompt)
    
    # 2. 使用 ControlNet 生成配对头像
    print("\n步骤2: 使用 ControlNet 生成配对头像...")
    sampler_output = "B.png"  # 生成的配对头像
    run_sampler(input_image, prompt, sampler_output)
    print("采样完成:", sampler_output)
    
    # 3. 风格迁移：让生成的头像与原图风格一致
    print("\n步骤3: 风格迁移，统一画风...")
    print(f"  - 内容图片: {sampler_output} (ControlNet 生成的配对头像)")
    print(f"  - 风格图片: {input_image} (原始输入图片)")
    
    stylized_output = "final_couple_avatar.png"
    final_path = run_adain(
        sampler_output, 
        input_image, 
        stylized_output, 
        alpha=alpha,
        preserve_color=preserve_color,
        content_size=content_size,
        style_size=style_size,
        crop=crop
    )
    
    if final_path:
        print(f"\n✅ 全部完成！")
        print(f"=====================================")
        print(f"原始图片: {input_image}")
        print(f"生成的配对图片: {sampler_output}")
        print(f"风格统一后的最终图片: {final_path}")
        print(f"=====================================")
    else:
        print("\n❌ 风格迁移失败！")