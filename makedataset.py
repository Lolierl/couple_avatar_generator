import os
import re
import json
from PIL import Image

def is_image_file(filename):
    """检查文件是否为图片"""
    try:
        with Image.open(filename) as img:
            return True
    except:
        return False

def rename_files_in_directory(directory):
    """重命名目录中的图片文件并返回排序后的图片列表"""
    files = sorted(os.listdir(directory), key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])
    
    image_files = [f for f in files if is_image_file(os.path.join(directory, f))]
    
    renamed_files = []
    for idx, filename in enumerate(image_files):
        ext = os.path.splitext(filename)[1]
        new_name = f"{idx}{ext}"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        os.rename(old_path, new_path)
        renamed_files.append(new_name)
        print(f"Renamed: {filename} -> {new_name}")
    
    return renamed_files

def generate_prompts(male_files, female_files, output_path):
    """生成prompt.json文件"""
    prompts = []
    
    # 确保两个目录文件数量相同
    min_length = min(len(male_files), len(female_files))
    
    for i in range(min_length):
        prompt = {
            "source": f"male/{male_files[i]}",
            "target": f"female/{female_files[i]}",
            "prompt": "a female couple avatar of this image"
        }
        prompts.append(prompt)
        prompt = {
            "source": f"female/{female_files[i]}",
            "target": f"male/{male_files[i]}",
            "prompt": "a male couple avatar of this image"
        }
        prompts.append(prompt)
    
    # 写入JSON文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(prompts, f, indent=4)
    
    print(f"Generated prompts file at: {output_path}")

# 主程序
if __name__ == "__main__":
    # 定义目录路径
    male_dir = "training/couple_avatar/male"
    female_dir = "training/couple_avatar/female"
    output_json = "training/couple_avatar/prompt.json"
    
    # 处理male目录
    male_files = []
    if os.path.exists(male_dir):
        print(f"Processing directory: {male_dir}")
        male_files = rename_files_in_directory(male_dir)
    else:
        print(f"Directory not found: {male_dir}")
    
    # 处理female目录
    female_files = []
    if os.path.exists(female_dir):
        print(f"Processing directory: {female_dir}")
        female_files = rename_files_in_directory(female_dir)
    else:
        print(f"Directory not found: {female_dir}")
    
    generate_prompts(male_files, female_files, output_json)