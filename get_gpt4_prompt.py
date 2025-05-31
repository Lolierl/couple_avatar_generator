import base64
import os
from openai import OpenAI
from PIL import Image
import io
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 直接设置API密钥和基础URL
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")
client = OpenAI(api_key='sk-SglvXVSfGEOgXGLrgmSzL5kyYixs4cUxD3PserEOc99YsMzl', base_url='https://api.nuwaapi.com/v1')

def encode_image(image_input):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")  # 强制转为RGB
    else:
        image = image_input.convert("RGB")  # 确保是RGB模式
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_prompt(image_path):
    # 读取并翻转输入图片
    image = Image.open(image_path)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    base64_image = encode_image(flipped_image)
    
    # 读取示例图片
    example1_path = "./test/male/391.png"  # 请确保这些示例图片存在
    example2_path = "./test/male/111.png"
    example3_path = "./test/female/385.png"
    
    base64_example1 = encode_image(example1_path)
    base64_example2 = encode_image(example2_path)
    base64_example3 = encode_image(example3_path)
    
    # 创建API请求
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful prompt generator. I have a pair of couple avatar, you need to observe one of the images and predict the feature of the other couple avatar to train a diffusion model. You should prompt it out with booru tags. One pair of couple avatars consists of a male and female avatar with same oritation. Do not directly describe the provided figure.
                            Try your best to predict the features of the other corresponding couple avatar, you should at least point out the gender, species, body orientation, position. You can also add a few guess features such as clothing, expression."""
            },
            {
                "role": "user",
                "content": "Only write a prompt to predict the features of the other couple avatar in booru tags. Here are two examples. "
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_example1}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "The response should be like: female, human, facing right, center of the image, white background, pick hair, pink eyes."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_example2}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "The response should be like: female, human, face left, center of the image, fleshcolor background, black hair."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_example3}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "The response should be like: male, human, face forward, center of the image, smiling, black robe."
                    },
                    {
                        "type": "text",
                        "text": "Instruction:Now Read this image carefully and write a prompt for it like examples."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=5000
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # 测试函数
    image_path = "test.jpg"
    response = get_prompt(image_path)
    print(response) 