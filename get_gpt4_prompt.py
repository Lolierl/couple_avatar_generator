import base64
import os
from openai import OpenAI
from PIL import Image
import io
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
_client = None
_examples = None
def encode_image(image_input, flip = False):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")  # 强制转为RGB
    else:
        image = image_input.convert("RGB")  # 确保是RGB模式
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def init_once(key, url):
    global _client, _examples
    if _client is None:
        _client = OpenAI(api_key=key, base_url=url)
        print("Client created")

    if _examples is None:
        _examples = {
            "example1": encode_image("./test/male/181.png", flip=True),
            "output1": encode_image("./test/female/181.png", flip = False),
            "example2": encode_image("./test/female/546.png", flip=True),
            "output2": encode_image("./test/male/546.png", flip = False)
        }
        print("Examples encoded")
        
def get_prompt(image, key, url):
    init_once(key, url)
    base64_image = encode_image(image, flip=True)
    
    # 读取示例图片
    #example1_path = "./test/male/391.png"  # 请确保这些示例图片存在
    #example2_path = "./test/male/111.png"
    #example3_path = "./test/female/385.png"
    
    #base64_example1 = encode_image(example1_path)
    #base64_example2 = encode_image(example2_path)
    #base64_example3 = encode_image(example3_path)
    base64_example1 = _examples["example1"]
    output1 = _examples["output1"]
    base64_example2 = _examples["example2"]
    output2 = _examples["output2"]
    # 创建API请求
    response = _client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful prompt generator. """
            },
            {
                "role": "user",
                "content": """I have a pair of couple avatar, you need to observe one of the images and predict the feature of the other couple avatar to train a diffusion model. You should prompt it out with booru tags. One pair of couple avatars consists of a male and female avatar with roughly same oritation and position in the image. Do not directly describe the provided figure.
                            Try your best to predict the features of the other corresponding couple avatar, you should at least point out the gender, species, body orientation, position, posture. You should also add a few guess features such as clothing, expression.Only write a prompt to predict the features of the other couple avatar in booru tags. Here are two examples. """
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
                        "text": "The corresponding couple avatar is like below:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{output1}",
                            "detail": "high"
                        }
                    },                    
                    {
                        "type": "text",
                        "text": "The response should be like: female, human, facing slightly right, center of image, upper body, head tilt, closed eyes, smiling, blush, holding doll, long pink hair, side braid, hair over one eye, off shoulder dress, frilled sleeves, pink theme, pastel color scheme, soft shading, anime style, simple background, light blue background, triangle pattern, chibi doll, silver hair doll, bow on shoulder."
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
                        "text": "The corresponding couple avatar is like below:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{output2}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "The response should be like: male, human, facing left, left side of image, profile view, side view, looking away, short brown hair, tousled hair, blush, casual clothing, dark shirt, soft shading, watercolor style, anime style, sky background, clouds, pastel color palette."
                    },
                    
                    {
                        "type": "text",
                        "text": "Instruction:Now Read this image carefully and write a prompt for its couple avatar like examples. You should only generate prompt based on the main character in the image, ignore other characters in the image. Remember to first generate species, gender, face and body orientation, position in the image, posture, then generate other features like clothing, expression, hair style, color, etc. Do not directly describe the provided figure."
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
    openai_api_key = os.environ.get("INFINI_API_KEY")
    openai_base_url = os.environ.get("INFINI_BASE_URL")
    image_path = "/root/autodl-tmp/ControlNet/test/male/630.png"
    image = Image.open(image_path).convert("RGB") 
  
    response = get_prompt(image, openai_api_key, openai_base_url)
    print(response) 