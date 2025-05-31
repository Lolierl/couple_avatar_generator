import os
import numpy as np
from PIL import Image
from dataset import MyDataset

def save_image(array, out_path, value_range='[0,1]'):
    """Convert float32 image array to uint8 and save as PNG."""
    array = np.squeeze(array)

    # 范围变换
    if value_range == '[0,1]':
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif value_range == '[-1,1]':
        array = ((np.clip(array, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    else:
        raise ValueError("Unsupported value_range")

    image = Image.fromarray(array)
    image.save(out_path)

dataset = MyDataset('test')
print("Dataset size:", len(dataset))

item = dataset[15]

save_dir = 'test_dataset'
os.makedirs(save_dir, exist_ok=True)

save_image(item['hint'], os.path.join(save_dir, 'source.png'), value_range='[0,1]')

save_image(item['jpg'], os.path.join(save_dir, 'target.png'), value_range='[-1,1]')

with open(os.path.join(save_dir, 'prompt.txt'), 'w', encoding='utf-8') as f:
    f.write(item['txt'])

print("Saved source.png, target.png, prompt.txt to test_dataset/")