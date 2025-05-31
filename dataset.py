import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
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

class MyDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        self.dataset_path = dataset_path
        with open(os.path.join(dataset_path, 'prompt.json'), 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        dataset_path = self.dataset_path
        source = cv2.imread(os.path.join(dataset_path,  source_filename))
        target = cv2.imread(os.path.join(dataset_path, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # Resize images to 512x512.
        source = process(source, flip = True)
        target = process(target, flip = False)
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

