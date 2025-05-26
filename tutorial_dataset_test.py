from tutorial_dataset import MyDataset
import os
dataset = MyDataset('training/couple_avatar')
print(len(dataset))

item = dataset[12]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
