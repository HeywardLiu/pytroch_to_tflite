import torch
import torchvision
import ast
import glob
import os
from PIL import Image
import json
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

"""
    Image data must go through two transforms before running inference:
    1. normalization: f = (input - mean) / std
    2. quantization: q = f / scale + zero_point
    The following code combines the two steps as such:
    q = (input - mean) / (std * scale) + zero_point

    However, if std * scale equals 1, and mean - zero_point equals 0, the input
    does not need any preprocessing (but in practice, even if the results are
    very close to 1 and 0, it is probably okay to skip preprocessing for better
    efficiency; we use 1e-5 below instead of absolute zero).
"""


def load_img_tensor(img_path, mean, std):
    img = Image.open(img_path).convert('RGB')   
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(224),
                    torchvision.transforms.CenterCrop(224), 
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean, std)
                ])
    img_tensor = transform(img).unsqueeze(0)  # add batch channel
    return img_tensor

    
def load_caffee_labels(file_path):
    with open(file_path) as json_files:
        map = json.load(json_files) 
    return map


# def load_label(labels_path):  
#     with open(labels_path, "r") as file:
#         labels = file.read()
#     labels_dict = ast.literal_eval(labels)  # convert into a dict
#     return labels_dict 


# def load_ans():
#     map_path = os.path.join(os.path.abspath(os.pardir), "deit_proj", "testbench", "labels", "ILSVRC2012_mapping.txt")
#     ans_path = os.path.join(os.path.abspath(os.pardir), "deit_proj", "testbench", "labels", "ILSVRC2012_validation_ground_truth.txt")
    
#     # Load the mapping labels from caffee to ImageNet-1k 
#     map = {}
#     with open(map_path, 'r') as files:  
#         for line in files:
#             token = line.split()
#             d = {token[0]: token[1]}
#             map.update(d)

#     # Transform the ILSVRC2012 ground truth from caffee into ImageNet-1k
#     ans = []
#     with open(ans_path) as files:
#         for line in files:
#             ans.append(map[line.split()[0]])
#     ans = tuple(ans)
#     return ans
    

