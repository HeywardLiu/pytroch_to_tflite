
"""
deit_tiny_patch16_224
 - arithmetic ops: 3.309 GFLOPs, 1.654 GMACs
"""

import glob
import os
import sys
import random
import tensorflow as tf
import torchvision
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
from pt_to_tflite import TFLITE_PATH, TF_PATH

VAL_PATH = os.path.abspath("/home/yysung/imagenet/val")

def gen_representative_data():
    calibration_times = 100
    val_set = sorted(glob.glob(os.path.join(VAL_PATH, "*", "*.JPEG")))
    img_set = []

    for i in range(calibration_times):
        img_path = random.choice(val_set)  # Select rep. data randomly
        print(img_path)
        img = Image.open(img_path).convert('RGB')
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        img_tensor = transform(img) 
        img_set.append(img_tensor.numpy())

    for steps, input_value in enumerate(tf.data.Dataset.from_tensor_slices(img_set).batch(1).take(calibration_times)):
        print("calibrating %d" %(steps+1))
        yield [input_value]


def tf_to_quant_tflite(tf_path, tflite_path, fully_quant:bool):
    print("\n\nConverting the model from tensorflow to tflite ...")
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)  # path to the SavedModel directory
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 

    if fully_quant:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,    # enable TensorFlow ops.
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8 
        ]
        converter.representative_dataset = gen_representative_data
        converter.inference_input_type = tf.uint8  
        converter.inference_output_type = tf.uint8 
        tflite_path = tflite_path.replace('from', 'fully_quant_from')
    else:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS     # enable TensorFlow ops.
        ]
        tflite_path = tflite_path.replace('from', 'dynamic_quant_from')
    
    print("\nQuantieing Model...\n")
    tflite_model = converter.convert()
    
    # Save the model.
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print("Quantize Successfully")
    print("Model: %s" %tflite_path)
    

if __name__ == "__main__":
    tf_to_quant_tflite(
        tf_path=TF_PATH, 
        tflite_path=TFLITE_PATH,
        fully_quant=True
    )