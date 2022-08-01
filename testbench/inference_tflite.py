"""
Usage

1. Command To Run:
python testbench/inference_tflite.py \
 -m ~/deit_proj/model_converter/models/tflite_models/fully_quant_from_onnx_deit_tiny_distilled_patch16_224.tflite

2. Dataset should be structured as ImageNet:
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
"""

import os
import glob
import time
import datetime
import argparse
import json
import tensorflow as tf
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from loader import load_img_tensor, load_caffee_labels

def eval_acc(
    tflite_path,  # path to tfilte model
    dataset_path,     # path to imagenet validation set
    label_path, mean, std):

    # Load labels & dataset
    caffee_maps = load_caffee_labels(label_path) 
    image_path_set = sorted(glob.glob(os.path.join(dataset_path, "*", "*.JPEG")))

    # Load the TFLite model & parameters
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    print(input_details)
    print(output_details)
    
    # channel-first or channel-last
    if(str(input_shape[1])=="3"):
        channel_first = True
    else:
        channel_first = False


    # Check if the model is quantized     
    if(input_details[0].get('dtype')==np.uint8):
        quant_scale = input_details[0]['quantization_parameters']['scales'][0]
        quant_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]

    with open("val_acc_%s"%os.path.split(tflite_path)[-1], 'a') as f:
        f.write("\n%s\n"%datetime.datetime.now())
        f.write(os.path.split(tflite_path)[-1])
        f.write(str(input_details))
        f.write("--------------------------------\n")


    corrects = 0
    avg_latency = 0 
    for i, image_path in enumerate(image_path_set, 1):
        # Load nomalized (standaraized) image into nparray
        img_numpy = load_img_tensor(
            img_path=image_path, 
            mean=IMAGENET_DEFAULT_MEAN, 
            std=IMAGENET_DEFAULT_STD
        ).numpy()

        # Quantization
        if(input_details[0].get('dtype')==np.uint8):
            img_numpy = (img_numpy/quant_scale) + quant_zero_point  
            img_numpy = np.array(img_numpy, dtype=np.uint8)

        # (N,C,H,W) --> (N,H,W,C) 
        if(channel_first==False):   
            img_numpy = np.transpose(img_numpy, (0, 2, 3, 1))


        # Inference
        interpreter.set_tensor(input_details[0]['index'], img_numpy)
        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Proccess result
        pred_class = np.argmax(output_data[0])
        prediction = caffee_maps.get(str(pred_class))[0]
        ans = image_path.split("/")[-2]  # Folder's name = caffee class
        latency = stop_time - start_time
        avg_latency = avg_latency*(i-1)/i + latency*(1/i)
        if(prediction==ans):
            corrects += 1

        print("----------------------------------")
        print(i)
        print("  image: %s" %image_path)
        print("  model: %s" %os.path.split(tflite_path)[-1])
        print("   pred: %s (%d)" %(prediction, pred_class))
        print("    ans: %s" %ans)
        print("latency: %f ms" %(latency*1000))
        print("avg_lat: %f ms" %(avg_latency*1000))
        print("    acc: %d/%d (%f)" %(corrects, i, corrects*100/i))
        print("----------------------------------")

        # Write result to file
        if((i+1)%500==0):
            with open("val_acc_%s"%os.path.split(tflite_path)[-1], 'a') as f:
                f.write("latency, acc: %f, %d/%d (%f)\n" %( (avg_latency*1000), corrects, i, (corrects*100)/i ))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    parser.add_argument(
        '-d', '--dir', default="/home/yysung/imagenet/val", help='Path to directory of imageNet validation set.')
    parser.add_argument(
        '-l', '--labels', default="/home/heyward-lab/deit_proj/testbench/labels/imagenet_class_index.json", help='Path of labels file.')
    parser.add_argument(
        '-mean', default=IMAGENET_DEFAULT_MEAN, help='Dataset mean for normalization.')
    parser.add_argument(
        '-std', default=IMAGENET_DEFAULT_STD, help='Dataset standard deviation for normalization.')
    args = parser.parse_args()

    eval_acc(
        tflite_path=args.model, 
        dataset_path=args.dir, 
        label_path=args.labels, 
        mean=args.mean, 
        std= args.std
    )


if __name__ == "__main__":
    main()    
