import os
import time
import datetime
import glob
import argparse

import tensorflow as tf
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from loader import load_img_tensor, load_caffee_labels


def eval_acc(
    tflite_path,  # path to tfilte model
    val_path,     # path to imagenet validation set
    label_path, mean, std):

    # GPU settings
    gpus_list = tf.config.list_physical_devices('GPU')
    if gpus_list is not None:
        try:
            for gpu in gpus_list:
                print("\nRestircting GPU Memory Growth...", gpu,"\n")
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
                )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus_list), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
        gpu_strategy = tf.distribute.MirroredStrategy(devices=tf.config.list_logical_devices('GPU'))

    # Load labels & dataset
    caffee_maps = load_caffee_labels(label_path) 
    image_path_set = sorted(glob.glob(os.path.join(val_path, "*", "*.JPEG")))

    # Load the TFLite model & parameters
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    if(input_details[0].get('dtype')==np.uint8):
        quant_scale = input_details[0]['quantization_parameters']['scales'][0]
        quant_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
        print("scales: %f\n" %quant_scale)
        print("zero points: %d\n" %quant_zero_point)

    with open("val_acc_%s"%os.path.split(tflite_path)[-1], 'a') as f:
        f.write("\n%s\n"%datetime.datetime.now())
        f.write(os.path.split(tflite_path)[-1])
        f.write("--------------------------------\n")

    with gpu_strategy.scope():
        corrects = 0
        latency_sum = 0 
        for i, image_path in enumerate(image_path_set):
            # Load nomalized (standaraized) image as nparray
            img_numpy = load_img_tensor(
                img_path=image_path, 
                mean=IMAGENET_DEFAULT_MEAN, 
                std=IMAGENET_DEFAULT_STD
            ).numpy()

            # Quantization
            if(input_details[0].get('dtype')==np.uint8):
                img_numpy = (img_numpy/quant_scale) + quant_zero_point  
                img_numpy = np.array(img_numpy, dtype=np.uint8)

            # Inference
            interpreter.set_tensor(input_details[0]['index'], img_numpy)
            start_time = time.time()
            interpreter.invoke()
            stop_time = time.time()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Proccess result
            pred_class = np.argmax(output_data[0])
            prediction = caffee_maps.get(str(pred_class))[0]
            ans = image_path.split("/")[-2]
            latency = stop_time - start_time
            latency_sum += latency
            if(prediction==ans):
                corrects += 1

            print("----------------------------------")
            print("  image: %s" %image_path)
            print("  model: %s" %os.path.split(tflite_path)[-1])
            print("   pred: %s (%d)" %(prediction, pred_class))
            print("    ans: %s" %ans)
            print("latency: %f ms" %(latency*1000))
            print("avg_lat: %f ms" %((latency_sum/(i+1))*1000))
            print("    acc: %d/%d (%f)" %(corrects, (i+1), corrects*100/(i+1)))
            print("----------------------------------")

            # Write result to file
            if((i+1)%500==0):
                with open("val_acc_%s"%os.path.split(tflite_path)[-1], 'a') as f:
                    f.write("latency, acc: %f, %d/%d (%f)\n" %( (latency_sum*1000)/(i+1), corrects, (i+1), (corrects*100)/(i+1) ))


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

    eval_acc(tflite_path=args.model, 
             val_path=args.dir, 
             label_path=args.labels, 
             mean=args.mean, 
             std= args.std)


if __name__ == "__main__":
    main()    
