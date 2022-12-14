import glob
import os
import sys
import random
import argparse

import torch
import torchvision
import onnx
import onnx_tf.backend 
import tensorflow as tf

import numpy as np
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


RESOLUTION = 224
CALIBRATION_TIMES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
channel_first = True


def pt_to_onnx(pt_model:torch.nn.Module, onnx_path, resolution=224):
    print("\n-----------------------------------------------")
    print("| Converting the model from pytorch to onnx ... |")
    print("-----------------------------------------------\n")
    global RESOLUTION
    RESOLUTION = resolution
    dummy_input = torch.randn((1, 3, resolution, resolution)).to(DEVICE)  # (N, C, H, W)
    torch.onnx.export(model = pt_model,
                      args = dummy_input,    # Give a random input to trace the structure              
                      f = onnx_path,         # path to export
                      verbose = True)        # print infos of converted layers on terminal


def onnx_to_tf(onnx_path, tf_path):
    print("\n----------------------------------------------------")
    print("| Converting the model from onnx to tensorflow ... |")
    print("----------------------------------------------------\n")

    # where the representation of tensorflow model will be stored
    onnx_model = onnx.load(onnx_path)  # load onnx model
    
    # prepare function converts an ONNX model to an internel representation
    # of the computational graph called TensorflowRep and returns
    # the converted representation.
    tf_rep = onnx_tf.backend.prepare(model = onnx_model,
                                     device = 'CPU')
    
    # export_graph function obtains the graph proto corresponding to the ONNX
    # model associated with the backend representation and serializes
    # to a protobuf file.
    tf_rep.export_graph(tf_path)


def tf_to_tflite(tf_path, tflite_path):
    print("\n------------------------------------------------------")
    print("| Converting the model from tensorflow to tflite ... |")
    print("------------------------------------------------------\n")

    # Convert model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS     # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    tflite_path = tflite_path.replace(   
        os.path.split(tflite_path)[-1], 
        "fp32_{}".format(os.path.split(tflite_path)[-1])
    )

    # Save model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print("Convert model Successfully")
    print("Saved in: %s" %tflite_path)


def gen_representative_data():
    VAL_SETS_PATH = os.path.abspath("/home/yysung/imagenet/val")
    val_set = sorted(glob.glob(os.path.join(VAL_SETS_PATH, "*", "*.JPEG")))
    rep_data_set = random.choices(val_set, k=CALIBRATION_TIMES)  # Select rep. data randomly
    img_set = []

    for img_path in rep_data_set:
        img = Image.open(img_path).convert('RGB')
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        img_numpy = transform(img).numpy()
        if(channel_first==False):  
            img_numpy = np.transpose(img_numpy, (1, 2, 0))  # (B, C, H, W) -> (B, H, W, C)
        img_set.append(img_numpy)
    
    for iter, input_value in enumerate(tf.data.Dataset.from_tensor_slices(img_set).batch(1), 0):
        print("Input Range Calibration: %d, %s" %(iter+1, rep_data_set[iter]))
        yield [input_value]


def tf_to_quant_tflite(tf_path, tflite_path, fully_quant:bool, calibration_times):
    print("\n-----------------------------------------------------------------------------------")
    print("| Converting the model from tensorflow to tflite and FULLY QUANTIZING TO UINT8... |")
    print("-----------------------------------------------------------------------------------\n")

    # Clarify model's shaple of input tensor
    tf_model =  tf.saved_model.load(tf_path)
    input_shape = tf_model.signatures["serving_default"].inputs[0].shape # input shape of the model
    global channel_first
    channel_first = True if input_shape[1]==3 else False  

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)  # path to the SavedModel directory
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 

    # Model configs
    global CALIBRATION_TIMES
    CALIBRATION_TIMES = calibration_times
    print(CALIBRATION_TIMES)

    if fully_quant:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,    # enable TensorFlow ops.
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8 
        ]
        converter._experimental_new_quantizer = True
        converter.representative_dataset = gen_representative_data
        converter.inference_input_type = tf.uint8  
        converter.inference_output_type = tf.uint8 
        tflite_path = tflite_path.replace(   
            "{}".format(os.path.split(tflite_path)[-1]), 
            "uint8_cal{}_{}".format(CALIBRATION_TIMES, os.path.split(tflite_path)[-1])
        )
    else:  # Dynamic Range Qunatization
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS     # enable TensorFlow ops. (e.g. Flex Delegates)
        ]
        tflite_path = tflite_path.replace(   
            "{}".format(os.path.split(tflite_path)[-1]), 
            "dynamic_fp32_{}".format(os.path.split(tflite_path)[-1])
        )
    
    print("\nQuantizing Model...\n")
    tflite_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

    # Save the model.
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print("Quantize model to uint8 Successfully")
    print("Saved in: %s" %tflite_path)
    

def main():
    # Model Configs
    DEIT_MODELS = (
        'deit_tiny_patch16_224',
        'deit_tiny_distilled_patch16_224',
        'deit_base_patch16_384',
        'deit_base_distilled_patch16_384'
    )
    MODEL_NAME = DEIT_MODELS[0]
    ONNX_PATH = os.path.abspath("model_converter/models/onnx_models/from_onnx_{}.onnx".format(MODEL_NAME))
    TF_PATH = os.path.abspath("model_converter/models/tf_models/from_onnx_{}".format(MODEL_NAME))
    TFLITE_PATH = os.path.abspath("model_converter/models/tflite_models/from_onnx_{}.tflite".format(MODEL_NAME))
    RESOLUTION = 224

    # MODEL_NAME = "mobilenetV2_1.0_224"
    # TF_PATH = os.path.abspath("model_converter/models/tf_models/{}".format(MODEL_NAME))
    # TFLITE_PATH = os.path.abspath("model_converter/models/tflite_models/from_onnx_{}.tflite".format(MODEL_NAME))


    # Load the deit-tiny model from torchhub
    deit_FP32 = torch.hub.load('facebookresearch/deit:main', MODEL_NAME, pretrained=True)
    deit_FP32.to(DEVICE)
                                
    # Convert the model
    # pt_to_onnx(
    #     pt_model=deit_FP32,
    #     onnx_path=ONNX_PATH,
    #     resolution=RESOLUTION
    # )
    # onnx_to_tf(
    #     onnx_path=ONNX_PATH,
    #     tf_path=TF_PATH
    # )
    # tf_to_tflite(
    #     tf_path=TF_PATH,
    #     tflite_path=TFLITE_PATH
    # )
    tf_to_quant_tflite(
        tf_path=TF_PATH,
        tflite_path=TFLITE_PATH,
        fully_quant=True,
        calibration_times=100
    )


if __name__ == "__main__":
    main()