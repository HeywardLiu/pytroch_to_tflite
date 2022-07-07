import glob
import os

import torch
import onnx
import onnx_tf.backend 
import tensorflow as tf
import numpy as np


# Configs
DEIT_MODELS = ('deit_tiny_patch16_224',
               'deit_tiny_distilled_patch16_224',
               'deit_base_patch16_384',
               'deit_base_distilled_patch16_384')
MODEL_NAME = DEIT_MODELS[1]
RESOLUTION = 224
ROOT_PATH = os.path.join(os.path.abspath(os.pardir), "deit_proj")
ONNX_PATH = os.path.join(os.path.abspath(os.pardir), "deit_proj", "model_converter", "models", "onnx_models", "from_onnx_%s.onnx" %MODEL_NAME)
TF_PATH = os.path.abspath(os.path.join(os.path.abspath(os.pardir), "deit_proj", "model_converter", "models", "tf_models", "from_onnx_%s" %MODEL_NAME))
TFLITE_PATH = os.path.abspath(os.path.join(os.path.abspath(os.pardir), "deit_proj", "model_converter", "models", "tflite_models", "from_onnx_%s.tflite" %MODEL_NAME))


def pt_to_onnx(pt_model:torch.nn.Module, onnx_path, resolution=224):
    print("\n\nConverting the model from pytorch to onnx ...")
    dummy_input = torch.randn((1, 3, resolution, resolution))  # (N, C, H, W)
    torch.onnx.export(model = pt_model,
                      args = dummy_input,    # Give a random input to trace the structure              
                      f = onnx_path,         # path to export
                      verbose = True)        # print infos of converted layers on terminal


def onnx_to_tf(onnx_path, tf_path):
    print("\n\nConverting the model from onnx to tensorflow ...")

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
    print("\n\nConverting the model from tensorflow to tflite ...")
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS     # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    
    # Save the model.
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)


def main():
    # Load the pytorch model from torchhub
    deit_FP32 = torch.hub.load('facebookresearch/deit:main',
                                MODEL_NAME,
                                pretrained = True)
    deit_INT8 = torch.quantization.quantize_dynamic(deit_FP32, {torch.nn.Linear}, dtype=torch.qint8)
    
    # Convert the model
    pt_to_onnx(pt_model=deit_FP32, onnx_path = ONNX_PATH, resolution=RESOLUTION)
    onnx_to_tf(onnx_path = ONNX_PATH, tf_path = TF_PATH)
    tf_to_tflite(tf_path=TF_PATH, tflite_path=TFLITE_PATH)


if __name__ == "__main__":
    main()