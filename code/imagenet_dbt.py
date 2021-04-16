import numpy as np
import argparse
import time
import os

import torch
from torch import nn


parser = argparse.ArgumentParser(description="Train a model --- ImageNet")
parser.add_argument("--DataDir", type=str, default="",
                    help="imagenet dataset root")
parser.add_argument("--BatchSize", type=int, default=32,
                    help="batch size")
parser.add_argument("--TrainEpochs", type=int, default=3,
                    help="train epochs")

parser.add_argument("--lr", type=float, default=0.1,
                    help="learning rate")
parser.add_argument("--momentum", type=float, default=0.1,
                    help="momentum value for optimizer")
parser.add_argument("--WeightDecay", type=float, default=1e-4,
                    help="weight decay rate")
parser.add_argument("--DecayMode", type=str, default="step",
                    help="learning rate scheduler mode. options are step, poly and cosine")
parser.add_argument("--DecayRate", type=float, default=0.1,
                    help="decay rate of learning rate. default is 0.1")
parser.add_argument("--WarmUpLr", type=float, default=0.0,
                    help="starting warmup learning rate. default is 0.0")
parser.add_argument("--WarmUpEpochs", type=int, default=0,
                    help="number of warmup epochs")

parser.add_argument("--TrainMode", type=str,
                    help="mode in which to train the model. options are symbolic, imperative, hybrid")
parser.add_argument("--ModelName", type=str, default="DBTNet-50",
                    help="type of model to use. options are DBTNet-50, DBTNet-101")
parser.add_argument("--InputSize", type=int, default=224,
                    help="size of the input image size. default is 224")
parser.add_argument("--CropRatio", type=float, default=0.875,
                    help="crop ratio during validation. default is 0.875")
parser.add_argument("--SaveFrequency", type=int, default=10,
                    help="frequency of model saving")
parser.add_argument("--SaveDir", type=str, default="./log",
                    help="directory of saved models")

parser.add_argument("--LogFrequency", type=int, default=50,
                    help="frequency of log writing")
parser.add_argument("--LogFileName", type=str, default="train_image.log",
                    help="file name of log")
args = parser.parse_args()


class Main(object):

    def __init__(self):
        # -------------
        # params
        # -------------
        self.input_size = args.InputSize
        self.epochs = args.TrainEpochs
        self.classes = 1000


    def __call__(self):
        pass
