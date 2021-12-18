import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils import read_split_data, MyDataSet, train_one_epoch, evaluate
from Swin_model import swin_tiny_patch4_window7_224 as create_model



