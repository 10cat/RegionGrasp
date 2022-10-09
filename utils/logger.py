import os
import sys
sys.path.append('.')
sys.path.append('..')
import json
import time
import datetime
import warnings
import torch
import numpy as np
from utils.utils import makepath

class Monitor:
    def __init__(self, log_folder_path):
        self.log_folder_path = log_folder_path
        makepath(self.log_folder_path)

        self.train_log = os.path.join(self.log_folder_path, "train.txt")

    def loss_message(self, LossMeters, epoch):
        message = f"Average loss for epoch {epoch}: "
        for k, v in LossMeters.items():
            message = message + k + f":{v}; "

        message = message + '\n'
        return message

    def metric_message(self, MetricMeters, epoch):
        message = f"Average metrics for epoch {epoch}: "
        for k, v in MetricMeters.items():
            message = message + k + f":{v}; "
        message = message + '\n'
        return message




        