from numpy.lib.function_base import sinc
from model.simple import simple
import torch
import numpy as np
#from experiment.exp import experiment
#import pandas as pd
#from experiment.exp_multi import experiment
#from experiment.some_transformer import experiment
# from experiment.some_transformer_info import experiment
# from experiment.some_transformer_info_T import experiment # cross validation 用
from experiment.seven_head import experiment

import argparse 
def parse_args():
    parser = argparse.ArgumentParser(description='emai')
    parser.add_argument('--hidden', type=int, default=5, help='hidden size of variates')
    parser.add_argument('--head', type=int, default=5, help='head size of variates')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    torch.manual_seed(114514)  # reproducible
    torch.cuda.manual_seed_all(114514)
    np.random.seed(114514)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
    exp=experiment()
    # exp.train()
    # exp.validate()
    exp.test_last7days()
    # exp.test()
    # exp.cross_validation()
