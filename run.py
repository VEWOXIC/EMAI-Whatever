from numpy.lib.function_base import sinc
from model.simple import simple
import torch
import numpy as np
#from experiment.exp import experiment
#import pandas as pd
#from experiment.exp_multi import experiment
from experiment.some_transformer import experiment
if __name__=='__main__':

    exp=experiment()
    #exp.train()
    exp.test()
