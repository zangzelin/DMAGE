import os
from multiprocessing import Process, Manager
import numpy as np
import signal
import time
from itertools import product
import subprocess

# parameter analysis for SAGloss

import tool

perplexity = [3,5,10,15,20,30,50,100, 200, 500]

cudalist = [
    0,
    1,
    2,
    # 3,
    4,
    5,
    6,
    # 7,
]

changeList = [perplexity]
paramName = ['perplexity']
mainFunc = "./main.py"
ater = tool.AutoTrainer(
    changeList,
    paramName,
    mainFunc,
    deviceList=cudalist,
    poolNumber=1*len(cudalist),
    name="autotrain",
    waittime=0.1,
)
ater.Run()
