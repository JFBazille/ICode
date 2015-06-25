import sys
sys.path.append('/volatile/hubert/HCode/')
import scipy.io as scio
import numpy as np
from ProgressBar import ProgressBar
import matplotlib.pyplot as plt
import Estimators.Wavelet as EW
import time
tic = time.time()
g = scio.loadmat('/volatile/hubert/datas/sortiewt')
from Estimators.Wavelet.HDW_plagiate import *

N = g['N']
Elogmuqj=g['Elogmuqj']
Varlogmuqj = g['Varlogmuqj']
nj = g['nj']
j1=2
j2=8
wtype = 1
sortie = regrespond_det2(Elogmuqj[0],Varlogmuqj[0],nj[0],j1,j2,wtype)