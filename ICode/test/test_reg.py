##This file needs /volatile/hubert/datas/sortiewt file you don"t necesseraly have !
## It is the out put of the Matlab function wtspecq_statlog3

import scipy.io as scio
import numpy as np
from ICode.ProgressBar import ProgressBar
import matplotlib.pyplot as plt
import ICode.Estimators.Wavelet as EW
import time
tic = time.time()
g = scio.loadmat('/volatile/hubert/datas/sortiewt')
from ICode.Estimators.Wavelet.HDW_plagiate import *

N = g['N']
Elogmuqj=g['Elogmuqj']
Varlogmuqj = g['Varlogmuqj']
nj = g['nj']
j1=2
j2=8
wtype = 1
sortie = regrespond_det2(Elogmuqj[0],Varlogmuqj[0],nj[0],j1,j2,wtype)