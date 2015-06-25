# -*- coding: utf-8 -*-
"""
My first code to extract MATLAB simulation

Created on Wed April 15 

@author: hubert.pelle@polytechnique.edu
"""
import numpy as n
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def extractSimulation(fs = 'fichier', pathSimul = '/volatile/hubert/datas'):
  simul =  os.path.join(pathSimul, fs)
  datas = np.loadtxt(simul)
  return datas