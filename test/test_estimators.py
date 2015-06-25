#This Code test our Hexp_Welchp function gives mean and var
#of the estimate Hurst exponent
import sys
sys.path.append('/volatile/hubert/HCode/')
import scipy.io as scio
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Estimators.Hexp_Welchp import Hurstexp_Welchper as HW
from Estimators.DFA import DFA
from Estimators.Whittle import Whittle
import pickle
f = scio.loadmat('/volatile/hubert/datas/simulations/simulationsfGn2.mat')

simulations = f['simulations']
#number of different h
n = simulations.shape[0]
#number of simulation for a given h
N = simulations.shape[1]
#length of simulation
l = simulations.shape[2]
#l for long s for short
dfal = np.zeros((n,N))
welchl = np.zeros((n,N))
whittlel = np.zeros((n,N))
dfas = np.zeros((n,N))
welchs = np.zeros((n,N))
whittles = np.zeros((n,N))
##estimate_s005 = np.zeros((n,N))
for i in np.arange(0,n):
  for j in np.arange(0,N):
    #dfal[i,j] = DFA(simulations[i,j,:])
    #dfas[i,j] = DFA(simulations[i,j,:514])
    whittlel = Whittle(simulations[i,j,:])
    whittles = Whittle(simulations[i,j,:514])
  #welchl[i]=HW(simulations[i,:,:])
  #welchs[i]=HW(simulations[i,:,:514])

#donnees = {'Welch_4096' : welchl,'Welch_514' : welchs,
	  #'DFA_4096' : dfal,'DFA_514' : dfas,
	  #'Whittle_4096' : whittlel,'Whittle_514' : whittles}
##whittle
donnees = {'Whittle_4096' : whittlel,'Whittle_514' : whittles}	  
###DFA
#donnees = {'DFA_4096' : dfal,'DFA_514' : dfas}

with open('/volatile/hubert/HCode/Results/resultat_test_whittle','wb') as fichier:
  monpickler = pickle.Pickler(fichier)
  monpickler.dump(donnees)
