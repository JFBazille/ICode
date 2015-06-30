#This Code test our DFA and Whittle function gives mean and var
#of the estimate Hurst exponent
import scipy.io as scio
import numpy as np
from ICode.Estimators import whittle_s
from ICode.Estimators import dfa_s

f = scio.loadmat('ICode/simulations/simulationsfGn514.mat')

simulations = f['simulations']

#We define anonymous functions that will be usefull
w = lambda i : whittle_s(simulations,i)
d = lambda i : dfa_s(simulations,i)

#dfai = np.zeros(simulations.shape[0])
whittlei = np.zeros(simulations.shape[0])
#we lauch the threads !
for i in range(simulations.shape[0]):
  #dfai[i] = d(i)
  whittlei[i] = w(i)
