#This Code test our Hexp_Welchp function gives mean and var
#of the estimate Hurst exponent
import sys
sys.path.append('/volatile/hubert/HCode/')
import scipy.io as scio
import numpy as np

import matplotlib.pyplot as plt
from Estimators.Hexp_Welchp import Hurstexp_Welchper_2 as HW

f = scio.loadmat('/volatile/hubert/datas/simulations/simulationsfGn2.mat')

simulations = f['simulations']
#number of different h
n = simulations.shape[0]
#number of simulation for a given h
N = simulations.shape[1]
#length of simulation
l = simulations.shape[2]

estimate = np.zeros((n,N))

for i in np.arange(0,n):
  ##NB :: il la fonction Hurstexp_Welchper est adaptee a un fBm !
  #       d'ou l'utilisation du cumsum
  simulation = np.cumsum(simulations[i,:,:514], axis = 0) 
  estimate[i]=HW(simulation)



fig = plt.figure(100)
bp = plt.boxplot(estimate.T, labels=np.arange(1,10)/10.)
s = 'Welch_per_cumsum'
plt.title('Estimation of Hurst\ncoeffician of fGn by\n'+s+'of length'+str(simulation.shape[1])+'method')
k=0
for line in bp['medians']:
  # get position data for median line
  x, y = line.get_xydata()[1] # top of median line overlay median value
  if(k <5):
    plt.text(x+0.4, y-0.02, '%.3f\n%.3e' % (np.mean(estimate[k,:]),
					    np.var(estimate[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    plt.text(x-0.8, y-0.02, '%.3f\n%.3e' % (np.mean(estimate[k,:]),
					  np.var(estimate[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
  
plt.show()