#This Code test our Hexp_Welchp function gives mean and var
#of the estimate Hurst exponent
import sys
sys.path.append('/volatile/hubert/HCode/')
import scipy.io as scio
import numpy as np
from ProgressBar import ProgressBar
import matplotlib.pyplot as plt
import Estimators.Wavelet as EW
import time
tic = time.time()
f = scio.loadmat('/volatile/hubert/datas/simulations/simulationsfGn2.mat')

simulations = f['simulations']
#number of different h
n = simulations.shape[0]
#number of simulation for a given h
N = simulations.shape[1]
#length of simulation
l = simulations.shape[2]

estimate = np.zeros((n,N))
Bar = ProgressBar(n, 60, 'Work in progress')
for i in np.arange(0,n):
  Bar.update(i)
  for j in np.arange(0,N):
    simulation=simulations[i,j,:514]
    #simulation = np.cumsum(simulations[i,:,:514], axis = 0) 
    estimate[i,j]=EW.Hurst_by_Wt4(simulation)['H']
  Bar.update(i+1)
print 'computation time   ::  ' + str(time.time() - tic)+'\n'

fig = plt.figure(100)
bp = plt.boxplot(estimate.T, labels=np.arange(1,10)/10.)
s = 'Wavelet'
#plt.title('Estimation of Hurst\ncoeffician of fBm by\n'+s+'of length'+str(l)+'method')
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
print 'computation time   ::  ' + str(time.time() - tic)+'\n'
plt.show()