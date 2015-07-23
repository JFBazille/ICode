import scipy.io as scio
import numpy as np
from ICode.progressbar import ProgressBar
import matplotlib.pyplot as plt
import time
tic = time.time()
from ICode.estimators.wavelet.hdwpab import *
from ICode.opas import get_simulation

simulations = get_simulation()
j1=2
j2=8
wtype = 1
#number of different h
n = simulations.shape[0]
#number of simulation for a given h
N = simulations.shape[1]
#length of simulations
l = simulations.shape[2]

estimate = np.zeros((n,N))
Bar = ProgressBar(n, 60, 'Work in progress')
for i in np.arange(0,n):
  Bar.update(i)
  for j in np.arange(0,N):
    simulation=np.cumsum(simulations[i,j,:])
    dico = wtspecq_statlog3(simulation, 7,1,np.array(2),int( np.log2(4096) ),0,0)
    sortie = regrespond_det2(dico['Elogmuqj'][0],dico['Varlogmuqj'][0],2,dico['nj'],j1,j2,wtype)
    #dico = wtspecq_statlog32(np.reshape(simulation,(1,4096)), 2,1,np.array(2),int( np.log2(4096) ),0,0)
    #sortie = regrespond_det2(dico['Elogmuqj'][0,0,:],dico['Varlogmuqj'][0,0,:],2,dico['nj'],j1,j2,wtype)
    
    estimate[i,j]=sortie['Zeta']/2. #normalement Zeta
  Bar.update(i+1)
print 'computation time   ::  ' + str(time.time() - tic)+'\n'

fig = plt.figure(100)
bp = plt.boxplot(estimate.T)
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