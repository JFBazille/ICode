import scipy.io as scio
import numpy as np
from ICode.ProgressBar import ProgressBar
import matplotlib.pyplot as plt
import ICode.Estimators.Wavelet as EW
import time
tic = time.time()
from ICode.Estimators.Wavelet.HDW_plagiate import *
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
#l = simulations.shape[2]
l =514
estimate = np.zeros((4,n,N))
Bar = ProgressBar(n, 60, 'Work in progress')
for i in np.arange(0,n):
  Bar.update(i)
  for j in np.arange(0,N):
    simulation=np.cumsum(simulations[i,j,:l])
    for VM in np.arange(0,4):
      dico = wtspecq_statlog32(simulation,(VM)+1,1,np.array(2),int( np.log2(l) ),0,0)
      sortie = regrespond_det2(dico['Elogmuqj'][0],dico['Varlogmuqj'][0],2,dico['nj'],j1,j2,wtype)
      estimate[VM,i,j]=sortie['Zeta']/2.
  Bar.update(i+1)
print 'computation time   ::  ' + str(time.time() - tic)+'\n'


f, myplots = plt.subplots(1,4,sharey=True)
f.suptitle('Estimation of Hurst\ncoefficient of fGn\nof length '+str(l)+' by Wavelet method, different regularities')
for N in np.arange(0,4):
  bp = myplots[N].boxplot(estimate[N,:,:].T, labels=np.arange(1,10)/10.)
  myplots[N].set_title('regularity :: ' + str((N)+1))
  k=0
  for line in bp['medians']:
    # get position data for median line
    x, y = line.get_xydata()[1]# top of median line overlay median value
    
    if(k <5):
      myplots[N].text(x+2, y-0.02, '%.3f\n%.3e' % (np.mean(estimate[N,k,:]),
					      np.var(estimate[N,k,:])),
      horizontalalignment='center') # draw above, centered
    else:
      myplots[N].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(estimate[N,k,:]),
					    np.var(estimate[N,k,:])),
	  horizontalalignment='center') # draw above, centered
    k = k+1
  k=0



plt.show()