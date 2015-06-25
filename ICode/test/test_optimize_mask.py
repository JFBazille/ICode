import sys
sys.path.append('/volatile/hubert/HCode/')
import opas
from Estimators import penalyzed as pen
import numpy as np
import Optimize
import matplotlib.pyplot as plt
from Estimators.Wavelet.HDW_plagiate import *
l=4096

j1=3
j2=6
wtype = 1
#s = opas.square(10)
s = opas.smiley()
mask = s>0.1
signal = opas.get_simulation_from_picture(s,lsimul=l)
signalshape = signal.shape
sig = np.reshape(signal,(signalshape[0]*signalshape[1],signalshape[2]))
N= sig.shape[0]

estimate = np.zeros(N)
aest = np.zeros(N)
simulation=np.cumsum(sig,axis =1 )
dico = wtspecq_statlog32(simulation, 2,1,np.array(2),int( np.log2(l) ),0,0)
Elog = dico['Elogmuqj'][:,0]
Varlog = dico['Varlogmuqj'][:,0]
nj = dico['nj']

for j in np.arange(0,N):
  sortie = regrespond_det2(Elog[j],Varlog[j],2,nj,j1,j2,wtype)
  estimate[j]=sortie['Zeta']/2. #normalement Zeta
  aest[j]  = sortie['aest']

retour = np.reshape(estimate,(signalshape[0],signalshape[1]))
plt.imshow(retour,cmap = "Greys_r")
plt.show()
