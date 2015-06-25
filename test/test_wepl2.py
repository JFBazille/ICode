##This function has been implemented to test wepl2
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
from Estimators.Wavelet.HDW_plagiate import *
from scipy.optimize import fmin_l_bfgs_b, check_grad
from Estimators import penalyzed

j1=3
j2=6
wtype = 1
 

simulations = f['simulations']
#number of different h
n = simulations.shape[0]
#number of simulation for a given h
N = simulations.shape[1]
#length of simulations
l = simulations.shape[2]

estimate = np.zeros(N)
aest = np.zeros(N)
aest2 = np.zeros(N)
aest3 = np.zeros(N)
aest4 = np.zeros(N)
simulation=np.cumsum(simulations[7,:,:],axis =1 )
dico = wtspecq_statlog32(simulation, 2,1,np.array(2),int( np.log2(4096) ),0,0)
Elog = dico['Elogmuqj'][:,0]
Varlog = dico['Varlogmuqj'][:,0]
nj = dico['nj']

for j in np.arange(0,N):
  sortie = regrespond_det2(Elog[j],Varlog[j],2,nj,j1,j2,wtype)
  estimate[j]=sortie['Zeta']/2. #normalement Zeta
  aest[j]  = sortie['aest']
  #aest2[j]  = sortie['aest2']
  #aest3[j]  = sortie['aest3']

#f = lambda x : penalyzed.J2(x,aest,Elog,Varlog,nj,j1,j2)
#g = lambda x : penalyzed.GradJ2(x,aest,Elog,Varlog,nj,j1,j2)


#cg= check_grad(f,g,estimate)
##On fait 1000 copie de H et on ajoute des petites variations
##H = np.outer(estimate, np.ones(len(estimate))) + np.random.rand(len(estimate),len(estimate))
##map1 = map(lambda x : check_grad(f, g, x), H)
#fg = lambda x, **kwargs: (f(x), g(x))
#monmin = fmin_l_bfgs_b(fg, estimate)

##un petit plot pour se donner une idee de ce qu'il se passe 
#jj=np.arange(1,10)
#plt.plot(jj,jj*estimate[1]+aest[1])
#plt.plot(jj,Elog[1,:])
#plt.show()

###Ici pour tester J2bis et GradJ2bis
fbis = lambda x : penalyzed.J2bis(x,Elog,Varlog,nj,j1,j2,l=0)
gbis = lambda x : penalyzed.GradJ2bis(x,Elog,Varlog,nj,j1,j2,l=0)


#cg= check_grad(fbis,gbis,np.concatenate((estimate,aest)))
##On fait 1000 copie de H et on ajoute des petites variations
##H = np.outer(estimate, np.ones(len(estimate))) + np.random.rand(len(estimate),len(estimate))
##map1 = map(lambda x : check_grad(f, g, x), H)
fgbis = lambda x, **kwargs: (fbis(x), gbis(x))
monmin = fmin_l_bfgs_b(fgbis, np.random.rand(2*len(estimate)))

#un petit plot pour se donner une idee de ce qu'il se passe 
jj=np.arange(1,10)
plt.figure(1)
#plt.plot(jj,jj*2*estimate[1]+aest2[1],'k--')
#plt.plot(jj,jj*2*estimate[1]+aest3[1],'k:')
plt.plot(jj,jj*2*estimate[1]+aest[1],'b')
plt.plot(jj,jj*2*estimate[1]+aest[1],'bo')
plt.plot(jj,Elog[1,:],'green')
plt.plot(jj,jj*2*monmin[0][1] + monmin[0][0.5*len(monmin[0]) + 1],'r')
plt.plot(jj,jj*2*monmin[0][1] + monmin[0][0.5*len(monmin[0]) + 1],'r^')
plt.xlabel('scale j')
plt.ylabel('E log S_j')
plt.figure(2)

#plt.plot(jj,jj*2*np.mean(estimate,axis =0)+np.mean(aest2,axis=0),'k--')
#plt.plot(jj,jj*2*np.mean(estimate,axis =0)+np.mean(aest3,axis=0),'k:')
plt.plot(jj,jj*2*np.mean(estimate,axis =0)+np.mean(aest,axis =0),'b-o')
plt.plot(jj,jj*2*np.mean(estimate,axis =0)+np.mean(aest,axis =0),'b')
plt.plot(jj,np.mean(Elog,axis=0),'green')

plt.plot(jj,jj*2*np.mean(monmin[0][:0.5*len(monmin[0])],axis =0)  + np.mean(monmin[0][0.5*len(monmin[0]) + 1 :],axis =0),'r^')
plt.plot(jj,jj*2*np.mean(monmin[0][:0.5*len(monmin[0])],axis =0)  + np.mean(monmin[0][0.5*len(monmin[0]) + 1 :],axis =0),'r')
plt.xlabel('scale j')
plt.ylabel('E log S_j')
plt.show()
