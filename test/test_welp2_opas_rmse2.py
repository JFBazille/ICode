##This function has been implemented to test wepl2
import sys
sys.path.append('/volatile/hubert/HCode/')
import scipy.io as scio
import numpy as np
from ProgressBar import ProgressBar
import matplotlib.pyplot as plt
import time
tic = time.time()
f = scio.loadmat('/volatile/hubert/datas/simulations/simulationsfGn2.mat')
from scipy.optimize import fmin_l_bfgs_b, check_grad
from Estimators import penalyzed
import opas
from Estimators.Wavelet.HDW_plagiate import *
from scipy.ndimage.filters import gaussian_filter
from math import ceil
j1=3
j2=6
wtype = 1
l=514

#s = opas.square(10)
s = opas.smiley()
signal = opas.get_simulation_from_picture(s,lsimul=l)
signalshape = signal.shape
shape = signalshape[:-1]
sig = np.reshape(signal,(signalshape[0]*signalshape[1],signalshape[2]))
N= sig.shape[0]

estimate = np.zeros(N)
aest = np.zeros(N)
simulation=np.cumsum(sig,axis =1 )

####Une facon de calculer les coefficient de Hurst plus longue mais plus precise
#estimate = np.zeros(N)
#nbvoies = int( np.log2(l) )
#nbvanishmoment  = 4 
#j2 =5
#nbvoies =min( int(np.log2(l/(2*nbvanishmoment+1)))  , nbvoies)
#Bar = ProgressBar(N, 60, 'Work in progress')
#Elog = np.zeros((N,nbvoies))
#Varlog = np.zeros((N,nbvoies))


#for j in np.arange(0,N):
  #Bar.update(j)
  #dico = wtspecq_statlog3(simulation[j], nbvanishmoment,1,np.array(2),nbvoies,0,0)
  #sortie = regrespond_det2(dico['Elogmuqj'][0],dico['Varlogmuqj'][0],2,dico['nj'],j1,j2,wtype)
  #estimate[j]=sortie['Zeta']/2. #normalement Zeta
  #Elog[j] = dico['Elogmuqj'][0]
  #Varlog[j] = dico['Varlogmuqj'][0]
  #if j==0:
    #nj = dico['nj']
  #Bar.update(j+1)

#print 'computation time   ::  ' + str(time.time() - tic)+'\n'



#######################################################################
##un Autre un peu moins precise mais tres rapide
dico = wtspecq_statlog32(simulation, 2,1,np.array(2),int( np.log2(l) ),0,0)
Elog = dico['Elogmuqj'][:,0]
Varlog = dico['Varlogmuqj'][:,0]
nj = dico['nj']

for j in np.arange(0,N):
  sortie = regrespond_det2(Elog[j],Varlog[j],2,nj,j1,j2,wtype)
  estimate[j]=sortie['Zeta']/2. #normalement Zeta
  aest[j]  = sortie['aest']
##########

lmax = 15
minimiseurs = np.zeros((lmax,)+s.shape)
cg = np.zeros(lmax)
rmse = np.zeros(lmax)
r = np.arange(lmax)
lbda = np.array((0,)+tuple(1.5**r[:-1]))


#Ici pour tester J2 et GradJ2
f = lambda x,lbda : penalyzed.J2(x,aest,Elog,Varlog,nj,j1,j2, l = lbda)
g = lambda x,lbda : penalyzed.GradJ2(x,aest,Elog,Varlog,nj,j1,j2,l=lbda)
H =estimate

fg = lambda x,lbda, **kwargs: (f(x,lbda), g(x,lbda))
fmin = lambda lbda : fmin_l_bfgs_b(lambda x :fg(x,lbda), H)
ckgrad= lambda lbda : check_grad(lambda x : f(x,lbda),lambda x : g(x,lbda),H)

for idx in r:
  monmin = fmin(lbda[idx])
  cg[idx] = ckgrad(lbda[idx])
  minimiseurs[idx] = np.reshape(monmin[0],s.shape)
  rmse[idx] =  np.sqrt(np.mean((minimiseurs[idx]-s)**2))

fig,axes = plt.subplots(nrows = 3, ncols =int(ceil(lmax/3.)) )
fig.suptitle('Minimisation J2 and GradJ2')
im = list()
for idx,(dat,ax) in enumerate(zip(minimiseurs, axes.flat)):
  im.append(ax.imshow(dat))
  ax.set_title("lambda = " + str(lbda[idx]))

cax = fig.add_axes([0.9,0.1,0.03,0.8])
fig.colorbar(im[0],cax = cax)

fig, ax1 = plt.subplots()
fig.suptitle('Minimisation J2 and GradJ2')
ax1.plot(lbda, cg)
ax1.set_xlabel('\lambda')
ax1.set_ylabel('check_gradient', color ='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b') #set the left graduation color in blue

ax2 = ax1.twinx()
ax2.plot(lbda,rmse, 'r')
ax2.set_ylabel('rmse',color = 'r')
for tl in ax2.get_yticklabels():
    tl.set_color('r') #set the left graduation color in blue


##Ici pour tester JH2 et GradJH2
f = lambda x,lbda : penalyzed.JH2(x,aest,shape,Elog,Varlog,nj,j1,j2,l=lbda)
g = lambda x,lbda : penalyzed.GradJH2(x,aest,shape,Elog,Varlog,nj,j1,j2,l=lbda)
fg = lambda x,lbda, **kwargs: (f(x,lbda), g(x,lbda))
fmin = lambda lbda : fmin_l_bfgs_b(lambda x :fg(x,lbda), H)
ckgrad= lambda lbda : check_grad(lambda x : f(x,lbda),lambda x : g(x,lbda),H)

for idx in r:
  monmin = fmin(lbda[idx])
  cg[idx] = ckgrad(lbda[idx])
  minimiseurs[idx] = np.reshape(monmin[0],s.shape)
  rmse[idx] =  np.sqrt(np.mean((minimiseurs[idx]-s)**2))

fig,axes = plt.subplots(nrows = 3, ncols =int(ceil(lmax/3.)) )
fig.suptitle('Minimisation JH2 and GradJH2')
for idx,(dat,ax) in enumerate(zip(minimiseurs, axes.flat)):
  im = ax.imshow(dat)
  ax.set_title("lambda = " + str(lbda[idx]))

cax = fig.add_axes([0.9,0.1,0.03,0.8])
fig.colorbar(im,cax = cax)

fig, ax1 = plt.subplots()
fig.suptitle('Minimisation JH2 and GradJH2')
ax1.plot(lbda, cg)
ax1.set_xlabel('\lambda')
ax1.set_ylabel('check_gradient', color ='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b') #set the left graduation color in blue

ax2 = ax1.twinx()
ax2.plot(lbda,rmse, 'r')
ax2.set_ylabel('rmse',color = 'r')
for tl in ax2.get_yticklabels():
    tl.set_color('r') #set the left graduation color in blue

  
####Ici pour tester J2bis et GradJ2bis
f = lambda x,lbda : penalyzed.J2bis(x,Elog,Varlog,nj,j1,j2,l=lbda)
g = lambda x,lbda : penalyzed.GradJ2bis(x,Elog,Varlog,nj,j1,j2,l=lbda)
H = np.concatenate((estimate,aest))

fg = lambda x,lbda, **kwargs: (f(x,lbda), g(x,lbda))
fmin = lambda lbda : fmin_l_bfgs_b(lambda x :fg(x,lbda), H)
ckgrad= lambda lbda : check_grad(lambda x : f(x,lbda),lambda x : g(x,lbda),H)

for idx in r:
  monmin = fmin(lbda[idx])
  cg[idx] = ckgrad(lbda[idx])
  minimiseurs[idx] = np.reshape(monmin[0][:monmin[0].shape[0]/2],s.shape)
  rmse[idx] =  np.sqrt(np.mean((minimiseurs[idx]-s)**2))

fig,axes = plt.subplots(nrows = 3, ncols =int(ceil(lmax/3.)) )
fig.suptitle('Minimisation J2bis and GradJ2bis')
for idx,(dat,ax) in enumerate(zip(minimiseurs, axes.flat)):
  im = ax.imshow(dat)
  ax.set_title("lambda = " + str(lbda[idx]))

cax = fig.add_axes([0.9,0.1,0.03,0.8])
fig.colorbar(im,cax = cax)

fig, ax1 = plt.subplots()
fig.suptitle('Minimisation J2bis and GradJ2bis')
ax1.plot(lbda, cg)
ax1.set_xlabel('\lambda')
ax1.set_ylabel('check_gradient', color ='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b') #set the left graduation color in blue

ax2 = ax1.twinx()
ax2.plot(lbda,rmse, 'r')
ax2.set_ylabel('rmse',color = 'r')
for tl in ax2.get_yticklabels():
    tl.set_color('r') #set the left graduation color in blue

####Ici pour tester JH2bis et GradJH2bis
f = lambda x,lbda : penalyzed.JH2bis(x,shape,Elog,Varlog,nj,j1,j2,l=lbda)
g = lambda x,lbda : penalyzed.GradJH2bis(x,shape,Elog,Varlog,nj,j1,j2,l=lbda)


fg = lambda x,lbda, **kwargs: (f(x,lbda), g(x,lbda))
fmin = lambda lbda : fmin_l_bfgs_b(lambda x :fg(x,lbda), H)
ckgrad= lambda lbda : check_grad(lambda x : f(x,lbda),lambda x : g(x,lbda),H)

for idx in r:
  monmin = fmin(lbda[idx])
  cg[idx] = ckgrad(lbda[idx])
  minimiseurs[idx] = np.reshape(monmin[0][:monmin[0].shape[0]/2],s.shape)
  rmse[idx] =  np.sqrt(np.mean((minimiseurs[idx]-s)**2))

fig,axes = plt.subplots(nrows = 3, ncols =int(ceil(lmax/3.)) )
fig.suptitle('Minimisation JH2bis and GradJH2bis')
for idx,(dat,ax) in enumerate(zip(minimiseurs, axes.flat)):
  im = ax.imshow(dat)
  ax.set_title("lambda = " + str(lbda[idx]))

cax = fig.add_axes([0.9,0.1,0.03,0.8])
fig.colorbar(im,cax = cax)

fig, ax1 = plt.subplots()
fig.suptitle('Minimisation JH2bis and GradJH2bis')
ax1.plot(lbda, cg)
ax1.set_xlabel('\lambda')
ax1.set_ylabel('check_gradient', color ='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b') #set the left graduation color in blue

ax2 = ax1.twinx()
ax2.plot(lbda,rmse, 'r')
ax2.set_ylabel('rmse',color = 'r')
for tl in ax2.get_yticklabels():
    tl.set_color('r') #set the left graduation color in blue
    
    
plt.show()