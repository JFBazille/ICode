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

##Ici pour tester J2bis et GradJ2bis
#f = lambda x : penalyzed.J2bis(x,Elog,Varlog,nj,j1,j2,l=0)
#g = lambda x : penalyzed.GradJ2bis(x,Elog,Varlog,nj,j1,j2,l=0)
#H = np.concatenate((estimate,aest))

##Ici pour tester JH2 et GradJH2
f = lambda x : penalyzed.JH2(x,aest,Elog,Varlog,nj,j1,j2,l=10)
g = lambda x : penalyzed.GradJH2(x,aest,Elog,Varlog,nj,j1,j2,l=10)
H = estimate

##Ici pour tester JH2bis et GradJH2bis
#f = lambda x : penalyzed.JH2bis(x,shape,Elog,Varlog,nj,j1,j2,l=1)
#g = lambda x : penalyzed.GradJH2bis(x,shape,Elog,Varlog,nj,j1,j2,l=1)
#H = np.concatenate((estimate,aest))


##Si min juste sur esimate
#cg= check_grad(f,g,estimate)
##Si bis, min sur estilmate et sur aest
cg= check_grad(f,g,H)
#On fait 1000 copie de H et on ajoute des petites variations
#H = np.outer(estimate, np.ones(len(estimate))) + np.random.rand(len(estimate),len(estimate))
#map1 = map(lambda x : check_grad(f, g, x), H)
fg = lambda x, **kwargs: (f(x), g(x))

##Si min juste sur esimate
monmin = fmin_l_bfgs_b(fg, estimate)
minimiseur  = monmin[0]
##Si bis, min sur estilmate et sur aest
#monmin = fmin_l_bfgs_b(fg, H)
#minimiseur  = monmin[0][:monmin[0].shape[0]/2]



plt.figure(1)
retour = np.reshape(minimiseur,(signalshape[0],signalshape[1]))
plt.imshow(retour,cmap = "Greys_r")
plt.figure(2)
reshape_estimate = np.reshape(estimate,(signalshape[0],signalshape[1]))
plt.imshow(reshape_estimate, cmap = "Greys_r")
plt.show()


plt.figure(1)
plt.title('Smothed computed Hurst exponent with Gaussian kernel')
plt.imshow(gaussian_filter(reshape_estimate,1))
plt.colorbar()
plt.figure(2)
plt.title('computed Hurst exponent')
plt.imshow(reshape_estimate)
plt.figure(3)
plt.title('Hurst exponent after JH2 minisation')
plt.imshow(retour)
plt.figure(4)
plt.title('Original Image')
plt.imshow(s)
plt.show()