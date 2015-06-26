#Wavelet estimator penalisation L2 first and simpler penalization
import numpy as np
from scipy.ndimage.filters import laplace
import ICode.Optimize as o
#this function is the objective function
#J = sum(f(Hi)) + normel2(H)^2
#where f(Hi) is the norme L2 of the difference between the estimated coefficient H
#and the data
#usually yij = Elogmuqj for all i but we write yij it is simpler and shorter
#we allow different possible weight but usually we will use the nj as weigth
def f(H,aest,yij,varyj,nj, j1,j2, wtype=1):
  jj = np.arange(j1-1,j2)
  J = len(jj)
  njj=nj[jj]
  djh = 2*np.outer(H,(jj+1)) + np.outer(aest,np.ones(len(jj))) 
  S = (yij[:,jj] - djh)**2
  N= sum(njj)
  if     wtype==0:#  % uniform weights
    wvarjj = nancynp.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = njj/N        
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = 1/varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  
  #then we multiply S by the pounderation
  S = S.dot(wvarjj)
  
  return np.sum(S)

def Gradf(H,aest,yij,varyj,nj, j1,j2, wtype=1):
  j1j2 = np.arange(j1-1,j2)
  J = len(j1j2)
  njj=nj[j1j2]
  
  #djh pour 2jH
  djh = 2*np.outer(H,j1j2+1) + np.outer(aest,np.ones(len(j1j2)))
  
  if     wtype==0:#  % uniform weights
    wvarjj = nancynp.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = njj/sum(njj)       
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = 1/varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  S = -4*(yij[:,j1j2] - djh).dot((j1j2+1)*wvarjj)
  return S

def fbis(Haest,yij,varyj,nj, j1,j2, wtype=1, pfunc =None):
  jj = np.arange(j1-1,j2)  #on commence par la partie en f(H)
  J = len(jj)
  njj=nj[jj]
  n = len(Haest)
  H = Haest[:n/2]
  aest = Haest[n/2:]
  djh = 2*np.outer(H,jj+1) + np.outer(aest,np.ones(len(jj))) 
  S = (yij[:,jj] - djh)**2
  N= sum(njj)
  if     wtype==0:#  % uniform weights
    wvarjj = nancynp.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = njj/N        
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = 1/varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  
  #then we multiply S by the pounderation
  S = S.dot(wvarjj)
  return sum(S)+pfunc(H)
  
##this function is allow to compute the gradient of f bis
# and add to the H part (the first part of the gradient)
# the part of the gradient linked to the penalization
def Gradfbis(Haest,yij,varyj,nj, j1,j2, wtype=1,Gpfunc=None):
  jj = np.arange(j1-1,j2)
  J = len(jj)
  njj=nj[jj]
  N=sum(njj)
  n = len(Haest)
  H = Haest[:n/2]
  aest = Haest[n/2:]
  #djh pour 2jH
  djh = 2*np.outer(H,jj+1) + np.outer(aest,np.ones(len(jj)))

  if     wtype==0:#  % uniform weights
    wvarjj = nancynp.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = njj/N        
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = 1/varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  #on calcule la parti du gradiant en aest
  G = -2*(yij[:,jj] - djh)
  #Gaest = 0.5*(G.dot(wvarjj) + G.dot(np.ones(len(jj))))
  #Gaest = G.dot(np.ones(len(jj)))
  Gaest = G.dot(wvarjj)
  #on peut reutiliser Gaest pour le calcul pour S
  S = 2*G.dot((jj+1)*wvarjj)
  GH = S
  if not (Gpfunc is None ): 
    GH += Gpfunc(H)
  #return np.reshape((S + 2*l*H),(1,len(H))) au cas ou on veuille regarder le chack_gradient terme a terme
  return np.concatenate((GH,Gaest))

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## Same function but allow to use a mask !The mask should correspond to a convex interrior space
## The H and aest should have the same shape the whole figure (masked part + unmasked part)
## But yij, varyij and nj should have a shape compatible with the masked part (the part where mask is true !) 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
def fmask(H,aest,yij,varyj,nj, j1,j2,mask, wtype=1):
  shape = mask.shape
  H = np.reshape(H,shape)[mask]
  aest = np.reshape(aest,shape)[mask]
  jj = np.arange(j1-1,j2)
  J = len(jj)
  njj=nj[jj]
  djh = 2*np.outer(H,jj) + np.outer(aest,np.ones(len(jj))) 
  S = (yij[:,jj] - djh)**2
  N= sum(njj)
  if     wtype==0:#  % uniform weights
    wvarjj = nancynp.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = njj/N        
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = 1/varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  
  #then we multiply S by the pounderation
  S = S.dot(wvarjj)
  
  return np.sum(S)

def Gradfmask(H,aest,yij,varyj,nj, j1,j2,mask, wtype=1):
  shape = mask.shape
  l = H.shape[0]
  S = np.zeros(l)
  H = np.reshape(H,shape)[mask]
  aest = np.reshape(aest,shape)[mask]
  jj = np.arange(j1-1,j2)
  J = len(jj)
  njj=nj[jj]
  N=sum(njj)
  #djh pour 2jH
  djh = 2*np.outer(H,jj) + np.outer(aest,np.ones(len(jj)))
  
  if     wtype==0:#  % uniform weights
    wvarjj = nancynp.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = njj/N        
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = 1/varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  
  S[np.reshape(mask,l)] = -4*(yij[:,jj] - djh).dot(jj*wvarjj)
  return S

def fmaskbis(Haest,yij,varyj,nj, j1,j2,mask, wtype=1, pfunc =None):
  shape = mask.shape 
  jj = np.arange(j1-1,j2)  #on commence par la partie en f(H)
  J = len(jj)
  njj=nj[jj]
  n = len(Haest)/2
  H = Haest[:n]
  aest = Haest[n:]
  Hm = np.reshape(H,shape)[mask]
  aest = np.reshape(aest,shape)[mask] 
  djh = 2*np.outer(Hm,jj) + np.outer(aest,np.ones(len(jj))) 
  S = (yij[:,jj] - djh)**2
  N= sum(njj)
  if     wtype==0:#  % uniform weights
    wvarjj = nancynp.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = njj/N        
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = 1/varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  
  #then we multiply S by the pounderation
  S = S.dot(wvarjj)
  return sum(S)+pfunc(H)
  
def Gradfmaskbis(Haest,yij,varyj,nj, j1,j2,mask, wtype=1,Gpfunc=None):
  shape = mask.shape

  jj = np.arange(j1-1,j2)
  J = len(jj)
  njj=nj[jj]
  N=sum(njj)
  n = len(Haest)/2
  H = Haest[:n]
  aest = Haest[n:]
  Gaest = np.zeros(n)
  GH = np.zeros(n)
  mmask = np.reshape(mask,n)
  Hm = np.reshape(H,shape)[mask]
  aest = np.reshape(aest,shape)[mask]
  #djh pour 2jH
  djh = 2*np.outer(Hm,jj) + np.outer(aest,np.ones(len(jj)))

  if     wtype==0:#  % uniform weights
    wvarjj = nancynp.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = njj/N        
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = 1/varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  #on calcule la parti du gradiant en aest
  G = -2*(yij[:,jj] - djh)
  #Gaest = 0.5*(G.dot(wvarjj) + G.dot(np.ones(len(jj))))
  #Gaest = G.dot(np.ones(len(jj)))
  Gaest[mmask] = G.dot(wvarjj)
  #on peut reutiliser Gaest pour le calcul pour S
  S = 2*G.dot(jj*wvarjj)
  GH[mmask] = S
  if not (Gpfunc is None ): 
    GH += Gpfunc(H)
  #return np.reshape((S + 2*l*H),(1,len(H))) au cas ou on veuille regarder le chack_gradient terme a terme
  return np.concatenate((GH,Gaest))


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#this function is the objective function
#J = sum(f(Hi)) + normel2(H)^2
def J2(H,aest,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return f(H,aest,yij,varyj,nj, j1,j2, wtype=1)+l*sum(H**2)

def GradJ2(H,aest,yij,varyj,nj, j1,j2,l=1, wtype=1):
 return Gradf(H,aest,yij,varyj,nj, j1,j2, wtype=1) + 2*l*H



## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##



#this function is the same objective function
#but we use also aest as a variable ! and see if it yields a different result
#Haest is a 2n array, the n first term are H and the n last aest, it is as simple as that !
def J2bis(Haest,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return fbis(Haest,yij,varyj,nj, j1,j2,wtype,pfunc = lambda x : l*np.sum(x**2))

def GradJ2bis(Haest,yij,varyj,nj, j1,j2,l=0, wtype=1):
  return Gradfbis(Haest,yij,varyj,nj, j1,j2, wtype,Gpfunc =lambda x : 2*l*x)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##



## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##




#this function is the objective function
#J = sum(f(Hi)) + normel2(Gradient(H))^2
#where f(Hi) is the norme L2 of the difference between the estimated coefficient H
#and the data
#usually yij = Elogmuqj for all i but we write yij it is simpler and shorter
#we allow different possible weight but usually we will use the nj as weigth
#I call it JH2 because the norm remind me of the H2 norm
def JH2(H,aest,shape,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return f(H,aest,yij,varyj,nj, j1,j2, wtype)+l*np.sum(np.array(np.gradient(np.reshape(H,shape)))**2)

def GradJH2(H,aest,shape,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return Gradf(H,aest,yij,varyj,nj, j1,j2, wtype) - 2*l*np.reshape(laplace(np.reshape(H,shape),mode = 'reflect'),H.shape)



## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##



#this function is the same objective function
#but we use also aest as a variable ! and see if it yields a different result
#Haest is a 2n array, the n first term are H and the n last aest, it is as simple as that !
def JH2bis(Haest,shape,yij,varyj,nj, j1,j2,l=1, wtype=1):
  """
  Haest is the concatenation of H and aest,
  shape is the shape of the original image
  """
  return fbis(Haest,yij,varyj,nj, j1,j2, wtype, pfunc =lambda x : l*np.sum(np.array(np.gradient(np.reshape(x,shape)))**2)) 

def GradJH2bis(Haest,shape,yij,varyj,nj, j1,j2,l=0, wtype=1):
  return Gradfbis(Haest,yij,varyj,nj, j1,j2, wtype, Gpfunc =lambda x :-2*l*np.reshape(laplace(np.reshape(x,shape),mode = 'reflect'),x.shape) )




## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##



#this function is the objective function
#J = sum(f(Hi)) + normel2(Gradient(H))^2
#where f(Hi) is the norme L2 of the difference between the estimated coefficient H
#and the data
#usually yij = Elogmuqj for all i but we write yij it is simpler and shorter
#we allow different possible weight but usually we will use the nj as weigth
#I implement my own norm of the gradient and Laplacian
def JH22(H,aest,shape,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return f(H,aest,yij,varyj,nj, j1,j2, wtype)+l*np.sum(np.array(o.hgrad(np.reshape(H,shape)))**2)

def GradJH22(H,aest,shape,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return Gradf(H,aest,yij,varyj,nj, j1,j2, wtype) - 2*l*np.reshape(o.hlaplacian(np.reshape(H,shape)),H.shape)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


#this function is the same objective function
#but we use also aest as a variable ! and see if it yields a different result
#Haest is a 2n array, the n first term are H and the n last aest, it is as simple as that !
def JH22bis(Haest,shape,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return fbis(Haest,yij,varyj,nj, j1,j2, wtype, pfunc =lambda x :l*np.sum(np.array(o.hgrad(np.reshape(x,shape)))**2))

def GradJH22bis(Haest,shape,yij,varyj,nj, j1,j2,l=0, wtype=1):
  return Gradfbis(Haest,yij,varyj,nj, j1,j2, wtype, Gpfunc =lambda x : - 2*l*np.reshape(o.hlaplacian(np.reshape(x,shape)),x.shape))


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


def JHu(H,aest,shape,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return f(H,aest,yij,varyj,nj, j1,j2, wtype)+l*np.sum(np.array(o.hgrad(np.reshape(H,shape)))**2)

def GradJHu(H,aest,shape,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return Gradf(H,aest,yij,varyj,nj, j1,j2, wtype) + 2*l*np.reshape(o.hflap(np.reshape(H,shape)),H.shape)

def GradJHud(H,aest,shape,yij,varyj,nj, j1,j2,epsilon =1,l=1, wtype=1):
  return Gradf(H,aest,yij,varyj,nj, j1,j2, wtype) + l*np.reshape(o.hflapd(np.reshape(H,shape),epsilon = epsilon),H.shape)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##



#this function is the same objective function
#but we use also aest as a variable ! and see if it yields a different result
#Haest is a 2n array, the n first term are H and the n last aest, it is as simple as that !
def JHubis(Haest,shape,yij,varyj,nj, j1,j2,l=1, wtype=1):
  return fbis(Haest,yij,varyj,nj, j1,j2, wtype, pfunc =lambda x : l*np.sum(np.array(o.hgrad(np.reshape(x,shape)))**2))

def GradJHubis(Haest,shape,yij,varyj,nj, j1,j2,l=0, wtype=1):
  return Gradfbis(Haest,yij,varyj,nj, j1,j2, wtype, Gpfunc = lambda x : l*np.reshape(o.hflap(np.reshape(x,shape)),x.shape))

def GradJHudbis(Haest,shape,yij,varyj,nj, j1,j2,epsilon = 1,l=0, wtype=1):
  return Gradfbis(Haest,yij,varyj,nj, j1,j2, wtype, Gpfunc = lambda x : l*np.reshape(o.hflapd(np.reshape(x,shape),epsilon = epsilon),x.shape))


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


def JHumask(H,aest,shape,yij,varyj,nj, j1,j2,mask,l=1, wtype=1):
  return f(H,aest,yij,varyj,nj, j1,j2, wtype)+l*np.sum((np.array(o.hgradmask(np.reshape(H,shape),mask))**2))


def GradJHudmask(H,aest,shape,yij,varyj,nj, j1,j2,mask,epsilon =1,l=1, wtype=1):
  return Gradf(H,aest,yij,varyj,nj, j1,j2, wtype) + l*np.reshape(o.hflapdmask(np.reshape(H,shape),mask,epsilon = epsilon),H.shape)


def JHumaskbis(Haest,shape,yij,varyj,nj, j1,j2,mask,l=1, wtype=1):
  return fbis(Haest,yij,varyj,nj, j1,j2, wtype, pfunc =lambda x : l*np.sum((np.array(o.hgradmask(np.reshape(x,shape),mask))**2)))

def GradJHumaskbis(Haest,shape,yij,varyj,nj, j1,j2,mask,l=0, wtype=1):
  return Gradfbis(Haest,yij,varyj,nj, j1,j2, wtype, Gpfunc = lambda x : l*np.reshape(o.hflapdmask(np.reshape(x,shape),mask,epsilon = 0),x.shape))


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

## This are the functions that should be tested on the real datas 
def JHm(H,aest,shape,yij,varyj,nj, j1,j2,mask,l=1, wtype=1):
  return fmask(H,aest,yij,varyj,nj, j1,j2,mask, wtype)+l*np.sum((np.array(o.hgradmask(np.reshape(H,shape),mask))**2))


def GradJHm(H,aest,shape,yij,varyj,nj, j1,j2,mask,epsilon =1,l=1, wtype=1):
  return Gradfmask(H,aest,yij,varyj,nj, j1,j2,mask, wtype) + l*np.reshape(o.hflapdmask(np.reshape(H,shape),mask,epsilon = epsilon),H.shape)


def JHmbis(Haest,shape,yij,varyj,nj, j1,j2,mask,l=1, wtype=1):
  return fmaskbis(Haest,yij,varyj,nj, j1,j2,mask, wtype, pfunc =lambda x : l*np.sum((np.array(o.hgradmask(np.reshape(x,shape),mask))**2)))

def GradJHmbis(Haest,shape,yij,varyj,nj, j1,j2,mask,epsilon = 0,l=0, wtype=1):
  return Gradfmaskbis(Haest,yij,varyj,nj, j1,j2,mask, wtype, Gpfunc = lambda x : l*np.reshape(o.hflapdmask(np.reshape(x,shape),mask,epsilon = 0),x.shape))

