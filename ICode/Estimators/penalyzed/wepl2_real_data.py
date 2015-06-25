import Optimize as o
import numpy as np
def fmrd(H,aest,yij,varyj,nj, j1,j2,mask, wtype=1):
  shape = mask.shape
  j2 = min((j2,nj.shape[0]))
  j1j2 = np.arange(j1-1,j2)  #on commence par la partie en f(H)
  J = len(j1j2)
  njj=nj[j1j2]
  Hm = np.reshape(H,shape)[mask]
  aestm = np.reshape(aest,shape)[mask] 
  djh = 2*np.outer(Hm,j1j2+1) + np.outer(aestm,np.ones(len(j1j2))) 
  S = (yij[:,j1j2] - djh)**2
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
  
  S  = S.dot(wvarjj)
  return S

def Gradfmrd(H,aest,yij,varyj,nj, j1,j2,mask, wtype=1):
  shape = mask.shape
  j2 = min((j2,nj.shape[0]))
  j1j2 = np.arange(j1-1,j2)
  J = len(j1j2)
  njj=nj[j1j2]
 
  #djh pour 2jH
  Hm = np.reshape(H,shape)[mask]
  aestm = np.reshape(aest,shape)[mask] 
  djh = 2*np.outer(Hm,j1j2+1) + np.outer(aestm,np.ones(len(j1j2))) 
  
  if     wtype==0:#  % uniform weights
    wvarjj = nancynp.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = njj/ sum(njj)       
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = 1/varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  S = np.zeros(shape)
  S[mask] = -4*(yij[:,j1j2]-djh).dot((j1j2+1)*wvarjj)
  return S

def fmrdbis(Haest,yij,varyj,nj, j1,j2,mask, wtype=1, pfunc =None):
  shape = mask.shape 
  j2 = min((j2,nj.shape[0]))
  jj = np.arange(j1-1,j2)  #on commence par la partie en f(H)
  J = len(jj)
  njj=nj[jj]
  n = len(Haest)/2
  H = Haest[:n]
  aest = Haest[n:]
  Hm = np.reshape(H,shape)[mask]
  aest = np.reshape(aest,shape)[mask] 
  djh = 2*np.outer(Hm,jj+1) + np.outer(aest,np.ones(len(jj))) 
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
  
def Gradfmrdbis(Haest,yij,varyj,nj, j1,j2,mask, wtype=1,Gpfunc=None):
  shape = mask.shape
  j2 = min((j2,nj.shape[0]))
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
  djh = 2*np.outer(Hm,jj+1) + np.outer(aest,np.ones(len(jj)))

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



## This are the functions that should be tested on the real datas 
def JHmrd(H,aest,shape,yij,varyj,nj, j1,j2,mask,l=1, wtype=1):
  return fmrd(H,aest,yij,varyj,nj, j1,j2,mask, wtype)+l*np.sum((np.array(o.hgradmask(np.reshape(H,shape),mask))**2))


def GradJHmrd(H,aest,shape,yij,varyj,nj, j1,j2,mask,epsilon =1,l=1, wtype=1):
  return np.reshape(Gradfmrd(H,aest,yij,varyj,nj, j1,j2,mask, wtype),H.shape) + np.reshape(l*o.hflapdmask(np.reshape(H,shape),mask,epsilon = epsilon),H.shape)


def JHmbis(Haest,shape,yij,varyj,nj, j1,j2,mask,l=1, wtype=1):
  return fmrdbis(Haest,yij,varyj,nj, j1,j2,mask, wtype, pfunc =lambda x : l*np.sum((np.array(o.hgradmask(np.reshape(x,shape),mask))**2)))

def GradJHmbis(Haest,shape,yij,varyj,nj, j1,j2,mask,epsilon = 0,l=0, wtype=1):
  return Gradfmrdbis(Haest,yij,varyj,nj, j1,j2,mask, wtype, Gpfunc = lambda x : l*np.reshape(o.hflapdmask(np.reshape(x,shape),mask,epsilon = 0),x.shape))
