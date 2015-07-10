import numpy as np
from ICode.optimize.objective_functions import gradient


def tv(img):
    """This function compute the total variation of an image
    """
    spatial_grad = gradient(img)
    return np.sum(np.sqrt(np.sum(spatial_grad[:-1] * spatial_grad[:-1],
                                    axis=0)))
    

def hgrad(H,renorm=1.):
  e = 0.5 /renorm
  shp= H.shape
  n = len(shp)
  x = np.arange(n)
  Y = tuple(np.roll(x,-1))#pour rouler la matrice aufur et a mesure
  i = 0
  Grad = list()
  while i<n:
    Gx = np.zeros(shp)#on g ici le gradient selon chaque dimension
    Gx[1:-1] =  e*(H[2:]-H[:-2])
    Gx[0] = (H[1]-H[0])/renorm
    Gx[-1] =  (H[-1]-H[-2])/renorm
    Grad.append(np.transpose(Gx,np.roll(x,i-n)))
    H = np.transpose(H,Y)
    i+=1
  return Grad

def hlaplacian(H,epsilon=1):
  e = 1 /epsilon**2
  shp= H.shape
  n = len(shp)
  x = np.arange(n)
  Y = tuple(np.roll(x,-1))#pour rouler la matrice aufur et a mesure
  i = 0
  lap = np.zeros(shp)
  while i<n:
    Gx = np.zeros(shp)#on g ici le gradient selon chaque dimension
    Gx[1:-1] =  e*(H[2:]-2*H[1:-1]+H[:-2])
    Gx[0] = e*(H[1]-H[0])#on applique une condition au bord du type 'reflection'
    Gx[-1] =  e*(H[-2]-H[-1])
    lap +=np.transpose(Gx,np.roll(x,i-n))
    H = np.transpose(H,Y)
    #H = np.rollaxis(H,-1) #on change l'axes selon lequelle on calcul le gradient
    i+=1
  
  return lap


#This function is an approximation of the laplacian 'with two steps'
# ie DH = 0.5* (2H[i] -H[i-2] - H[i+2])
# usefull two compute the gradient of the norm of the gradient !!
def hflap(H,epsilon=1):
  e = 1 /epsilon**2
  shp= H.shape
  n = len(shp)
  x = np.arange(n)
  Y = tuple(np.roll(x,-1))#pour rouler la matrice aufur et a mesure
  i = 0
  lap = np.zeros(shp)
  while i<n:
    Gx = np.zeros(shp)#on g ici le gradient selon chaque dimension
    Gx[2:-2] = 0.5*e*(-H[:-4]+2*H[2:-2]-H[4:])
    Gx[0] = -2*e*(H[1]-H[0])-0.5*e*(H[2] - H[0])#on applique une condition au bord du type 'reflection'
    Gx[1] = 2*e*(H[1]-H[0]) - 0.5 *e*(H[3]-H[1])  
    Gx[-1] = 2*e*(H[-1]-H[-2])+ 0.5*e*(H[-3] - H[-1])
    Gx[-2] = -2*e*(H[-1]-H[-2]) + 0.5*e*(H[-2]-H[-4])
    lap +=np.transpose(Gx,np.roll(x,i-n))
    H = np.transpose(H,Y)
    #H = np.rollaxis(H,-1) #on change l'axes selon lequelle on calcul le gradient
    i+=1
  
  return lap

#This function is an approximation of the laplacian 'with two steps'
# ie DH = 0.5* (2H[i] -H[i-2] - H[i+2])
# usefull two compute the gradient of the norm of the gradient !!
def hflapd(H,epsilon=1.,renorm = 1.):
  e =epsilon
  rn = 1 /renorm**2
  shp= H.shape
  n = len(shp)
  x = np.arange(n)
  Y = tuple(np.roll(x,-1))#pour rouler la matrice aufur et a mesure
  i = 0
  lap = np.zeros(shp)
  while i<n:
    Gx = np.zeros(shp)#on g ici le gradient selon chaque dimension
    Gx[2:-2] = 0.5*rn*(-H[:-4]+2*H[2:-2]-H[4:]) +e/2.
    Gx[0] = -2*rn*(H[1]-H[0])-0.5*rn*(H[2] - H[0])+1.25*e#on applique une condition au bord du type 'reflection'
    Gx[1] = 2*rn*(H[1]-H[0]) - 0.5 *rn*(H[3]-H[1])  +1.25*e
    #Gx[-1] = 0
    Gx[-1] = 2*rn*(H[-1]-H[-2]) + 0.5*rn*(H[-3] - H[-1])+1.25*e
    Gx[-2] = -2*rn*(H[-1]-H[-2]) + 0.5*rn*(H[-2]-H[-4]) +1.25*e
    lap +=np.transpose(Gx,np.roll(x,i-n))
    H = np.transpose(H,Y)
    #H = np.rollaxis(H,-1) #on change l'axes selon lequelle on calcul le gradient
    i+=1
  
  return lap

def hgradmask(H,mask,renorm=1):
  e = 0.5 /renorm
  shp= H.shape
  n = len(shp)
  x = np.arange(n)
  Y = tuple(np.roll(x,-1))#pour rouler la matrice aufur et a mesure
  i = 0
  gmask = np.gradient(mask.astype(int))
  Grad = list()
  while i<n:
    gmaski = np.transpose(gmask[i],tuple(np.roll(x,-i)))
    mask0 = np.all(((gmaski>0),np.transpose(mask,tuple(np.roll(x,-i)))),axis=0)
    mask1 = np.roll(np.all(((gmaski>0),np.transpose(mask,tuple(np.roll(x,-i)))),axis =0),1,axis=0)
    maskm1 = np.all((gmaski<0,np.transpose(mask,tuple(np.roll(x,-i)))),axis=0)
    maskm2 = np.roll(np.all(((gmaski>0),np.transpose(mask,tuple(np.roll(x,-i)))),axis =0),-1,axis=0)
    Gx = np.zeros(H.shape)#on g ici le gradient selon chaque dimension
    Gx[1:-1] =  e*(H[2:]-H[:-2])
    Gx[0] =  e*(H[1]-H[0])
    Gx[-1] =  e*(H[-1]-H[-2])
    Gx[mask0] = e*(H[mask1]-H[mask0])#on applique une condition au bord du type 'reflection'
    Gx[maskm1] =  e*(H[maskm1]-H[maskm2])
    Grad.append(np.transpose(Gx*np.transpose(mask,tuple(np.roll(x,-i))),np.roll(x,i-n)))
    H = np.transpose(H,Y)
    #H = np.rollaxis(H,-1) #on change l'axes selon lequelle on calcul le gradient
    i+=1
  return Grad

def hflapdmask(H,mask,epsilon=1.,renorm = 1):
  e =epsilon
  rn = 1 /renorm**2
  shp= H.shape
  n = len(shp)
  x = np.arange(n)
  gmask = np.gradient(mask.astype(int))
  Y = tuple(np.roll(x,-1))#pour rouler la matrice aufur et a mesure
  i = 0
  lap = np.zeros(shp)
  while i<n:
    gmaski = np.transpose(gmask[i],tuple(np.roll(x,-i)))
    mask0 = np.all(((gmaski>0),np.transpose(mask,tuple(np.roll(x,-i)))),axis=0)
    mask1 = np.roll(np.all(((gmaski>0),np.transpose(mask,tuple(np.roll(x,-i)))),axis =0),1,axis=0)
    mask2 = np.roll(np.all(((gmaski>0),np.transpose(mask,tuple(np.roll(x,-i)))),axis =0),2,axis=0)
    mask3 = np.roll(np.all(((gmaski>0),np.transpose(mask,tuple(np.roll(x,-i)))),axis =0),3,axis=0)
    maskm1 = np.all((gmaski<0,np.transpose(mask,tuple(np.roll(x,-i)))),axis=0)
    maskm2 = np.roll(np.all((gmaski<0,np.transpose(mask,tuple(np.roll(x,-i)))),axis=0),-1,axis=0)
    maskm3 = np.roll(np.all((gmaski<0,np.transpose(mask,tuple(np.roll(x,-i)))),axis=0),-2,axis=0)
    maskm4 = np.roll(np.all((gmaski<0,np.transpose(mask,tuple(np.roll(x,-i)))),axis=0),-3,axis=0)
    Gx = np.zeros(H.shape)#on stock ici le laplacien selon chaque dimension
    Gx[2:-2] = 0.5*rn*(-H[:-4]+2*H[2:-2]-H[4:]) +e/2.#la formule du laplacien avec pas de 2
    #notre formule appliquee uniquement au bord
    Gx[mask0] = -0.5*rn*(H[mask1]-H[mask0])-0.5*rn*(H[mask2] - H[mask0])+0.5*e#on applique une condition au bord du type 'reflection'
    Gx[mask1] = 0.5*rn*(H[mask1]-H[mask0]) - 0.5 *rn*(H[mask3]-H[mask1])  +.5*e
    #Gx[-1] = 0
    Gx[maskm1] = 0.5*rn*(H[maskm1]-H[maskm2]) + 0.5*rn*(H[maskm3] - H[maskm1])+0.5*e
    Gx[maskm2] = -0.5*rn*(H[maskm1]-H[maskm2]) + 0.5*rn*(H[maskm2]-H[maskm4]) +0.5*e
    lap +=np.transpose(Gx*np.transpose(mask,tuple(np.roll(x,-i))),np.roll(x,i-n))
    H = np.transpose(H,Y)
    #H = np.rollaxis(H,-1) #on change l'axes selon lequelle on calcul le gradient
    del mask0
    del mask1
    del mask2
    del mask3
    del maskm1
    del maskm2
    del maskm3
    del maskm4
    i+=1
  
  return lap