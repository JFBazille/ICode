import numpy as np 
##This function allow to reduce the mask at maximum to keep as little False Value
##as possible while keeping True values
def reducedMask(mask):
  shp = mask.shape
  n = len(shp)
  i = 0
  
  while i >n:
    
    
    i+=1
  
  return r

def hgradmaskrd(H,mask,renorm=1):
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
    mask0 = np.all(((gmaski>0),mask),axis=0)
    mask1 = np.roll(np.all(((gmaski>0),mask),axis =0),1,axis=0)
    maskm1 = np.all((gmaski<0,mask),axis=0)
    maskm2 = np.roll(np.all(((gmaski>0),mask),axis =0),-1,axis=0)
    Gx = np.zeros(shp)#on g ici le gradient selon chaque dimension
    Gx[1:-1] =  e*(H[2:]-H[:-2])
    Gx[0] =  e*(H[1]-H[0])
    Gx[-1] =  e*(H[-1]-H[-2])
    Gx[mask0] = e*(H[mask1]-H[mask0])#on applique une condition au bord du type 'reflection'
    Gx[maskm1] =  e*(H[maskm1]-H[maskm2])
    Grad.append(np.transpose(Gx*mask,np.roll(x,i-n)))
    H = np.transpose(H,Y)
    #H = np.rollaxis(H,-1) #on change l'axes selon lequelle on calcul le gradient
    i+=1
  return Grad

def hflapdmaskrd(H,mask,epsilon=1.,renorm = 1):
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
    gmaski = np.transpose(gmask[0],tuple(np.roll(x,-i)))
    del gmask[0]
    mask0 = np.all(((gmaski>0),mask),axis=0)
    mask1 = np.roll(np.all(((gmaski>0),mask),axis =0),1,axis=0)
    mask2 = np.roll(np.all(((gmaski>0),mask),axis =0),2,axis=0)
    mask3 = np.roll(np.all(((gmaski>0),mask),axis =0),3,axis=0)
    maskm1 = np.all((gmaski<0,mask),axis=0)
    maskm2 = np.roll(np.all((gmaski<0,mask),axis=0),-1,axis=0)
    maskm3 = np.roll(np.all((gmaski<0,mask),axis=0),-2,axis=0)
    maskm4 = np.roll(np.all((gmaski<0,mask),axis=0),-3,axis=0)
    Gx = np.zeros(shp)#on stock ici le laplacien selon chaque dimension
    Gx[2:-2] = 0.5*rn*(-H[:-4]+2*H[2:-2]-H[4:]) +e/2.#la formule du laplacien avec pas de 2
    Gx[0] = -0.5*rn*(H[1]-H[0])-0.5*rn*(H[2] - H[0])+0.5*e
    Gx[1] = 0.5*rn*(H[1]-H[0])-0.5*rn*(H[3] - H[1])+0.5*e
    Gx[-1] = 0.5*rn*(H[-1]-H[-2]) + 0.5*rn*(H[-3] - H[-1])+0.5*e
    Gx[-2] = -0.5*rn*(H[-1]-H[-2]) + 0.5*rn*(H[-2]-H[-4]) +0.5*e
    #notre formule appliquee uniquement au bord
    Gx[mask0] = -0.5*rn*(H[mask1]-H[mask0])-0.5*rn*(H[mask2] - H[mask0])+0.5*e#on applique une condition au bord du type 'reflection'
    Gx[mask1] = 0.5*rn*(H[mask1]-H[mask0]) - 0.5 *rn*(H[mask3]-H[mask1])  +.5*e
    #Gx[-1] = 0
    Gx[maskm1] = 0.5*rn*(H[maskm1]-H[maskm2]) + 0.5*rn*(H[maskm3] - H[maskm1])+0.5*e
    Gx[maskm2] = -0.5*rn*(H[maskm1]-H[maskm2]) + 0.5*rn*(H[maskm2]-H[maskm4]) +0.5*e
    lap +=np.transpose(Gx*mask,np.roll(x,i-n))
    H = np.transpose(H,Y)
    #H = np.rollaxis(H,-1) #on change l'axes selon lequelle on calcul le gradient
    i+=1
  
  return lap