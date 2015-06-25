
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
    mask0 = np.all(((gmaski>0),mask),axis=0)
    mask1 = np.roll(np.all(((gmaski>0),mask),axis =0),1,axis =0)
    mask2 = np.roll(np.all(((gmaski>0),mask),axis =0),2,axis =0)
    mask3 = np.roll(np.all(((gmaski>0),mask),axis =0),3,axis =0)
    maskm1 = np.all((gmaski<0,mask),axis=0)
    maskm2 = np.roll(np.all((gmaski<0,mask),axis=0),-1,axis =0)
    maskm3 = np.roll(np.all((gmaski<0,mask),axis=0),-2,axis =0)
    maskm4 = np.roll(np.all((gmaski<0,mask),axis=0),-3,axis =0)
    #plt.imshow(H*np.any((mask0,mask1,mask2,mask3,maskm1,maskm2,maskm3,maskm4),axis=0))
    Gx = np.zeros(shp)#on stock ici le laplacien selon chaque dimension
    Gx[2:-2] = 0.5*rn*(-H[:-4]+2*H[2:-2]-H[4:]) +e/2.#la formule du laplacien avec pas de 2
    #notre formule appliquee uniquement au bord
    Gx[mask0] = -2*rn*(H[mask1]-H[mask0])-0.5*rn*(H[mask2] - H[mask0])+1.25*e#on applique une condition au bord du type 'reflection'
    Gx[mask1] = 2*rn*(H[mask1]-H[mask0]) - 0.5 *rn*(H[mask3]-H[mask1])  +1.25*e
    #Gx[-1] = 0
    Gx[maskm1] = 2*rn*(H[maskm1]-H[maskm2]) + 0.5*rn*(H[maskm3] - H[maskm1])+1.25*e
    Gx[maskm2] = -2*rn*(H[maskm1]-H[maskm2]) + 0.5*rn*(H[maskm2]-H[maskm4]) +1.25*e
    lap +=np.transpose((Gx*mask),np.roll(x,i-n))
    H = np.transpose(H,Y)
    #H = np.rollaxis(H,-1) #on change l'axes selon lequelle on calcul le gradient
    #plt.show()
    i+=1
  
  return lap