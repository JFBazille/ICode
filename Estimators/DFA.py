import numpy as np
import matplotlib.pyplot as plt
#import nitime.timeseries as ts
import  pylab,math
from scipy import signal


def DFA(data, CS =1, j1 =2,j2=8,fignummer =0):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    """
      
    #first we should use numpy cumsum to sum up the data array
    if CS ==1:
      Y = np.cumsum(data)
    else:
      Y=data
    N = len(Y)
    #then we take a M the default M is 3
    b2=j2#from Estimators.Hexp_Welchp import Hurstexp_Welchper_scipy as HWs
    if j2> np.log2(N/2):
        b2=int(np.log2(N/4))
    
    scales = np.arange(j1,b2+1)
    scales = 2**scales
    n = N/scales
 
    F = np.zeros(len(scales))
    
    for i in np.arange(0,len(scales)):
      for j in np.arange(0,n[i]):
	data0 = signal.detrend(Y[j*scales[i]:(j+1)*scales[i]])
	F[i] = F[i] + np.var(data0)**(1/2.)
      F[i] = F[i] / n[i]
      ##???????????? doit-on considerer la variance ou l'ecart-type ? il y a 
      ## un Iatus entre la litterature et la function matlab HDFAEstim !
      ## J'ai choisi d'etre coherent avec la fonction matlab HDFAEstim...
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg = 1)
    alpha = tmp[0]
    if fignummer >0:
      qq = np.polyval(tmp, np.log2(scales))
      plt.figure(fignummer)
      plt.plot(np.log2(scales), qq, color = "blue")
      plt.plot(np.log2(scales),np.log2(F),color = "red")
      plt.xlabel("log(L)")
      plt.ylabel("log(F(L)**2)")
      plt.title("DFA\nestimated alpha :" + str(alpha))
      plt.legend()
      plt.show()
    return alpha
  
def DFAt(data,pointeur,idx,CS =1, j1 =2,j2=8):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    """
      
    #first we should use numpy cumsum to sum up the data array
    if CS ==1:
      Y = np.cumsum(data)
    else:
      Y=data
    N = len(Y)
    #then we take a M the default M is 3
    b2=j2
    if j2> np.log2(N/2):
        b2=int(np.log2(N/4))
    
    scales = np.arange(j1,b2+1)
    scales = 2**scales
    n = N/scales
 
    F = np.zeros(len(scales))
    
    for i in np.arange(0,len(scales)):
      for j in np.arange(0,n[i]):
	data0 = signal.detrend(Y[j*scales[i]:(j+1)*scales[i]])
	F[i] = F[i] + np.var(data0)**(1/2.)
      F[i] = F[i] / n[i]
      ##???????????? doit-on considerer la variance ou l'ecart-type ? il y a 
      ## un Iatus entre la litterature et la function matlab HDFAEstim !
      ## J'ai choisi d'etre coherent avec la fonction matlab HDFAEstim...
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg = 1)
    ##l'idee ici est de donnee a l'adresse entree en argument la valeur de retour
    #la fonction fonctionne en trichant un peu avec les pointeurs et les references
    pointeur[idx]= tmp[0]
    
def DFANormt(data,pointeur,idx,CS =1, j1 =2,j2=8):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    """
    x = (data-np.mean(data))/np.std(data)  
    #first we should use numpy cumsum to sum up the data array
    if CS ==1:
      Y = np.cumsum(data)
    else:
      Y=data
    N = len(Y)
    #then we take a M the default M is 3
    b2=j2
    if j2> np.log2(N/2):
        b2=int(np.log2(N/4))
    
    scales = np.arange(j1,b2+1)
    scales = 2**scales
    n = N/scales
 
    F = np.zeros(len(scales))
    
    for i in np.arange(0,len(scales)):
      for j in np.arange(0,n[i]):
	data0 = signal.detrend(Y[j*scales[i]:(j+1)*scales[i]])
	F[i] = F[i] + np.var(data0)**(1/2.)
      F[i] = F[i] / n[i]
      ##???????????? doit-on considerer la variance ou l'ecart-type ? il y a 
      ## un Iatus entre la litterature et la function matlab HDFAEstim !
      ## J'ai choisi d'etre coherent avec la fonction matlab HDFAEstim...
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg = 1)
    ##l'idee ici est de donnee a l'adresse entree en argument la valeur de retour
    #la fonction fonctionne en trichant un peu avec les pointeurs et les references
    pointeur[idx]= tmp[0]
    
def DFAS(data,idx,CS =1, j1 =2,j2=8):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    """
    x = data[idx,:]
    #first we should use numpy cumsum to sum up the data array
    if CS ==1:
      Y = np.cumsum(x)
    else:
      Y=x
    N = len(Y)
    #then we take a M the default M is 3
    b2=j2
    if j2> np.log2(N/2):
        b2=int(np.log2(N/4))
    
    scales = np.arange(j1,b2+1)
    scales = 2**scales
    n = N/scales
 
    F = np.zeros(len(scales))
    
    for i in np.arange(0,len(scales)):
      for j in np.arange(0,n[i]):
	data0 = signal.detrend(Y[j*scales[i]:(j+1)*scales[i]])
	F[i] = F[i] + np.var(data0)**(1/2.)
      F[i] = F[i] / n[i]
      ##???????????? doit-on considerer la variance ou l'ecart-type ? il y a 
      ## un Iatus entre la litterature et la function matlab HDFAEstim !
      ## J'ai choisi d'etre coherent avec la fonction matlab HDFAEstim...
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg = 1)
    ##l'idee ici est de donnee a l'adresse entree en argument la valeur de retour
    #la fonction fonctionne en trichant un peu avec les pointeurs et les references
    return tmp[0]
  
def DFANormS(data,idx,CS =1, j1 =2,j2=8):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    """
    datanorm = (data- np.mean(data))/np.std(data)
    x = datanorm[idx,:]
    #first we should use numpy cumsum to sum up the data array
    if CS ==1:
      Y = np.cumsum(x)
    else:
      Y=x
    N = len(Y)
    #then we take a M the default M is 3
    b2=j2
    if j2> np.log2(N/2):
        b2=int(np.log2(N/4))
    
    scales = np.arange(j1,b2+1)
    scales = 2**scales
    n = N/scales
 
    F = np.zeros(len(scales))
    
    for i in np.arange(0,len(scales)):
      for j in np.arange(0,n[i]):
	data0 = signal.detrend(Y[j*scales[i]:(j+1)*scales[i]])
	F[i] = F[i] + np.var(data0)**(1/2.)
      F[i] = F[i] / n[i]
      ##???????????? doit-on considerer la variance ou l'ecart-type ? il y a 
      ## un Iatus entre la litterature et la function matlab HDFAEstim !
      ## J'ai choisi d'etre coherent avec la fonction matlab HDFAEstim...
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg = 1)
    ##l'idee ici est de donnee a l'adresse entree en argument la valeur de retour
    #la fonction fonctionne en trichant un peu avec les pointeurs et les references
    return tmp[0]
  
def main(R = 20, N= 128) :
    print "********** \n Nous allons tester notre fonction \n "
    # print "sin(x) avec x entre 0 et 1000 echatilonnee a 1Hz \n"
    # data = np.sin(np.arange(0,1000))
    # estimate_alpha(data,0,titre ="sin 1Hz")
    # print "sin(x) avec x entre 0 et 1000 echatilonnee a 2Hz \n"
    # data = np.sin(np.arange(0,2000)/2)
    # estimate_alpha(data,1, titre = "sin 2Hz")
    
    # print "Un bruit blanc de longueur 1000 \n"
    # data=np.random.randn(1000)
    # estimate_alpha(data,2, titre = "Bruit Blanc")

    print "Un Mouvement Brownien fractal"

    np.random.seed(2)
    
    H=3./4.
    HH=2*H
    covariance = np.zeros((N,N))
    A=np.zeros((N,N))


    for i in range(N):
        for j in range(N):
            x=abs(i-j)
            covariance[i,j]=(abs(x - 1)**HH + (x + 1)**HH - 2*x**HH)/2.
    w,v=np.linalg.eig(covariance)
    
    for i in range(N):
        for j in range(N):
            A[i,j]=sum(math.sqrt(w[k])*v[i,k]*v[j,k] for k in range(N))
    xi=np.random.randn((N))
    eta=np.dot(A,xi)
    xfBm=[sum(eta[0:i]) for i in range(len(eta+1))]


    estimate_alpha(xfBm,3,D=R,titre = "DFA pour un Mouvement Brownien Fractal H = " +str(H))
    
