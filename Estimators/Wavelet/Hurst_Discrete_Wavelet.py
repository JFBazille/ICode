###This function doesn't work nut still it can't be kept and fixed later

import numpy as np
import pywt
import math
from scipy.special import gammainc
def Hurst_DWt(data,j1=1,j2=8):
  q=2
  c = pywt.wavedec(data,'db5', level =j2 )
  lenc = len(c)
  S = np.zeros(lenc)
  for i in np.arange(1,lenc):
    j = lenc - i
    S[j] = np.mean(c[i][lenc/10. :2:]**2)
  
  #la regression
  tmp = np.polyfit(np.log2(S)[j1:],np.arange(0,j2+1)[j1:],deg =1)
  H = -0.5*tmp[0]
  #on prie pour que tout marche
  return {'H':H, 'alpha' : tmp[0], 'S':S, 'c' :c}

def Hurst_DWt2(data,j1=1,j2=8):
  q=2
  c = pywt.wavedec(data,'db5', level =j2 )
  lenc = len(c)
  S = np.zeros(lenc)
  nj = np.zeros(lenc)
  Varlogmuqj = np.zeros(lenc)
  Elogmuqj = np.zeros(lenc)
  loge = np.log2(np.exp(1)) #faisons comme si tout etait normal
  for i in np.arange(1,lenc):
    j = lenc - i
    S[j] = np.mean(c[i][lenc/10. :2:]**2)
    nj[j] = len(c[i][lenc/10. :2:])
    V  = np.sqrt(np.pi)*math.gamma(q+1/2)/math.gamma((q+1)/2)**2 - 1
    Varlogmuqj[j] =  (loge**2) * V /nj[j] #il s'agit d'une variance "theorique"
    gj              = -loge/2 * V /nj[j]
    Elogmuqj[j] = S[j] - gj 
  
  #la regression
  r = regrespond_det2(Elogmuqj,Varlogmuqj,nj,j1,j2,wtype=0)
  
  #on prie pour que tout marche
  return {'slope':r['Zeta'], 'Varslope' : r['Vzeta'],'Q' : r['Q'], 'S':S, 'c' :c, 'nj': nj}



#Input:     q:  the order of the statistic used (here only used in the plot title)
#%             N:  the number of vanishing moments of the wavelet ((here only used in the plot title)
#%             nj: the vector[1..scalemax] of the actual number of coefficients averaged at octave j. (these
#%                 numbers are not powers of two because of the border effects at each scale). 
#%             yj: the vector[1..scalemax] of log( sum |d(j)|^q ) - g(j) :   -  the bias term has already been
#%                 removed prior to calling.
#%             varyj: The variances of yj,  precalculated. 
#%             wtype: The kind of weighting used
#%                     0 -  no weigthing  (ie uniform weights)
#%                     1 -  1/nj weights  (suitable for fully Gaussian data)
#%                     2 -  use variance estimates varj
#%               
#%             j1:   the lower limit of the scales chosen,  1<= j1 <= scalemax-1  
#%             j2:   the upper limit of the octaves chosen, 2<= j2 <= scalemax
#%       printout:  0:  batch mode, no output
#%                  1:  printout plut log-log plot and regression line and +- 2*std(yj) around each point
#%           
#%  Output:    zetaest:   estimate of the spectral parameter  zeta
#%               Vzeta:   Exact variance of zetaest assuming given variances are correct 
#%                   Q:   A goodness of fit measure,  Chi2 based probability that null hyp is true given the
#%                        observed data over the scale range  (j1,j2)
def regrespond_det2(yj,varyj,nj,j1,j2,wtype):
  q = 2

  scalemax = len(yj)
  j=np.arange(0,scalemax)
  
  #--- Check and clean up the inputted j1 and j2
  j1 = max(1,j1)                    # make sure j1 is not too small
  j2 = max(j1+1,min(j2,scalemax))   # make sure j2 is not chosen too large
   
  #--- Convenience variables in the chosen scaling range
  jj = np.arange(j1,j2+1)
  J  = len(jj)
  njj  = nj[jj]
  yjj  = yj[jj]
  varyjj= varyj[jj]
  #%%%% Regression,  VERSION DETERMINISTE
  #%%%% -- Calculate the weighted linear regression over  j1..j2

  #%%% Notes
  #%  wtype = 1:  Here use     varyj ~ C/nj  to create the weights, rather than use the varyj, to avoid
  #%  poor varyj estimation from generating bad weights.   This captures the essence of the weights,
  #%  and the result is independant of C.  Thus no need to tell this function which "version" 
  #%  is being used (see wtspecq_statlog for more details)

  #%% define the effective varjj's used, depending on weight type
  if     wtype==0:#  % uniform weights
    wvarjj = np.ones(J)   
    wstr = 'Uniform'
  elif wtype==1:#  % Gaussian type weights
    wvarjj = 1./njj        
    wstr = 'Gaussian'
  elif  wtype==2:#   % weights from data 
    wvarjj = varyjj
    wstr = 'Estimated'
  else :#% all other cases
    print '** Weight option not recognised, using uniform weights\n'
    wvarjj = np.ones(1,J) 
    wstr = 'Uniform'
  #% use weighted regression formula in all cases
  S0 = sum(1./wvarjj) 
  S1 = sum(jj/wvarjj)
  S2 = sum(jj**2./wvarjj)
  wjj = (S0 * jj - S1) / wvarjj / (S0*S2-S1*S1)
  vjj = (S2 - S1 * jj) / wvarjj / (S0*S2-S1*S1)


  #%  Estimate  zeta
  zetaest  = sum(wjj * yjj ) #       % zeta is just the slope, unbiased regardless of the weights
  aest     = sum(vjj * yjj ) #       % intercept  'a'

 # Calculation of the variance of zetahat
  Vzeta = sum(varyjj*wjj*wjj) #       % this is exact if the varyjj are


  #%%%%--- Goodness of fit, based on inputted variances
  #%   If at least 3 points, Apply a Chi-2 test , no level chosen, should be viewed as a function of j1

  if J>2:
    J2 = (J-2)/2. 
    X  = sum( ((yjj - zetaest  * jj - aest )**2)/ varyjj )
    Q  = 1  - gammainc(X /2,J2) 
  else :
    print '\n***** Cannot calculate Q, need at least 3 points.\n'
    Q=0
  
  return {'Zeta':zetaest,'Vzeta': Vzeta, 'Q': Q}