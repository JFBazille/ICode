from nitime.analysis.spectral import SpectralAnalyzer
from nitime.timeseries import TimeSeries
#from scipy.signal import welch
import numpy as np

##NB :: il la fonction Hurstexp_Welchper est adaptee a un fBm !
def Hurstexp_Welchper(d,fs = 0.2, f_max = 0.1):
  ##data could be two dimensional(but no more...) in that cas time should be on second
  # position
  t1 = TimeSeries(data = d,sampling_rate=fs)
  s1 = SpectralAnalyzer(t1)
  
  f,pw = s1.psd
  ##We need to take only the small frequency, but the exact choice is a bit arbitrary
  # we need to have alpha between 0 and 1
  masker = np.all([(f>0),(f<f_max)],axis=0)
  tmp = np.polyfit(np.log2(f[masker]),np.log2(pw.T[masker]),deg =1)
  
  return (1-tmp[0])/2

def Hurstexp_Welchper_2(d,fs = 0.2, f_max = 0.1):
  ##data could be two dimensional(but no more...) in that cas time should be on second
  # position
  t1 = TimeSeries(data = d,sampling_rate=fs)
  s1 = SpectralAnalyzer(t1)
  
  f,pw = s1.psd
  ##We need to take only the small frequency, but the exact choice is a bit arbitrary
  # we need to have alpha between 0 and 1
  masker = np.all([(f>0),(f<f_max)],axis=0)
  tmp = np.polyfit(np.log2(f[masker]),np.log2(pw.T[masker]),deg =1)
  beta = -tmp[0]
  
  masker = beta<1
  print np.sum(masker)
  retour = np.array((beta-1)/2)
  retour[masker] = (beta[masker]+1)/2
  
  return retour


#def Hurstexp_Welchper_scipy(d,fs = 0.2, f_max = 0.1):
  ###data could be two dimensional(but no more...) in that cas time should be on second
  ## position

  #f,pw = welch(d,fs = fs)
  ###We need to take only the small frequency, but the exact choice is a bit arbitrary
  ## we need to have alpha between 0 and 1
  #masker = np.all([(f>0),(f<f_max)],axis=0)
  #tmp = np.polyfit(np.log2(f[masker]),np.log2(pw.T[masker]),deg =1)
  
  #return (1-tmp[0])/2