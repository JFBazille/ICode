import os
import numpy as np
import scipy.io as scio
from math import ceil
from ICode.opas.scenarios import create_AR_noise

__all__ = ["base_dir", "get_simulation", "get_simulation_from_picture", "square", "square2", "smiley"]


def base_dir():
    """ base_dir
    """
    return os.path.dirname(__file__)


def get_simulation(fichier =  'simulationsfGn2.mat'):
    fichier = os.path.join(base_dir(), fichier)
    f =  scio.loadmat(fichier)
    return f['simulations']


def get_simulation_from_picture(picture = None, lsimul= 4096):
  fichier = os.path.join(base_dir(), 'simulationsfGn2.mat')
  f = scio.loadmat(fichier)
  simulations = f['simulations']
  n = simulations.shape[1]
  if(picture is None):
    return simulations
  
  s = picture.shape  
  retour = np.zeros(s[0:]+(lsimul,))
  if np.max(picture)>0.9:
    picture = 0.9*(picture/np.max(picture))
  for i in np.arange(0,9):
    mask = np.all(((picture>=i/10.),(picture<(i+1.)/10.)), axis =0)
    summask = np.sum(mask)
    if summask<n:
      retour[mask] = simulations[i,:np.sum(mask),:lsimul]
    else :
      lj= ceil(1.*summask/n)
      ar = np.arange(0,summask)
      r= np.ones(retour[mask].shape)
      for j in np.arange(0,lj):
	l = (j+1)*n
	if j==lj-1:
	  l = summask
	overmask = np.all(((ar>=j*n),(ar<l)),axis = 0)
	r[overmask] = simulations[i,:(l-j*n),:lsimul]
      retour[mask]=r
	
  return retour

def add_parterns(simulation,*functions,**kwargs):
  """cette fonction ajoute un paterns functions
      chaque fonction de la liste de fonction doit prendre un seul parametre
      la taille des simulations
      a une simulation 
  """
  s = (reduce(lambda a,b : a*b , (1,) + simulation.shape[:-1]))
  retour = np.reshape(np.array(simulation),(s,simulation.shape[-1]))
  for fnc in functions : 
    retour = retour + func(s,retour.shape[-1])
  return np.reshape(retour, simulation.shape)

def add_AR_noise(simulation, v_noise, order=2, v_corr=0.1, transpose=True):
    if transpose:
        simulation = simulation.T

    simulation += create_AR_noise(simulation.shape, v_noise, order, v_corr)
    
    if transpose:
        return simulation.T
    
    return simulation

def get_cumsum_AR_noised_simulation(lsimul=4096, v_noise=0.5, order=2, v_corr=0.1):
    simulations = get_simulation()

    s = (reduce(lambda a,b : a*b , (1,) + simulations.shape[:-1]))
    simulations = np.reshape(np.array(simulations),(s,simulations.shape[-1]))[:,:lsimul]
    simulations = np.cumsum(simulations, axis=-1)

    return add_AR_noise(simulations, v_noise, order, v_corr)

def get_cumsum_AR_noised_simulation_from_picture(picture=None,lsimul=4096, v_noise=0.5, order=2, v_corr=0.1):
    if picture is None:
        raise ValueError("'picture' is a None value")

    simulations = get_simulation_from_picture(picture, lsimul)
    s = (reduce(lambda a,b : a*b , (1,) + simulations.shape[:-1]))
    simulations = np.reshape(np.array(simulations),(s,simulations.shape[-1]))
    simulations = np.cumsum(simulations, axis=-1)

    return add_AR_noise(simulations, v_noise, order, v_corr)



def square(l=10):
  square = 0.5*np.ones((2*l,2*l))
  square[0:l,0:l] = 0.8*np.ones((l,l))
  square[l:,l:] = 0.2*np.ones((l,l))
  
  return square

def square2(l=10):
  square = 0.5*np.ones((2*l,2*l))
  square[0:l,l:] = 0.7*np.ones((l,l))
  square[0:l,0:l] = 0.8*np.ones((l,l))
  square[l:,l:] = 0.6*np.ones((l,l))
  
  return square

def smiley(l=10):
  #le fond
  smiley = 0.1*np.ones((4*l+1,4*l+1))
  #le visage
  dist = np.outer(np.linspace(-1,1,4*l+1)**2,np.ones(4*l+1)) + np.outer(np.linspace(-1,1,4*l+1)**2,np.ones(4*l+1)).T 
  mask = dist<1
  smiley[mask] = 0.5*np.ones(np.sum(mask))
  #le nez
  mask = dist< 0.05
  smiley[mask] = 0.3*np.ones(np.sum(mask))
  #les yeux
  dist = np.outer(np.linspace(-1,3,4*l+1)**2,np.ones(4*l+1)) + np.outer(np.linspace(-1,3,4*l+1)**2,np.ones(4*l+1)).T
  mask = dist < 0.2
  smiley[mask] = 0.8*np.ones(np.sum(mask))
  mask = np.fliplr(mask)
  smiley[mask] = 0.8*np.ones(np.sum(mask))
  #le sourire
  u = np.outer(np.linspace(-1.4,4.,4*l+1),np.ones(4*l+1)).T
  dist = np.outer(np.linspace(-0.5,3.5,4*l+1)**2,np.ones(4*l+1)) + u**2
  mask = np.all(((np.fliplr(dist).T + np.fliplr(np.fliplr(dist).T) < 8),(np.fliplr(dist).T + np.fliplr(np.fliplr(dist).T) > 6),np.fliplr(u).T<0),axis =0)
  smiley[mask] = 0.8*np.ones(np.sum(mask))
  return smiley
  