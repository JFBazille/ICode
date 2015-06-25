import sys
import data_class as dat
import pickle

def save_session(title =' ', masker = None, signal = None,H = None, image = None,
		 comments = ' ',ecrirealasuite = False, fichier ='/volatile/hubert/datas/mysession'):
  """This function save the computed H and other usefull datas in the
  file fichier as a list of dataClass """
  
  if ecrirealasuite:
    try:
      with open(fichier,'rb') as myfile:
	monunpickler = pickle.Unpickler(myfile)
	donnees = monunpickler.load()
      
    except:
      sys.stderr.write('no such file or directory a new one will be created')
      donnees = list()
  else:
    donnees = list()

  newdata = dat.DataClass() 
  newdata.update(title,masker,signal,H,image,
		 comments)
  donnees.append(newdata)
  
  
  with open(fichier,'wb') as myfile:
    monpickler = pickle.Pickler(myfile)
    monpickler.dump(donnees)
  return donnees


def save_session_liste(liste, ecrirealasuite = True, fichier = '/volatile/hubert/datas/mysession'):
  
  if ecrirealasuite:
    try:
      with open(fichier,'rb') as myfile:
	monunpickler = pickle.Unpickler(myfile)
	donnees = monunpickler.load()
      
    except:
      sys.stderr.write('no such file or directory a new one will be created')
      donnees = list()
  for i in liste:    
    donnees.append(i)
  
  
  with open(fichier,'wb') as myfile:
    monpickler = pickle.Pickler(myfile)
    monpickler.dump(donnees)
  return donnees
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## previous function save in one single list of data_class following
## one save a series of data_class object
def save_end(title =' ', masker = None, signal = None,H = None, image = None,
		 comments = ' ', fichier ='/volatile/hubert/datas/mysession2'):
  """This function save data at the end of a file"""


  newdata = dat.DataClass() 
  newdata.update(title,masker,signal,H,image,
		 comments)

  
  
  with open(fichier,'ab') as myfile:
    monpickler = pickle.Pickler(myfile)
    monpickler.dump(newdata)
  return newdata

def save_list_end(donnees, fichier ='/volatile/hubert/datas/mysession2'):
  with open(fichier,'ab') as myfile:
    monpickler = pickle.Pickler(myfile)
    for d in donnees:
      monpickler.dump(d) 
  return donnees

