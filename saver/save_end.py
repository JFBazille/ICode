import data_class as dat
import pickle
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
  return newdata,H

