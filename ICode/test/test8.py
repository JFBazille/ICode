import ICode.data_class as dat
import pickle

fichier ='/volatile/hubert/datas/mysession'
sortie ='/volatile/hubert/datas/mysession2'
with open(fichier,'rb') as myfile:
  monunpickler = pickle.Unpickler(myfile)
  donnees = monunpickler.load()
idx=0
with open(sortie,'wb') as myfile:
  monpickler = pickle.Pickler(myfile)
  for idx,d in enumerate(d):
      if idx==0:
	monpickler.dump(d)
      else:
	data = dat.DataClass()
	data.update(d.title, None, None,d.H,None,
		d.comments)
	monpickler.dump(data)
