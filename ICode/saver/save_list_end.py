def save_list_end(donnees, fichier ='/volatile/hubert/datas/mysession2'):
  with open(fichier,'ab') as myfile:
    monpickler = pickle.Pickler(myfile)
    for d in donnees:
      monpickler.dump(d) 
  return donnees
