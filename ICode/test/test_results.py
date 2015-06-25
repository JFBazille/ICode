import pickle
import matplotlib.pyplot as plt
import numpy as np
with open('/volatile/hubert/HCode/Results/resultatTest','rb') as fichier:
  monunpickler = pickle.Unpickler(fichier)
  donnees = monunpickler.load()

#with open('/volatile/hubert/HCode/Results/resultatTest','wb') as fichier:
  #monpickler = pickle.Pickler(fichier)
  #monpickler.dump(donnees)
  
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#i =0
#j=0
#f, myplots = plt.subplots(2,4)
#for cle,valeur in donnees.items():
  
  #myplots[i,j].boxplot(valeur.T, labels=np.arange(1,10)/10.)

  #myplots[i,j].set_title('Estimation of Hurst coeffician of fGn by\n'+cle+'method')
  
  #j = j+1
  #if j ==4:
    #j=0
    #i=i+1


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

i=0
j=0
k=0

f, myplots = plt.subplots(1,4,sharey=True)
for cle,valeur in donnees.items():
  if '514' in cle:
    bp = myplots[j].boxplot(valeur.T, labels=np.arange(1,10)/10.)
    myplots[j].set_title('Estimation of Hurst\ncoeffician of fGn by\n'+cle+'method')
    
    for line in bp['medians']:
      # get position data for median line
      x, y = line.get_xydata()[1] # top of median line
      # overlay median value
      if(k <6):
	myplots[j].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
					       np.var(valeur[k,:])),
	horizontalalignment='center') # draw above, centered
      else:
	myplots[j].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
					      np.var(valeur[k,:])),
	    horizontalalignment='center') # draw above, centered
      k = k+1
    k=0
    j = j+1

     
f, myplots = plt.subplots(1,4,sharey=True)
for cle,valeur in donnees.items():

  if '4096' in cle:
    bp = myplots[i].boxplot(valeur.T, labels=np.arange(1,10)/10.)
    myplots[i].set_title('Estimation of Hurst\ncoeffician of fGn by\n'+cle+'method')
    
    for line in bp['medians']:
      # get position data for median line
      x, y = line.get_xydata()[1] # top of median line
      # overlay median value
      
      if(k<6):
	myplots[i].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
					      np.var(valeur[k,:])),
		 horizontalalignment='center') # draw above, centered
      else:
	myplots[i].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
					      np.var(valeur[k,:])),
		 horizontalalignment='center') # draw above, centered
      k = k+1
    i = i+1
    k=0


 
plt.show() 
