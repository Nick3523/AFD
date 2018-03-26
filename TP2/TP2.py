
# coding: utf-8

# # TP - Régression et Régularisation
# Dans ce TP vous devrez implémenter la regression logistique (en partant du code de la régression linéaire) et vous l'appliquerez sur des données décrivant des voitures (`Auto2.csv`)

# ### D'abord, chargeons les données
# 
# Les données décrivent des voitures.
# On a des variables comme le poids de la voiture, son accelération, etc...
# et on cherche à prédire sa cylindrée, qui peut être 4 cylindres (classe 0) ou plus (classe 1)

# In[10]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random


# Pour charger les données depuis un fichier csv, on utilise un module python qui s'appelle `pandas`.

# In[11]:

import pandas
autos = pandas.read_csv( "Auto2.csv")
autos.drop(labels=['name','origin'],axis=1,inplace=True)
autos = autos[autos.cylinders != 3]
autos = autos[autos.cylinders != 5]
autos.head()


# maintenant, on converti ces données en tableaux numpy:
# * `X` sera le tableau de données à 5 variables
# * la cylindrée sera stockée dans `y`

# In[12]:

X = autos[['mpg','displacement','weight','acceleration','year']].as_matrix()
y = (autos[['cylinders']].as_matrix().squeeze() >= 6).astype(int)


# In[13]:

print('X=',X,'\ny=',y[:20])


# In[141]:

# Affichons les points de X, en utilisant seulement les 3ieme et 4ieme colonnes de X.
plt.scatter(X[:,2],X[:,3],c=y)
plt.xlabel('poids')
plt.ylabel('acceleration');


# ### Sujet du TP
# 
# 
# Avant tout, on va mélanger les données avec la commande `X,y = shuffle(X,y)`. Cette commande provient du module python `scikit-learn`
# 
# #### Régression
# 
# * Lancez la fonction `reg_lineaire` (ci-dessous en annexe) avec les bons paramètres (que vous trouverez par essai-erreur). Vous lirez le code pour comprendre parfaitement ce qu'elle fait.
# * Notre problème est un problème de classification, donc la régression linéaire n'est pas adaptée. Ce qu'on cherche à faire est plutot une régression logistique. Ecrivez la fonction `reg_logistique` en modifiant la fonction `reg_lineaire`, et lancez-la et testez-la sur les données.
# * Ecrivez une fonction  `erreur_empirique(X,y,theta)` qui calcule l'erreur empirique de la regression logistique sur l'ensemble des données `X,y`
# * Ecrivez une fonction  `log_vraissemblance_empirique(X,y,theta)` qui calcule le log de la vraissemblance empirique de la regression logistique sur l'ensemble des données `X,y`
# 
# #### Régularisation
# 
# * Ajoutez une régularisation l1 à la descente de gradient. Pour rappel, la norme l1 est la somme des valeurs absolues des theta_j. Donc dans la formule du gradient, il faut ajouter lambda*signe(theta_j) pour chaque coordonnée j (ce qui correspond à la dérivée de la norme l1).
# * Pour différentes valeurs du paramètre lambda, relancez la régression logistique. Affichez en fonction de lambda le taux d'erreur en classification, et le nombre de theta_j non-nuls (en réalité, les theta_j ne sont jamais exactement égaux à zero. Donc on choisira un petit epsilon, et on comptera le nombre de theta_j donc la valeur absolue est plus grande que epsilon).
# 
# 

# ### Annexes
# 
# Fonctions qui seront utiles pour faire le TP

# In[125]:

# Pour faire la descente de gradient, on a besoin de la fonction g(z)=1/(1+exp(-z))
# mais cette fonction est "numeriquement instable", car l'exponentiel peut générer des
# valeurs hors des limites des floats.
# Donc on utilise la version suivante de g(), équivalente mais stable

def g(z):
    "Numerically stable sigmoid function."
    if z >= 0:
        ez = np.exp(-z)
        return 1 / (1 + ez)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        ez = np.exp(z)
        return ez / (1 + ez)


# In[126]:

# Voici un algorithme de base de régression lineaire appliqué aux données.
# NB: pour simplifier, le modèle prédit n'a pas de constante theta0 comme vu en cours.

def reg_lineaire(X,y,alpha):

    n,d = X.shape #n : nombre d'exemples, d : nombre de variables
    theta = np.zeros(d) #un tableau de taille d, rempli de zeros
    #print(str(n)+" "+str(d))
    
    for t in range(1000):
        i = random.randint(0,n-1)
        xi = X[i]
        yi = y[i]

        h = np.dot(theta,xi)
        #print(t,h)#,theta,xi)
        #print("est ce un tableau ? (oui) : "+str(theta))
        theta -= alpha*xi*(h-yi) #alpha : le pas d'apprentissage

    return theta



# In[127]:

X,y = shuffle(X,y)


# In[128]:

theta = reg_lineaire(X,y,0.0000001)
print(theta)


# In[129]:

np.dot(X,theta)


# In[130]:

y


# In[131]:

def reg_logistique(X,y,alpha):
    n,d = X.shape #n : nombre d'exemples, d : nombre de variables
    theta = np.zeros(d) #un tableau de taille d, rempli de zeros
    
    for t in range(1000):
        i = random.randint(0,n-1)
        xi = X[i]
        yi = y[i]

        h = np.dot(theta,xi)
        prediction = g(h)
        #print("est ce un tableau ? (oui) : "+str(theta))
        theta -= alpha*xi*(prediction-yi) #alpha : le pas d'apprentissage

    return theta
    


# In[136]:

theta = reg_logistique(X,y,0.001)
print(theta)


# In[139]:

[g(z) for z in np.dot(X,theta)]
#on a les memes valeurs dans les prédictions et dans les Y


# In[140]:

y


# In[ ]:



