# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2021

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
def plot2DSet(desc,label):
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    data_negatifs = desc[label == -1]
    data_positifs = desc[label == +1]
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color='red')
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color='blue')
    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])    
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    a=n/2
    desc=np.random.uniform(inf,sup,(n,p))
    lab=np.asarray([-1 for i in range (0,int(a))] + [+1 for i in range (0,int(a))])
    np.random.shuffle(lab)
    return (desc,lab)
# ------------------------ 
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    data_positif=np.random.multivariate_normal(positive_center,positive_sigma,nb_points)
    data_negatif=np.random.multivariate_normal(negative_center,negative_sigma,nb_points)
    data_desc=np.vstack((data_positif,data_negatif))
    data_label=np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])
    #np.random.shuffle(data_label)
    return (data_desc,data_label)
# ------------------------ 
def create_XOR(nb_points,var):
    data_positif=np.random.multivariate_normal([0,0],[[var,0],[0,var]], nb_points//4)
    data_positif2 =np.random.multivariate_normal([1,1],[[var,0],[0,var]], nb_points//4)
    data_negatif=np.random.multivariate_normal([1,0],[[var,0],[0,var]], nb_points//4)
    data_negatif2=np.random.multivariate_normal([0,1],[[var,0],[0,var]], nb_points//4)
    data_desc=np.vstack((data_positif, data_positif2 ,data_negatif, data_negatif2))
    data_label=np.asarray([-1 for i in range(0,nb_points//2)] + [+1 for i in range(0,nb_points//2)])
    return (data_desc,data_label)
 # ------------------------ 
