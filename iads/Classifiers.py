# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2021

# Import de packages externes 
import numpy as np
import pandas as pd
import random

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
        
       
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

        
    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_correct=0
        for i in range(label_set.size):
            if (self.predict(desc_set[i])*label_set[i] > 0):
                nb_correct+=1
        return nb_correct/label_set.size

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.w=np.random.uniform(-1,1,input_dimension)
        
        
    def train(self, desc_set, label_set):
        print("Pas d'apprentissage pour ce classifieur")
        return 
    
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        score = 0
        for i in range(self.input_dimension):
            score += self.w[i]*x[i]
        return score
    
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) > 0:
            return 1
        return -1
        
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    #TODO: A Compléter
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.k = k
        self.dataset = ([],[])
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist_tab = []
        for i in range(len(self.dataset[0])):
            distance = np.sqrt(np.sum((np.array(self.dataset[0][i])-np.array(x))**2))
            dist_tab.append(distance)
        
        dSortedKLab = self.dataset[1][np.argsort(dist_tab)][:self.k]
        nbPos = np.count_nonzero(dSortedKLab == 1)
        return nbPos/self.k
        
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if (self.score(x)> 0.5): return 1 #si k est pair et nbPos = k/2 ?? 
        return -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.dataset = desc_set, label_set
        
 # ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, history = False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.history = history
        self.learning_rate = learning_rate
        self.dim = input_dimension
        self.w = np.array([random.choice([-1, 1])*learning_rate for i in range (self.dim)])
        #self.w = np.array([0]*self.dim)
        self.allw = []
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """   
        tab = ([k for k in range(label_set.size)])
        random.shuffle(tab)
        for i in tab:
            x = desc_set[i]
            y = label_set[i]
            if (self.predict(x)*y <= 0):
                if (self.history):
                    self.allw.append(self.w)
                self.w += self.learning_rate*(x*y)
    
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) > 0 else -1
 # ---------------------------
class ClassifierPerceptronKernel(Classifier):
    
    def __init__(self, input_dimension, learning_rate, noyau):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : 
                - noyau : Kernel à utiliser
            Hypothèse : input_dimension > 0
        """
        self.dim = input_dimension
        self.learning_rate = learning_rate
        self.kernel = noyau
        self.w = np.array([random.choice([-1, 1])*learning_rate for i in range (self.kernel.get_output_dim())])
        
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, self.kernel.transform(x))
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) > 0 else -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        for i in range(len(desc_set)):
            x = desc_set[i]
            y = label_set[i]
            xk = self.kernel.transform(x)
            if (self.predict(x)*y <= 0):
                self.w += self.learning_rate*(xk*y)

# ------------------------ 
    

       
