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
import copy

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

    def score(self, x):
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
        cpt = 0
        for i in range(label_set.size):
            if(self.predict(desc_set[i]) == label_set[i]):
                cpt = cpt+1
        return (cpt/label_set.size)

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
        self.input_dimension = input_dimension
        self.w = np.random.uniform(-1, 1, input_dimension)

    def train(self, desc_set, label_set):
        print("Pas d'apprentissage pour ce classifieur")
        return

    def score(self, x):
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

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.k = k

    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist_tab = []
        for i in range(len(self.desc_set)):
            distance = np.sqrt(
                np.sum((np.array(self.desc_set[i])-np.array(x))**2))
            dist_tab.append(distance)
        tab_tri = np.argsort(dist_tab)
        kneighbors = []
        for i in range(self.k):
            kneighbors.append(self.label_set[tab_tri[i]])
        cpt = 0
        for i in kneighbors:
            if (i > 0):
                cpt = cpt+1
        return cpt/self.k

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if (self.score(x) > 0):
            return 1  # si k est pair et nbPos = k/2 ??
        return -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set

 # ---------------------------


class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """

    def __init__(self, input_dimension, learning_rate, history=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.history = history
        self.learning_rate = learning_rate
        self.dim = input_dimension
        self.w = np.array(
            [random.choice([-1, 1])*learning_rate for i in range(self.dim)])
        # self.w = np.array([0]*self.dim)
        self.allw = []

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # raise NotImplementedError("Please Implement this method")
        taille = np.array([i for i in range(len(desc_set))])
        np.random.shuffle(taille)
        for i in taille:
            x = desc_set[i]
            y = label_set[i]
            if self.score(x) * y < 1:
                if self.history:
                    self.allw.append(self.w)
                self.w = self.w + self.learning_rate*x*y

    def getW(self):
        """ rend le vecteur de poids actuel du perceptron
        """
        return self.w

    def score(self, x):
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
        self.noyau = noyau
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.w = [0 for _ in range(noyau.get_output_dim())]

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        x = self.noyau.transform(np.array([x]))[0]
        res = 0
        for i in range(len(self.w)):
            res += x[i] * self.w[i]
        return res

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if (self.score(x) < 0):
            return -1
        return 1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        desc_set = self.noyau.transform(desc_set)
        sur_place = 0
        n = len(desc_set)
        cpt = 0
        while (sur_place < n and cpt < 10 * n):
            i = random.randrange(n)
            xi = desc_set[i]
            yi = label_set[i]
            if (self.score(xi) * yi > 0):
                sur_place += 1
            else:
                sur_place = 0
                for k in range(len(self.w)):
                    self.w[k] += self.learning_rate * xi[k] * yi
            cpt += 1

    def copy(self):
        return ClassifierPerceptronKernel(self.input_dimension, self.learning_rate, self.noyau)

# ------------------------


class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """

    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.epsilon = learning_rate
        self.dim = input_dimension
        self.w = np.zeros(input_dimension)
        self.history = history
        self.niter_max = niter_max
        self.allw = []
        self.allw.append(np.copy(self.w))

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        for n in range(self.niter_max):
            i = random.randint(0, len(desc_set)-1)
            self.w -= self.epsilon * \
                (desc_set[i].T * (desc_set[i]*self.w - label_set[i]))
            if self.history:
                self.allw.append(np.copy(self.w))

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return x@self.w

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        res = self.score(x)
        if res < 0:
            return -1
        else:
            return +1


# ------------------------

class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE
    """

    def __init__(self, input_dimension, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(input_dimension)
        self.dim = input_dimension
        self.history = history
        self.niter_max = niter_max
        self.allw = []
        self.allw.append(np.copy(self.w))

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        Xt = desc_set.T
        self.w = np.linalg.solve(Xt@desc_set, Xt@label_set)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return x@self.w

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        res = self.score(x)
        if res < 0:
            return -1
        else:
            return +1


# ------------------------
 # code de la classe pour le classifieur MultiOAA
class ClassifierMultiOAA(Classifier):
    """
    Classifier multi classes générique
    """

    def __init__(self, clbin):
        self.clbin = clbin
        self.class_list = []
        self.label = {}

    def train(self, desc_set, label_set):
        i = 0
        for l in label_set:
            if l not in self.label:
                self.label[l] = i
                i += 1
                self.class_list.append(copy.deepcopy(self.clbin))

        for cl in self.label:
            ytmp = [1 if l == cl else -1 for l in label_set]
            self.class_list[self.label[cl]].train(desc_set, ytmp)

    def score(self, x):
        scores = []
        for c in self.class_list:
            scores.append(c.score(x))
        return scores

    def predict(self, x):
        index = np.argmax(self.score(x))
        for l in self.label:
            if self.label[l] == index:
                return l

    def accuracy(self, desc_set, label_set):
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()


# ------------------------
