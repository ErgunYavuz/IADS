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

# ------------------------




import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
def plot2DSet(desc, label):
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    data_negatifs = desc[label == -1]
    data_positifs = desc[label == +1]
    plt.scatter(data_negatifs[:, 0],
                data_negatifs[:, 1], marker='x', color='red')
    plt.scatter(data_positifs[:, 0],
                data_positifs[:, 1], marker='o', color='blue')
    plt.grid(True)


def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax = desc_set.max(0)
    mmin = desc_set.min(0)
    x1grid, x2grid = np.meshgrid(np.linspace(
        mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step))
    grid = np.hstack((x1grid.reshape(x1grid.size, 1),
                     x2grid.reshape(x2grid.size, 1)))

    # calcul de la prediction pour chaque point de la grille
    res = np.array([classifier.predict(grid[i, :]) for i in range(len(grid))])
    res = res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1

    plt.contourf(x1grid, x2grid, res, colors=[
                 "darksalmon", "skyblue"], levels=[-1000, 0, 1000])
# ------------------------


def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    a = n/2
    desc = np.random.uniform(inf, sup, (n, p))
    lab = np.asarray([-1 for i in range(0, int(a))] +
                     [+1 for i in range(0, int(a))])
    np.random.shuffle(lab)
    return (desc, lab)
# ------------------------


def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    pos = np.random.multivariate_normal(
        positive_center, positive_sigma, nb_points)
    neg = np.random.multivariate_normal(
        negative_center, negative_sigma, nb_points)
    res = np.vstack((neg, pos)), np.asarray(
        [-1 for i in range(nb_points)] + [+1 for i in range(nb_points)])
    return res
# ------------------------


def create_XOR(nb_points, var):
    data_positif = np.random.multivariate_normal(
        [0, 0], [[var, 0], [0, var]], nb_points//4)
    data_positif2 = np.random.multivariate_normal(
        [1, 1], [[var, 0], [0, var]], nb_points//4)
    data_negatif = np.random.multivariate_normal(
        [1, 0], [[var, 0], [0, var]], nb_points//4)
    data_negatif2 = np.random.multivariate_normal(
        [0, 1], [[var, 0], [0, var]], nb_points//4)
    data_desc = np.vstack((data_positif, data_positif2,
                          data_negatif, data_negatif2))
    data_label = np.asarray(
        [-1 for i in range(0, nb_points//2)] + [+1 for i in range(0, nb_points//2)])
    return (data_desc, data_label)
 # ------------------------


class Kernel():
    """ Classe pour représenter des fonctions noyau
    """

    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out

    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim

    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """
        raise NotImplementedError("Please Implement this method")


class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """

    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3
            rajoute une 3e dimension au vecteur donné
        """
        V_proj = np.append(V, np.ones((len(V), 1)), axis=1)
        return V_proj

 # ------------------------


def crossval(X, Y, n_iterations, iteration):
    # A COMPLETER
    index = np.random.permutation(len(X))  # mélange des index
    nb_elem = len(X)//n_iterations
    test = [index[i] for i in range(nb_elem)]
    app = [index[i] for i in range(nb_elem, len(X))]
    Xtest = X[index]
    Ytest = Y[index]
    Xapp = X[app]
    Yapp = Y[app]

    return Xapp, Yapp, Xtest, Ytest


def crossval_strat(X, Y, n_iterations, iteration):
    classes = np.unique(Y)
    Xc = [X[Y == c] for c in classes]
    Yc = [Y[Y == c] for c in classes]
    d = X.shape[1]
    Xapp, Yapp, Xtest, Ytest = np.array([]).reshape(0, d), np.array([]).reshape(
        0, d), np.array([]).reshape(0, d), np.array([]).reshape(0, d)
    for i in range(len(classes)):
        Xappc, Yappc, Xtestc, Ytestc = crossval(
            Xc[i], Yc[i], n_iterations, iteration)
        Xapp = np.vstack((Xapp, Xappc)) if Xapp.size else Xappc
        Yapp = np.concatenate((Yapp, Yappc)) if Yapp.size else Yappc
        Xtest = np.vstack((Xtest, Xtestc)) if Xtest.size else Xtestc
        Ytest = np.concatenate((Ytest, Ytestc)) if Ytest.size else Ytestc
    return Xapp, Yapp, Xtest, Ytest


def plot2DSetMulticlass(desc, labels):
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    val = np.unique(labels)
    data = []
    for i in val:
        data.append(desc[labels == i])
    for i in range(len(data)):
        rgb = np.random.rand(3,)
        plt.scatter(data[i][:, 0], data[i][:, 1], marker='o', color=[rgb])
        plt.grid(True)

        # calcul de C pour les différentes valeurs de w puis affichage de la courbe correspondante


def normalisation(X):
    nom = X - X.min(axis=0)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return nom/denom

# Affiche le résultat graphique de l'algorithme k-moyennes


def affiche_resultat(data, centroides, affectation):
    # On plot chaque cluster d'une couleur différente
    for i in range(len(centroides)):
        # On génère une couleur aléatoire
        r = rd.random()
        b = rd.random()
        g = rd.random()
        c = (r, b, g)

        # On récupère les points
        data_norm = data[affectation[i]]

        # On plot le cluster
        plt.scatter(data_norm[:, 0], data_norm[:, 1], color=c)

    # On plot en croix rouges les centroides
    centroides = np.array(centroides)
    plt.scatter(centroides[:, 0], centroides[:, 1], color='r', marker='x')
