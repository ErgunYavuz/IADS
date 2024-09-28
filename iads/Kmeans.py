import numpy as np
import pandas as pd
import random

# code de la du K-means


def kmoyennes(k, base, epsilon, iter_max=100, affiche=False):
    cent = initialisation(k, base, afficheInit=affiche)
    for i in range(iter_max):
        affect = affecte_cluster(base, cent)
        old_inert = inertie_globale(base, affect)
        cent = nouveaux_centroides(base, affect)
        affect = affecte_cluster(base, cent)
        new_inert = inertie_globale(base, affect)
        if (affiche == True):
            print("iteration ", i, "Inertie :", old_inert,
                  "Difference: ", abs(new_inert - old_inert))
        if (abs(new_inert - old_inert) < epsilon):
            break
    return cent, affect, old_inert


def dist_vect(v1, v2):
    somme = 0
    for i in range(len(v1)):
        somme += (v2[i]-v1[i])**2
    return np.sqrt(somme)


def centroide(data):
    return data.mean(axis=0)


def inertie_cluster(array):
    cent = centroide(array)
    inertie_ensembe = 0
    for i in array:
        inertie_i = dist_vect(cent, i)**2
        # print("centre: ",cent,"\tExemple: ",i,"\tDistance = ",inertie_i)
        inertie_ensembe += inertie_i
    return inertie_ensembe


def initialisation(k, base, afficheInit=False):
    """hypothese : k > 1"""
    selec = random.sample(base.tolist(), k)
    if afficheInit == True:
        print("Sélectionnés: ", selec)
    return np.asarray(selec)


def plus_proche(x, centroides):
    closest = centroides[0]
    i_closest = 0
    for i in range(1, len(centroides)):
        if(dist_vect(x, centroides[i]) < dist_vect(x, centroides[i_closest])):
            closest = centroides[i]
            i_closest = i
    return i_closest


def affecte_cluster(base, k):
    dic = {i: [] for i in range(len(k))}
    for i in range(len(base)):
        pproche = plus_proche(base[i, :], k)
        dic[pproche].append(i)
    return dic


def nouveaux_centroides(base, affect):
    newCent = []
    for i in affect:
        mean = sum(base[affect[i]])/len(affect[i])
        newCent.append(mean)
    return np.asarray(newCent)


def inertie_globale(base, affect):
    i_global = 0
    for i in affect:
        i_global += inertie_cluster(base[affect[i]])
    return i_global
