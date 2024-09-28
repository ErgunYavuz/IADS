# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import random as rd


def dist_intracluster(base):
    d_max = 0
    for i in range(base.shape[0]):
        for j in range(i+1, base.shape[0]):
            d = km.dist_vect(base[i], base[j])
        if d > d_max:
            d_max = d
    return d_max


def global_intraclusters(base, affectation):
    dist_intra = []
    for i in affectation.values():
        dist_intra.append(dist_intracluster(base[i]))
    return max(dist_intra)


def sep_clusters(centres):
    d_min = 1000
    size = centres.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            if i != j:
                d = km.dist_vect(centres[i], centres[j])
                if d < d_min:
                    d_min = d
    return d_min


def evaluation(base, centres, affectation):
    return global_intraclusters(base, affectation)/sep_clusters(centres)
