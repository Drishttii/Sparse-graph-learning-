from __future__ import print_function
from pygsp import graphs, filters, plotting 
from numpy.random import seed
from numpy.random import rand
from scipy import sparse
from scipy import stats
from scipy import linalg
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import normalize
from sklearn.linear_model import orthogonal_mp
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from PIL import Image
from scipy.fftpack import fft
from scipy.fftpack import dct, idct
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math



def function_to_gererate_signals(Lx, N):
    
    AA = [None] * 15
    DD1=0
    h=0
    for h in range(0,15) :
        AA[h] = ((-2)**h/math.factorial(h))*(Lx**h)
        DD1 = DD1 + AA[h]
        
    BB = [None] * 15
    DD2=0
    h=0
    BB[0] = 0
    for h in range(1,15) :
        BB[h] = (-(-1)**h/math.factorial(h))*(Lx**h)
        DD2 = DD2 + BB[h]
        
    rvs = stats.norm(loc=0, scale=1).rvs
    Xa = sparse.random(N, 1,density=0.1, data_rvs=rvs )
    Xb = sparse.random(N, 1,density=0.1, data_rvs=rvs )
    Xp = Xa.toarray()
    Xq = Xb.toarray()
    
    for i in range(1, 200):
        Xi = sparse.random(N, 1, density=0.1, data_rvs=rvs)
        Xi = Xi.toarray()
        Xp = np.append(Xp, Xi, axis=1)
        
    for f in range(1, 200):
        Xf = sparse.random(N, 1, density=0.1, data_rvs=rvs)
        Xf = Xf.toarray()
        Xq = np.append(Xq, Xf, axis=1)
        
    X = np.append(Xp, Xq, axis=0)
    D = np.append(DD1, DD2, axis=1)
    Y = D.dot(X)
    
    return Y, X
