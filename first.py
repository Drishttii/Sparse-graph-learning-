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

def to_get_the_gradient(Lx, N, Y):
    
    A = [None] * 15
    D1=0
    x=0
    for x in range(0,15) :
        A[x] = ((-2)**x/math.factorial(x))*(Lx**x)
        D1 = D1 + A[x]

    B = [None] * 15
    D2=0
    x=0
    B[0] = 0
    for x in range(1,15) :
        B[x] = (-(-1)**x/math.factorial(x))*(Lx**x)
        D2 = D2 + B[x]

    DL = np.append(D1, D2, axis=1)
    DL = normalize(DL, norm='l1', axis=1)
    
    Za = orthogonal_mp(D1, Y, 2, tol=None,precompute=False, copy_X=True, return_path=False, return_n_iter=False)
    Zb = orthogonal_mp(D2, Y, 2, tol=None,precompute=False, copy_X=True, return_path=False, return_n_iter=False)
    Z = np.append(Za, Zb, axis=0)
    #Z = orthogonal_mp(D, Y, 2,tol=None,precompute=False, copy_X=True, return_path=False, return_n_iter=False)
    
    P = Y - DL.dot(Z)
    P = P.transpose()
    degree = np.zeros((N, N))
    colsum = W.sum(axis=0)
    
    for j in range(0,N):
        degree[j][j] = colsum[j]

    degree = np.linalg.matrix_power(degree, -1)
    Degree = linalg.sqrtm(degree)

    O = np.ones((N, N))
    I = np.identity(N)
    
    sum2 = 0
    Bk = 0
    F1 = 0

    sum1 = 0
    FF1 = 0
    BBk = 0

    for k in range(1, 15):
        for r in range(1, k-1):

            L1 = NL**(k-r-1)
            L2 = NL**r
            
            F = Degree.dot(L1.dot(Za.dot(P.dot(L2.dot(Degree)))))
            F2 = 2*(F.transpose())
        
            F1 = F1 + F2
            F1 = -F1
            Degree.dot(W.dot(F.dot(Degree)))
            F.dot(W.dot(degree))
            
            B = Degree.dot(W.dot(F.dot(Degree))) + F.dot(W.dot(degree))
            Bk = Bk + B

        R = O*(np.multiply(Bk, I))

        Gk = ((-2)**k/math.factorial(k))*(F1+R)
        sum2 = sum2 + Gk 

    for t in range(1, 15):
        for c in range(1, t-1):

            LL1 = NL**(t-c-1)
            LL2 = NL**c
            
            FF = Degree.dot(LL1.dot(Zb.dot(P.dot(LL2.dot(Degree)))))
            
            FF2 = 2*(FF.transpose())
            FF1 = FF1 + FF2

            FF1 = -FF1
          
            Degree.dot(W.dot(FF.dot(Degree)))
            
            FF.dot(W.dot(degree))
            
            BB = Degree.dot(W.dot(FF.dot(Degree))) + FF.dot(W.dot(degree))
            BBk = BBk + BB

        RR = O*(np.multiply(BBk, I))

        Gt = (-(-1)**t/math.factorial(t))*(FF1+RR)
        Gt[0]=0
        sum1 = sum1 + Gt 

    Bw = 1
    beta = Bw * (np.sign(W))
   
    sumf = sum2 + sum1 + beta

    for y in range(0,N):
        for u in range(0, N):
            if(y<u):
                sumf[y][u] = 0
               
    return sumf, Z, DL
     
