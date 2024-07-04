from math import isnan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sympy import plot
from numba import jit, njit, set_num_threads
from numba.experimental import jitclass          # import the decorator
from numba import int32, float32, float64, int64


np.random.seed(seed=12)
# minst dataset

def import_minst():
    X , y = fetch_openml ("mnist_784", version =1 , return_X_y =True , as_frame = False )
    x = X/ 255.0
    y = np.array(y, dtype = int)

    return x, y

x,y = import_minst()
#x = x[:1000, :]
#y = y[:1000]

"""
class Binclass:

    def __init__(self, x,y, ng) -> None:
        self.x = x
        self.y = y
        self.gs = ng
        matrix_size = (self.gs, len(x[0,:]))
        self.mus = np.random.uniform(0, 1, matrix_size)
        alpha = np.ones(ng)
        self.pis = np.random.dirichlet(alpha)
        self.pis = self.pis/ np.sum(self.pis)
       
  
    def belonging(self, xl, gind):
        pall = 0
        prip = np.array([])
        for i in range(self.gs):
            #prod = 1
            tempsum = 0
            for j in range(len(self.mus[0,:])):
            #    prod *= (self.mus[i,j]**(self.x[xl, j]) * 
            #    (1-self.mus[i,j])**(1-self.x[xl, j]))
                 tempsum += x[xl, j] * np.log(mus[i, j]) + (1 - x[xl, j]) * np.log((1 - mus[i, j]))
            pripi = np.exp(tempsum) -np.log(tempsum)
            prip = np.append(prip, pripi)
            pall += prip[i]*self.pis[i]

        return (prip[gind]/pall)
    
    def update(self):

        psumgrup = 0
        newmus = np.zeros((self.gs, len(self.x[0,:])))
        pjs = np.zeros(self.gs)

        
        for j in range(self.gs):
            psum = np.zeros(len(self.x[0,:]))
            pj = 0
            print(j)
            for k in range(len(self.x[:,0])):
                temp = self.belonging(k, j)
                psumgrup += temp
                psum += temp*self.x[k,:]
                pj += temp
            
            newmus[j, :] = psum
            pjs[j] = pj

        self.mus = newmus/psumgrup
        self.pis = pjs/psumgrup
"""


@jit(nopython=True)
def belonging(xl, mus, pis, gs, gind, bin):
    pall = 0
    max = np.inf
    prip = np.empty(gs)
    for i in range(gs):
        tempsum = 0
        for j in range(len(mus[0, :])):
            #tempsum += x[xl, j] * np.log(mus[i, j]) + (1 - x[xl, j]) * np.log((1 - mus[i, j]))
            tempsum +=  np.log(mus[i, j]**x[xl, j] * (1 - mus[i, j])**(bin - x[xl, j]))
    
        prip[i] = tempsum

    mp = np.max(prip)
    
    for i in range(len(prip)):
        prip[i] = np.exp(prip[i] -mp)
        pall += prip[i] * pis[i]
    

    return prip[gind] /pall

@jit(nopython=True)
def update(x,mus, pis,gs, bin):
    psumgrup = 0
    newmus = np.zeros((gs, len(x[0, :])))
    pjs = np.zeros(gs)

    for j in range(gs):
        print(j)
        psum = np.zeros(len(x[0, :]))
        pj = 0
        for k in range(len(x[:, 0])):
            temp = belonging(k,mus, pis, gs, j, bin)
            psumgrup += temp
            
            psum += temp * x[k, :]
            pj += temp
        newmus[j, :] = psum/pj
        pjs[j] = pj

    mus = newmus
    pis = pjs / psumgrup
    return mus, pis

@jit(nopython=True)
def binclass(x, y, ng, iter, bin=1):
    gs = ng
    matrix_size = (gs, len(x[0, :]))
    mus = np.random.uniform(0.2, 0.8, matrix_size)
    alpha = np.ones(ng)
    pis = np.random.dirichlet(alpha)
    pis = pis / np.sum(pis)

    for i in range(iter):
        temp = update(x,mus, pis, gs, bin)
        mus = temp[0]
        pis =temp[1]

    return mus, pis

def plotnum(mus):
    fig, axes = plt.subplots(2, 5, figsize=(7, 4))  # 2 rows, 5 columns for 10 plots

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(mus[i].reshape((28, 28)))
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"./plots/num_3iter_bin2.png")
    plt.show()

def plot_binomial_grid(mu):
    fig, axs = plt.subplots(2, 5, figsize=(7, 4), tight_layout=True)
    axs = axs.ravel()

    for n in range(1, 11):
        x_new = np.random.binomial(n=n, p=mu)
        image_matrix = x_new.reshape((28, 28))

        axs[n-1].imshow(image_matrix, cmap='viridis')
        axs[n-1].set_title(f'n={n}')
        axs[n-1].axis('off')

    plt.savefig(f"./plots/1newgen_3iter.png")

if __name__=='__main__':

    #bin = Binclass(x,y,10)
    #for i in range(1):
    #    bin.update()
    mus, pis = binclass(x,y,10,3, 2)
    plotnum(mus)
    #plot_binomial_grid(mus[2])
        
