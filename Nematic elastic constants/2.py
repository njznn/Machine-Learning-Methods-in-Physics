from audioop import avg
from xml.parsers.expat import model
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
#from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras import Sequential
from scikeras.wrappers import KerasRegressor
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
import tensorflow
from tensorflow.keras.metrics import MeanSquaredLogarithmicError
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from DataGeneration import theta_time_evolution
from matplotlib import cm
from matplotlib.colors import Normalize
from basic_units import cos, degrees, radians
from sklearn.model_selection import KFold
from matplotlib.colors import LogNorm
import re


import matplotlib.colors as colors
from scipy import interpolate


def load_measurments():
    path = "/data/PSUF_naloge/3-naloga/ExpData/"
    data = np.zeros((30,15600))
    for i in range(30):
        data[i,:] = np.load(path + f'exp_intensity_{i}.npy')

    return data

def measurmets_interp(xold,sign, newbeg,dur=1.2, pts=400):
    newx = np.linspace(newbeg, newbeg+dur, pts)
    newsign=np.zeros((len(sign[:,0]),pts))
    
    for i in range(len(sign[:,0])):
        f = interpolate.interp1d(xold, sign[i,:])
        newsign[i,:] = f(newx)
    

    return(newsign)



def load_data(lbd, noise, sidranje):
    path = "/data/PSUF_naloge/3-naloga/DataK13"
    K_max = 20e-12
    Kval = np.load((path +f'_{sidranje}/'+ 'Kvalues.npy'))/K_max
    inten = np.load(path +f'_{sidranje}/' + f'intensity{lbd}noise{noise}.npy')
    print(Kval)
    return inten, Kval


def plot_combined_histogram(matrix, line1=6.6, line2=9.0, savefile=""):
    # Plot combined histogram with lines
    plt.figure(figsize=(6, 4))
    # Calculate average and deviation for each column
    avg_col1 = np.round(np.mean(matrix[:, 0]),2)
    print(avg_col1)
    dev_col1 = np.std(matrix[:, 0])
    avg_col2 = np.mean(matrix[:, 1])
    dev_col2 = np.std(matrix[:, 1])

    # Histogram for the first column vector
    counts_col1, bins_col1, _ = plt.hist(matrix[:,0], bins=40, color='blue', alpha=0.7,
                                         label=r"$ \langle K_{11, pred} \rangle =$" + '{:.2f}'.format(avg_col1)+ "$\pm$"+ '{:.2f}'.format(dev_col1) + r"$ \ pN $", )
    plt.axvline(x=line1, color='blue', linestyle='dashed', linewidth=2, 
                label="$K_{11}$"+f'$={line1} \ pN$')
                

    # Histogram for the second column vector
    counts_col2, bins_col2, _ = plt.hist(matrix[:, 1], bins=40, color='green', alpha=0.7, 
                                         label=r"$ \langle K_{33,pred} \rangle =$" + '{:.2f}'.format(avg_col2)+ "$\pm$"+ '{:.2f}'.format(dev_col2) + r"$ \ pN $")
    plt.axvline(x=line2, color='green', linestyle='dashed', linewidth=2, label="$K_{33}$"+f'$={line2} \ pN$')

    
    
    
    plt.legend()

    

    plt.title(r'Elastic constant prediction')
    plt.xlabel('K [pN]')
    plt.ylabel('N')

    plt.savefig(f'./plots2/'+savefile+'.png')


def calculate_prediction_avg(lbd, noise,sidranje,test_size,num_nn, activ_layers, optimizer,
                          loss, metrics,lr, epoch, batch_size,Nfold, save=False):
    X,Y=load_data(lbd, noise, sidranje)
    #divide data
    nt = int((1.0- test_size)*len(X[:,0]))
    Xtnv=X[:nt,:]
    Ytnv = Y[:nt]
    
    Xtest=X[nt:,:]
    Ytest=Y[nt:]
    
    kf = KFold(n_splits=Nfold)
    #Ypredict = (1.0)* np.zeros((len(Ytest),2))
    for i, (train , test) in enumerate(kf.split(Xtnv, Ytnv)):
        model = Sequential()
        model.add(Dense(num_nn[0],activation=activ_layers[0], input_shape=(Xtnv.shape[1],)))
        for j in range(1,len(num_nn)):
            model.add(Dense(num_nn[j],activation=activ_layers[j]))
        
        model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics=[metrics])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=2, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0,)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-5,patience=10,verbose=2, mode='auto',baseline=None,restore_best_weights=True)
        
        model.fit(Xtnv[train], Ytnv[train], epochs = epoch, verbose=2,batch_size=batch_size,validation_data=(Xtnv[test], Ytnv[test]), shuffle=True,callbacks=[reduce_lr, earlystop ])
        
        #Ypredict += model.predict(Xtest)[:,0:2]
        if save==True:
            model.save(f'./modeli/{lbd}_{sidranje}_{i}.keras')
        del model
        tensorflow.keras.backend.clear_session()
        gc.collect() #garbage collector collect
        

    return None #Ytest, Ypredict/Nfold

def calculate_prediction_one(lbd, noise,sidranje,test_size,num_nn, activ_layers, optimizer,
                          loss, metrics,lr, epoch, batch_size,save='False'):
    X,Y=load_data(lbd, noise, sidranje)
    
    nt = int((1.0- test_size)*len(X[:,0]))

    Xtrain=X[:nt,:]
    Ytrain = Y[:nt]

    Xval=X[nt:,:]
    Yval = Y[nt:]

    model = Sequential()
    model.add(Dense(num_nn[0],activation=activ_layers[0], input_shape=(X.shape[1],)))
    for j in range(1,len(num_nn)):
        model.add(Dense(num_nn[j],activation=activ_layers[j]))
        
    model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics=[metrics])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=2, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0,)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-5,patience=10,verbose=2, mode='auto',baseline=None,restore_best_weights=True)
        
    model.fit(Xtrain, Ytrain, epochs = epoch, verbose=2,batch_size=batch_size,validation_data=(Xval, Yval), shuffle=True,callbacks=[reduce_lr, earlystop ])
    if save==True:
            model.save(f'./modeli/{lbd}_{sidranje}_one4.keras')
        

    return model.history

def plot_predictions(Ytest, Ypred, noise=0, sidranje='hh'):
    Ytest *=20
    Ypred *= 20
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'wspace': 0.4})
    K = ["$11$", "33"]
    for i in range(2):
        h = axs2[i].hist2d(Ytest[:, i], Ypred[:, i], bins=100, cmap='viridis', cmin=1, norm=colors.LogNorm())
        fig2.colorbar(h[3], ax=axs2[i], label='N')
        axs2[i].set_xlabel(f'$K{K[i]} \ [pN]$')
        axs2[i].set_ylabel(f'$K{K[i]}-predict  \ [pN]$')

    
    fig2.suptitle(f'{sidranje}')
    fig2.savefig(f'./plots2/konst_hist_{sidranje}_real.png')
    return None
   

def avg_pred(modarr, datapred, stmod):
    Kavg = np.zeros((len(datapred[:,0]),2))
    
    for i in range(stmod):
        print()
        Kavg += modarr[i].predict(datapred)[:len(datapred[:,0]),:]
        

    return Kavg/stmod

def expand_measurments(xold, initarr, measurments):
    newmatrix = measurmets_interp(xold,measurments,0)
    for i in initarr:
        newmatrix = np.concatenate((newmatrix, 
                measurmets_interp(xold,measurments,i)),axis=0)
    
    return newmatrix

if __name__=="__main__":

    
    #calculate_prediction_avg(505,0,"pp",0.0,[400,200,100,2], ['relu', 'relu', 'relu', 'relu'],
    #                        optimizers.Adam, 'mean_squared_logarithmic_error','mean_squared_error',
    #                                       5e-4,150,20,5, True)
    
    #plot_predictions(Ytt, Ypred, sidranje='pp')
    #print(load_measurments())
    #calculate_prediction_one(505,0,"ph",0.1,[400,200,100,2], ['relu', 'relu', 'relu', 'relu'],
    #                        optimizers.Adam, 'mean_squared_logarithmic_error','mean_squared_error',
    #     
    #                                   5e-4,150,20, save=True)

    
    t = np.linspace(0,1.95, 15600)
    exp = load_measurments()
    
    #exp_scal = measurmets_interp(t, exp,0.0 )
    shift = np.linspace(0,0.1,50)
    exp_scal=expand_measurments(t,shift,exp)
    
    model_arr = np.array([])
    for i in range(10):
        model_arr = np.append(model_arr, load_model(f'./modeli/505_ph_one{i}.keras'))
    
    
    Ypred=avg_pred(model_arr, exp_scal,10)
    Ypred *=20
    


    #one_const = tensorflow.keras.models.load_model('./modeli/505_ph_alllearn.keras')
    #print(one_const.summary())
    #Ypred = one_const.predict(exp_scal)[:len(exp_scal[:,0]),:]
    #Ypred *= 20 
    
    plot_combined_histogram(Ypred, savefile="hist_hist_exp_50_01_avg")
    