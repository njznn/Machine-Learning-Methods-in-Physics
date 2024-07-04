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





path = "/data/PSUF_naloge/3-naloga/DataK/"
#cas
dt = 5e-6
num_timesteps = 240000
nth_step_save=600
K_max = 20e-12

T = dt*num_timesteps
Nt = num_timesteps//nth_step_save
time = np.linspace(0,T, Nt)


def load_data(lbd, noise):
    K_max = 20e-12
    Kval = np.load((path + 'Kvalues.npy'))/K_max
    inten = np.load(path + f'intensity{lbd}noise{noise}.npy')

    return inten, Kval




def plot_int(lbd, noise, indeksi):
    X,Y = load_data(lbd, noise)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in indeksi:
        ax.plot(time, X[i,:], label=f'K={round(Y[i]/1e-12, 2)} pN')
    ax.set_xlabel('t')
    ax.set_ylabel('I(t)')
    #ax.set_xlim(0,0.8)
    ax.legend()
    plt.title('$\lambda =$'+f'{lbd} nm')
    plt.savefig(f'./plots/int_{lbd}_{noise}_special_2.png')



def plot_thetaz0(indeks):
    Y = np.load((path + 'Kvalues.npy'))
    theta0z = np.load(path + 'theta0.npy')
    #order = np.argsort(Y)
    fig, ax = plt.subplots(figsize=(8, 6))
    #pltind = [i*(int(len(order)/10)) for i in range(0,10)]
    #pltind = [pltind[1],pltind[4], pltind[7]]
    for i in indeks:
        ax.plot(np.linspace(0,10, len(theta0z[0, :])),theta0z[i,:], label=f'K={round(Y[i]/1e-12, 2)} pN', yunits=radians)
        
    ax.set_xlabel('$z[\mu m]$')
    ax.set_ylabel(r'$\theta_z(0)$')
    unit   = 0.5
    y_tick = np.arange(-1, 1+unit, unit)

    y_label = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$",   r"$+ \pi$"]
    ax.set_yticks(y_tick*np.pi)
    ax.set_yticklabels(y_label)
    
    ax.legend()
    plt.title('$\lambda =$'+f'{lbd} nm')
    plt.savefig(f'./plots/thetaz_0_special2.png')





def kfold_nn_model(X,Y,num_nn, activ_layers, optimizer, loss, metrics,lr, epoch, batch_size,Nfold,  savetitle='',save='False', ):
    loss_ = np.zeros((Nfold, epoch))
    val_loss_ = np.zeros((Nfold, epoch))
    kf = KFold(n_splits=Nfold)
    model = None
    for i,(train , test) in enumerate(kf.split(X,Y)):
        model = Sequential()
        model.add(Dense(num_nn[0],activation=activ_layers[0], input_shape=(X.shape[1],)))
        for j in range(1,len(num_nn)):
            model.add(Dense(num_nn[j],activation=activ_layers[j]))
        
        model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics=[metrics])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0,)
        
        earlystop = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-5,
        patience=10,
        verbose=2,
        mode='auto',
        baseline=None,
        restore_best_weights=False
        )
        
        history = model.fit(X[train], Y[train], epochs = epoch, verbose=2,batch_size=batch_size, 
                            validation_data=(X[test], Y[test]), shuffle=True, callbacks=[reduce_lr, earlystop])
        
        
        loss_[i,:len(history.history["loss"])] = history.history["loss"]
        val_loss_[i,:len(history.history["val_loss"])] = history.history["val_loss"]
         # Clean memory after use
        del history
        tensorflow.keras.backend.clear_session()
        gc.collect() #garbage collector collect
        
    


    return loss_, val_loss_

def plot_activation_dep(activationfun, X,Y,num_nn, optimizer, loss, metrics,lr, epoch, batch_size,Nfold,  savetitle='',save='False', ):
   
    cmap = plt.get_cmap("tab10")
    for k, act in enumerate(activationfun):
        loss_, val_loss = kfold_nn_model(X,Y,num_nn, [act, act, 'relu'],
                                          optimizer, loss, metrics,lr, epoch,batch_size,Nfold,  savetitle=savetitle,save=save, )
        #avgloss = np.average(loss_, axis=0)
        avgvalloss = np.nanmean(val_loss, axis=0)
        avgvalloss = np.trim_zeros(avgvalloss)
        val_loss_err = np.std(val_loss, axis=0)
        val_loss_err = val_loss_err[:len(avgvalloss)]

        

        epoch_arr = np.arange(epoch)
        plt.plot(epoch_arr[:len(avgvalloss)], avgvalloss,label=activationfun[k], c=cmap(k))
        plt.fill_between(epoch_arr[:len(avgvalloss)], avgvalloss-val_loss_err, avgvalloss + val_loss_err, color=cmap(k), alpha=0.25)
        

    plt.xlabel('Epohe')
    plt.ylabel('Funkcija izgube')
    plt.yscale('log')
    plt.grid()
    plt.title(f'optimizer={str(optimizer).split(".")[-1][:-2]},lr={lr}, loss={loss}, \n batch_size={batch_size}, hidden layer={int(len(num_nn)-2)} ')
    plt.legend(loc="upper right")

    plt.savefig('./plots/activation_loss_half.png')


def plot_lossf_dep(lossf,X,Y,num_nn, activ_layers, optimizer, metrics,lr, epoch, batch_size,Nfold):
    cmap = plt.get_cmap("tab10")
    for k, act in enumerate(lossf):
        clear_session()
        loss_, val_loss = kfold_nn_model(X,Y,num_nn, activ_layers,
                                          optimizer, act, metrics,lr, epoch,batch_size,Nfold )
        avgloss = np.average(loss_, axis=0)
        avgvalloss = np.nanmean(val_loss, axis=0)
        avgvalloss = np.trim_zeros(avgvalloss)
        val_loss_err = np.std(val_loss, axis=0)
        val_loss_err = val_loss_err[:len(avgvalloss)]

        

        epoch_arr = np.arange(epoch)
        plt.plot(epoch_arr[:len(avgvalloss)], avgvalloss, label=lossf[k], c=cmap(k))
        plt.fill_between(epoch_arr[:len(avgvalloss)], avgvalloss-val_loss_err, avgvalloss + val_loss_err, color=cmap(k), alpha=0.25)

    plt.xlabel('Epohe')
    plt.ylabel('Funkcija izgube')
    plt.grid()
    plt.yscale('log')
    plt.title(f'optimizer={str(optimizer).split(".")[-1][:-2]},lr={lr}, activation={activ_layers[0]}, \n batch_size={batch_size}, hidden layer={int(len(num_nn)-2)} ')
    plt.legend(loc="upper right")

    plt.savefig('./plots/lossf_loss.png')

def plot_batch_dep(batch,X,Y,num_nn, activ_layers, optimizer, loss, metrics,lr, epoch,Nfold):
    cmap = plt.get_cmap("tab10")
    for k, act in enumerate(batch):
        clear_session()
        loss_, val_loss = kfold_nn_model(X,Y,num_nn, activ_layers,
                                          optimizer, loss, metrics,lr, epoch,act,Nfold )
        avgloss = np.average(loss_, axis=0)
        avgvalloss = np.nanmean(val_loss, axis=0)
        avgvalloss = np.trim_zeros(avgvalloss)
        val_loss_err = np.std(val_loss, axis=0)
        val_loss_err = val_loss_err[:len(avgvalloss)]

        

        epoch_arr = np.arange(epoch)
        plt.plot(epoch_arr[:len(avgvalloss)], avgvalloss, label=f'batch_size={batch[k]}', c=cmap(k))
        plt.fill_between(epoch_arr[:len(avgvalloss)], avgvalloss-val_loss_err, avgvalloss + val_loss_err, color=cmap(k), alpha=0.25)

    plt.xlabel('Epohe')
    plt.ylabel('Funkcija izgube')
    plt.grid()
    plt.yscale('log')
    plt.title(f'optimizer={str(optimizer).split(".")[-1][:-2]},lr={lr}, activation={activ_layers[0]},\n loss={loss}, hidden layer={int(len(num_nn)-2)} ')
    plt.legend(loc="upper right")

    plt.savefig('./plots/batch_loss.png')

def plot_nn_dep(num_nn_arr,X,Y, activ_layers, optimizer, loss, metrics,lr, epoch,batch_size,Nfold):
    cmap = plt.get_cmap("tab10")
    for k, act in enumerate(num_nn_arr):
        clear_session()
        loss_, val_loss = kfold_nn_model(X,Y,act, activ_layers[k],
                                          optimizer, loss, metrics,lr, epoch,batch_size, Nfold )
        avgloss = np.average(loss_, axis=0)
        avgvalloss = np.nanmean(val_loss, axis=0)
        avgvalloss = np.trim_zeros(avgvalloss)
        val_loss_err = np.std(val_loss, axis=0)
        val_loss_err = val_loss_err[:len(avgvalloss)]

        epoch_arr = np.arange(epoch)
        plt.plot(epoch_arr[:len(avgvalloss)], avgvalloss, label=f'N_nn_inner={k+1}', c=cmap(k))
        plt.fill_between(epoch_arr[:len(avgvalloss)], avgvalloss-val_loss_err, avgvalloss + val_loss_err, color=cmap(k), alpha=0.25)

    plt.xlabel('Epohe')
    plt.ylabel('Funkcija izgube')
    plt.grid()
    plt.yscale('log')
    plt.title(f'optimizer={str(optimizer).split(".")[-1][:-2]},lr={lr}, activation={activ_layers[0][0]}, \n loss={loss}, batch_size={batch_size} ')
    plt.legend(loc="upper right")

    plt.savefig('./plots/nn_loss.png')

def plot_neutron_dep(neutronn, X,Y, activ_layers, optimizer, loss, metrics,lr, epoch,batch_size,Nfold):
    cmap = plt.get_cmap("tab10")
    for k, act in enumerate(neutronn):
        clear_session()
        loss_, val_loss = kfold_nn_model(X,Y,act, activ_layers,
                                          optimizer, loss, metrics,lr, epoch,batch_size, Nfold )
        avgloss = np.average(loss_, axis=0)
        avgvalloss = np.nanmean(val_loss, axis=0)
        avgvalloss = np.trim_zeros(avgvalloss)
        val_loss_err = np.std(val_loss, axis=0)
        val_loss_err = val_loss_err[:len(avgvalloss)]

        epoch_arr = np.arange(epoch)
        plt.plot(epoch_arr[:len(avgvalloss)], avgvalloss, label=f'N_nn={neutronn[k][0]}-{neutronn[k][1]}-{neutronn[k][2]}', c=cmap(k))
        plt.fill_between(epoch_arr[:len(avgvalloss)], avgvalloss-val_loss_err, avgvalloss + val_loss_err, color=cmap(k), alpha=0.25)

    plt.xlabel('Epohe')
    plt.ylabel('Funkcija izgube')
    plt.grid()
    plt.yscale('log')
    plt.title(f'optimizer={str(optimizer).split(".")[-1][:-2]},lr={lr}, activation={activ_layers[0]}, \n loss={loss}, batch_size={batch_size} ')
    plt.legend(loc="upper right")

    plt.savefig('./plots/nn_num_loss.png')


def calculate_prediction_avg(lbd, noise,test_size,num_nn, activ_layers, optimizer,
                          loss, metrics,lr, epoch, batch_size,Nfold):
    X,Y=load_data(lbd, noise)
    #divide data
    nt = int((1.0- test_size)*len(X[:,0]))
    Xtnv=X[:nt,:]
    Ytnv = Y[:nt]
    Xtnv,Ytnv = interpolate_and_combine(Xtnv, Ytnv, 0.1)
    
    Xtest=X[nt:,:]
    Ytest=Y[nt:]

    kf = KFold(n_splits=Nfold)
    Ypredict = (1.0)* np.zeros(len(Ytest))
    for i, (train , test) in enumerate(kf.split(Xtnv, Ytnv)):
        model = Sequential()
        model.add(Dense(num_nn[0],activation=activ_layers[0], input_shape=(Xtnv.shape[1],)))
        for j in range(1,len(num_nn)):
            model.add(Dense(num_nn[j],activation=activ_layers[j]))
        
        model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics=[metrics])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=2, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0,)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-5,patience=10,verbose=2, mode='auto',baseline=None,restore_best_weights=True)
        
        model.fit(Xtnv[train], Ytnv[train], epochs = epoch, verbose=2,batch_size=batch_size,validation_data=(Xtnv[test], Ytnv[test]), shuffle=True,callbacks=[reduce_lr, earlystop ])
        Ypredict += model.predict(Xtest)[:,0]
        del model
        tensorflow.keras.backend.clear_session()
        gc.collect() #garbage collector collect
        

    return Ytest, Ypredict/Nfold

def calculate_prediction_one(lbd, noise,test_size, val_size,num_nn, activ_layers, optimizer,
                          loss, metrics,lr, epoch, batch_size,savetitle='',save='False'):
    X,Y=load_data(lbd, noise)
    Xn,Yn=load_data(lbd, 100)
    #divide data
    nt = int((1.0- test_size-val_size)*len(X[:,0]))
    ntt = int((1.0-val_size)*len(X[:,0]))

    Xtrain=X[:nt,:]
    Ytrain = Y[:nt]

    Xval=X[nt:ntt,:]
    Yval = Y[nt:ntt]

    Xtest=Xn[ntt:,:]
    Ytest=Yn[ntt:]
    
    model = Sequential()
    model.add(Dense(num_nn[0],activation=activ_layers[0], input_shape=(X.shape[1],)))
    for j in range(1,len(num_nn)):
        model.add(Dense(num_nn[j],activation=activ_layers[j]))
        
    model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics=[metrics])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=2, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0,)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-5,patience=10,verbose=2, mode='auto',baseline=None,restore_best_weights=True)
        
    model.fit(Xtrain, Ytrain, epochs = epoch, verbose=2,batch_size=batch_size,validation_data=(Xval, Yval), shuffle=True,callbacks=[reduce_lr, earlystop ])
    Ypredict = model.predict(Xtest)[:,0]
        

    return Ytest, Ypredict

def plot_predictions(Ytest, Ypred, noise=0):
    Ytest *=20
    Ypred *= 20
    fig1, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(Ytest, Ypred, s=2, c='b')
    ax.plot(Ytest, Ytest, c='r')

    ax.set_xlabel('$Y_{test}[pN]$')
    ax.set_ylabel('$Y_{predict}[pN]$')


    fig2, ax2 = plt.subplots(figsize=(6, 4))
    h = ax2.hist2d(Ytest, Ypred, bins=100, cmap='viridis', cmin=1, norm=colors.LogNorm())
    fig2.colorbar(h[3], label='N')
    ax2.set_xlabel('$K_{test}[pN]$')
    ax2.set_ylabel('$K_{predict}[pN]$')
    

    

    return fig1, fig2

def grid_search(lbd, noise, Nfold=3, data_percent=0.3, lr=5e-4,epochs=150, batch_size=100, filename=""):
    X,Y = load_data(lbd, noise)

    # Save function parameters to a text file
    with open("./modeli/"+filename, 'w') as f:
        f.write("lam: %f\n" % lbd)
        f.write("noise: %f\n" % noise)
        f.write("folds: %d\n" % Nfold)
        f.write("data_percent: %f\n" % lr)
        f.write("epochs: %d\n" % epochs)
        f.write("batch_size: %d\n" % batch_size)
    #divide data
    
    n = Y.shape[0]
    Y = Y[:(int)(data_percent * n)]
    X = X[:(int)(data_percent * n), :]
    
    def nn_model_2(momentum=1e-4, lr=5e-4, **kwargs):
        """
        model = Sequential()
        model.add(Dense(arch[0],activation='relu', input_shape=(X.shape[1],)))
        for j in range(1,len(arch)-1):
            model.add(Dense(arch[j],activation='relu', input_shape=(X.shape[1],)))

        """
        model = Sequential()
        
        model.add(Dense(400,activation='relu', input_shape=(X.shape[1],)))
        model.add(Dense(200,activation='relu', input_shape=(X.shape[1],)))
        model.add(Dense(100,activation='relu', input_shape=(X.shape[1],)))
        model.add(Dense(1,activation='relu'))
        model.compile(optimizer=optimizers.SGD(learning_rate=lr, momentum=momentum), 
                      loss='mean_squared_logarithmic_error', metrics=['mean_squared_error'])
        return model

    param_grid = {
        "model__momentum": [1e-4, 1e-3, 1e-2, 1e-1, 1],
        "model__lr": [5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
        # "model__lr": [1e-4, 1e-5, 1e-6],
        #"model__architecture": [(400, 200, 100), (200, 100, 50), (400,400,400), (200,100)]
        
    }
    
    
    est = KerasRegressor(model=nn_model_2, verbose=2, epochs=epochs, batch_size=batch_size)
    gs = GridSearchCV(estimator=est, param_grid=param_grid,verbose=2, cv=Nfold, scoring='neg_mean_squared_log_error', n_jobs=-1)
    gs_res = gs.fit(X, Y, verbose=2)
   
    means = gs_res.cv_results_['mean_test_score']
    print(means)
    stds = gs_res.cv_results_['std_test_score']
    params = gs_res.cv_results_['params']
    # Save results to a text file
    with open("./modeli/"+filename, 'a') as f:
        f.write("Best: %f using %s\n" % (gs_res.best_score_, gs_res.best_params_))
        f.write("Mean scores and standard deviations:\n")
        for i, (mean, std, params) in enumerate(zip(gs_res.cv_results_['mean_test_score'],
                                                    gs_res.cv_results_['std_test_score'],
                                                    gs_res.cv_results_['params'])):
            f.write("Fold %d: %f +/- %f\n" % (i+1, mean, std))
            f.write("Params: %s\n" % params)
    print(f'means={means} \n')
    print(f'stds={stds} \n')
    print(f'params={params} \n')
    print(f'batch={batch_size}, epoch={epochs}, nfold={Nfold}')

def plot_min_scores(file_path, savefile):
    with open(file_path, 'r') as file:
        content = file.read()

    fold_data_list = []
   
    # Extracting Best and Fold information using regular expressions
    
    fold_matches = re.findall(r'Fold (\d+): (.+?)\nParams: (.+?)\n', content)

    for fold_match in fold_matches:
        fold_num = int(fold_match[0])
        fold_score = float(fold_match[1].split(' ')[0])
        fold_params = eval(fold_match[2])
        fold_data_list.append({'score': abs(fold_score), 'momentum': fold_params['model__momentum'], 'lr': fold_params['model__lr'], 'fold_num': fold_num})

    dict_list = fold_data_list
    
    learning_rates = sorted(list(set(item['lr'] for item in dict_list)))
    architectures = sorted(list(set(item['momentum'] for item in dict_list)))
    
    # Create a matrix to store scores
    score_matrix = np.zeros((len(learning_rates), len(architectures)))

    for item in dict_list:
        lr_idx = learning_rates.index(item['lr'])
        arch_idx = architectures.index(item['momentum'])
        score_matrix[lr_idx, arch_idx] = item['score']
    

    plt.figure(figsize=(6, 4))
    im = plt.imshow(score_matrix, cmap='viridis', aspect='auto', interpolation='none',
                    norm=LogNorm(vmin=score_matrix.min(), vmax=score_matrix.max()))

    # Set the tick labels for x-axis (architectures)
    plt.xticks(np.arange(len(architectures)), architectures, rotation = 45)
    plt.xlabel('Momentum')

    # Set the tick labels for y-axis (learning rates)
    plt.yticks(np.arange(len(learning_rates)), ["{:.0e}".format(lr) for lr in learning_rates])
    plt.ylabel('Learning Rate')

    # Add color bar
    cbar = plt.colorbar(im, label='log(MSLE)')
    plt.subplots_adjust(bottom=0.3)
    # Add title with parameters
    title='epoch=50, batch_size=50, Nfold=3'
    plt.title(title)
    plt.savefig('plots/'+savefile)

def interpolate_and_combine(original_matrix, original_vector, lower_shift):
    def interpolate_matrix(matrix, vector, lower_shift):
        num_rows, num_columns = matrix.shape
        
        # Create an array of new intensities based on the lower shift
        new_intensities = np.linspace(lower_shift, 1.2, num_columns)

        # Initialize arrays to store the interpolated matrix and the unchanged vector
        interpolated_matrix = np.zeros((num_rows, num_columns))
        unchanged_vector = vector.copy()

        # Interpolate each row of the matrix individually
        for i in range(num_rows):
            f = interpolate.interp1d(new_intensities, matrix[i, :], kind='linear', fill_value='extrapolate')
            interpolated_matrix[i, :] = f(np.linspace(lower_shift, 1.2, num_columns))

        return interpolated_matrix, unchanged_vector

    # Interpolate the original matrix and keep the vector unchanged
    interpolated_matrix, unchanged_vector = interpolate_matrix(original_matrix, original_vector, lower_shift)

    # Combine the original and interpolated matrices
    combined_matrix = np.concatenate((original_matrix, interpolated_matrix), axis=0)
    vec = np.concatenate((unchanged_vector, unchanged_vector))
    return combined_matrix, vec
#############################################################
if __name__=="__main__":
    
   
    

    
    #order = np.argsort(np.argmax(X, axis=1))
    #print(order)
    #pltind = [i*(int(len(order)/10)) for i in range(0,10)]
    #pltind = [pltind[1],pltind[4], pltind[7]]
    #plot_int(500,0,order[pltind])
    #plot_int(500,0,order[-5:])


    
    
    #order = np.argsort(np.argmax(X, axis=1))
    #plot_thetaz0(order[-5:])
    """
    
    #plot_learning_progress_activations(500, 0, ["relu", "sigmoid", "tanh", "softmax"], lr=5e-4,epochs=100, batch_size=100, folds=5)
    
    #plot_activation_dep( ['relu', 'sigmoid', 'tanh', 'softmax'],X,Y,[200,100,1],optimizers.Adam, 'mean_absolute_error', 'mean_squared_error' ,5e-4, 100, 100, 5)
    
    #plot_lossf_dep(['mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error'], 
    #              X,Y,[200,100,1], ['relu', 'relu','relu'], optimizers.Adam,'mean_squared_error' ,5e-4, 100, 100, 5)

    #plot_batch_dep([16,32,64,128,256], 
    #              X,Y,[200,100,1], ['relu', 'relu','relu'], optimizers.Adam,'mean_absolute_error','mean_squared_error' ,5e-4, 100,  5)
    
    #plot_nn_dep([[200,100,1], [200,100,100,1],[200,100,100,100,1]], 
    #              X,Y, [['relu', 'relu','relu'], ['relu', 'relu','relu', 'relu'], ['relu', 'relu','relu', 'relu', 'relu']]
    #                                , optimizers.Adam,'mean_absolute_error','mean_squared_error' ,5e-4, 100, 100 ,5)

    #plot_neutron_dep([[200,100,1], [300,100,1], [400,200,1], [200,200,1]], X,Y, ['relu', 'relu','relu'],
    #                  optimizers.Adam,'mean_absolute_error','mean_squared_error',5e-4, 100,100,5)



    #predict:
    """
    """
    Ytt, Ypred = calculate_prediction_avg(500,100,0.5,[400,200,100,1], ['relu','relu', 'relu','relu'],
                                          optimizers.Adam,'mean_squared_logarithmic_error','mean_squared_error',
                                            5e-4,150,20,5)
    f1, f2 = plot_predictions(Ytt,Ypred)
    f2.savefig('./plots/pred_avg_500_0_hist_4-2-1-1_relu_adam_msle_mse_lrc_150_20_5_shift01.png')
    f1.savefig('./plots/pred_avg_500_0_4-2-1-1_relu_adam_msle_mse_lrc_150_20_5_shift01.png')
    """
    """
    
    Ytt, Ypred = calculate_prediction_one(500,0,0.1,0.2,[400,200,100,1], ['relu', 'relu', 'relu','relu']
                                          ,optimizers.Adam,'mean_squared_logarithmic_error','mean_squared_error',
                                            5e-4,150,20)
    f1, f2 = plot_predictions(Ytt,Ypred)
    f2.savefig('./plots/pred_one_500_0_hist_4-2-1-1_relu_adam_msle_mse_lrc_150_20_onnoise.png')
    f1.savefig('./plots/pred_one_500_0_4-2-1-1_relu_adam_msle_mse_lrc_150_20_onnoise.png')
    """
    
    #grid_search(500,0,3,0.6,5e-4,50,50, "gs1acc_sgd_mom")
    #calculate_grid_search(500,0, filename="gs2acc")
    #plot_min_scores("modeli/gs1acc_sgd_mom", "gc1acc_shd_mom.png")