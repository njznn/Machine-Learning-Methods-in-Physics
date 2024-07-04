from re import X
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import Normalize
from sympy import HadamardPower, fraction
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from scikeras.wrappers import KerasRegressor
from tensorflow import keras
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from matplotlib.colors import LogNorm
from numpy import linalg as LA
from lorentz_dopri_5 import lorenz63
from scipy.integrate import ode
import re
import gc


plt.rcParams['figure.figsize'] = (6, 4)
### lorents exact:



def plot_3d_curve_with_subplots(data,predicted, t_end):
    dt=0.01
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 2)

    N = int(t_end/dt) 
    t = np.linspace(0, t_end, N)
    

    # 3D subplot for the curve
    ax3d = fig.add_subplot(gs[0], projection='3d')
    for i in range(N-1):
        pl = ax3d.plot(data[i:i+2,0], data[i:i+2,1], data[i:i+2,2], color=cm.viridis(np.linspace(0,1,N))[i])
        ax3d.plot(predicted[i:i+2,0],predicted[i:i+2,1], predicted[i:i+2,2], color=cm.viridis(np.linspace(0,1,N))[i], linestyle='dashed')
    
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    
    ax3d.legend()
    cmappable = cm.ScalarMappable(Normalize(0,t_end))
    plt.colorbar(cmappable, ax=ax3d, label='t', fraction=0.03)

    ax2d = fig.add_subplot(gs[1])
    ax2d.plot(t, data[:N, 0], label='x(t)', color='green')
    ax2d.plot(t, data[:N, 1], label='y(t)', color='brown')
    ax2d.plot(t, data[:N, 2], label='z(t)', color='blue')
    ax2d.plot(t, predicted[:N, 0], color='green', linestyle='dashed')
    ax2d.plot(t, predicted[:N, 1],  color='brown', linestyle='dashed')
    ax2d.plot(t, predicted[:N, 2], color='blue', linestyle='dashed')
    ax2d.plot(1, 1, color='black',linestyle='dashed', label='predicted')
    ax2d.set_xlabel('t')
    
    ax2d.legend()
    fig.subplots_adjust(right=0.85)
    # Adjust layout
    plt.tight_layout()

    # Add a title
    

    # Show the plot
    return fig


#f1 = plot_3d_curve_with_subplots(data, 20)
#f1.savefig('./plots/1_lor_int.png')

####################### helper funcitons ##############
def scaler(input_field, inverse=False):
    mins = np.amin(data, axis=0) # z axis=0 najdemo min za vse 3 komponente
    maxs = np.amax(data, axis=0)
    if not inverse: # normalizacija
        return (input_field - mins) / (maxs - mins)
    else: # inverz normalizacije
        return input_field * (maxs - mins) + mins



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps, n_future_steps, mtm=False):
    X, y = list(), list()
    step = 1 if mtm==False else n_steps
    n_future_steps = 1 if mtm==False else n_future_steps
    for i in range(0,len(sequences), step):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix + n_future_steps > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        if mtm== True:
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:end_ix + n_steps, :]
            
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def split_sequences_diff(sequences, n_steps, n_future_steps=1):
    X, y = list(), list()
    for i in range(0,len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix + n_future_steps > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]-sequences[end_ix-1,:]
            
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



########################################
#splitting and making train and valitade set
def trainnval_set(k=10, Nend_train = 200000, Nend_validate = 50000, mtm=False):
    # k dolzina vhodnega zaporedja pri treningu
    # Nend_train velikost mnozice za ucenje
    #Nend_validate velikost mnozice za validacijo


    x_train,y_train = split_sequences(normalized_data[:Nend_train], k, k, mtm)
    x_validate,y_validate = split_sequences(normalized_data[Nend_train: Nend_train + Nend_validate], k, k, mtm)
    #shuffle
    new_order = np.random.choice(range(x_train.shape[0]),x_train.shape[0],replace=False)
    x_train = x_train[new_order]
    y_train = y_train[new_order]
    return x_train, y_train, x_validate, y_validate

def trainnval_set_diff(k=10, Nend_train = 200000, Nend_validate = 50000):
    # k dolzina vhodnega zaporedja pri treningu
    # Nend_train velikost mnozice za ucenje
    #Nend_validate velikost mnozice za validacijo


    x_train,y_train = split_sequences_diff(normalized_data[:Nend_train], k, k)
    x_validate,y_validate = split_sequences_diff(normalized_data[Nend_train: Nend_train + Nend_validate], k, k)
    #shuffle
    new_order = np.random.choice(range(x_train.shape[0]),x_train.shape[0],replace=False)
    x_train = x_train[new_order]
    y_train = y_train[new_order]
    return x_train, y_train, x_validate, y_validate



def SNN(Xt,Yt,Xval,Yval,hiddstate_arr, actf, loss, optimizer,epoch=100,batchs=32, mtm=False, k=1, savename="" ):
    inpt_shp = (None, 3) if mtm==False else (k,3)
    model = tf.keras.models.Sequential()
    seqsw = True if mtm==True or len(hiddstate_arr)>1 else False

    model.add(SimpleRNN(hiddstate_arr[0],activation = actf[0], input_shape=inpt_shp, return_sequences=seqsw))
    for i in range(1,len(hiddstate_arr)):
        if i==(len(hiddstate_arr)-1) and mtm== False:
            model.add(SimpleRNN(hiddstate_arr[i],activation = actf[i], return_sequences=False))
        else:
            model.add(SimpleRNN(hiddstate_arr[i],activation = actf[i], return_sequences=True))
    
    if mtm==False:
        model.add(Dense(3)) 
    else:
        model.add(tf.keras.layers.TimeDistributed(Dense(3)))
    
    model.compile(loss=loss, optimizer=optimizer)
    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=0, mode='min')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,patience=3, min_lr=0.0001)
    history = model.fit(Xt, Yt, validation_data=(Xval,Yval),
    epochs=epoch, batch_size=batchs, verbose=2, callbacks=[early_stop, reduce_lr])
    #early stop is out!!

    if not(savename == ""):
        model.save(f'./models/{savename}.keras')

    return history

def lstm(Xt,Yt,Xval,Yval,hiddstate_arr, actf, loss, optimizer,epoch=150,batchs=32, mtm=False, k=1, savename="" ):
    inpt_shp = (None, 3) if mtm==False else (k,3)
    model = tf.keras.models.Sequential()
    seqsw = True if mtm==True or len(hiddstate_arr)>1 else False

    model.add(LSTM(hiddstate_arr[0],activation = actf[0], input_shape=inpt_shp, return_sequences=seqsw))
    for i in range(1,len(hiddstate_arr)):
        if i==(len(hiddstate_arr)-1) and mtm== False:
            model.add(LSTM(hiddstate_arr[i],activation = actf[i], return_sequences=False))
        else:
            model.add(LSTM(hiddstate_arr[i],activation = actf[i], return_sequences=True))
    
    if mtm==False:
        model.add(Dense(3)) 
    else:
        model.add(tf.keras.layers.TimeDistributed(Dense(3)))
    
    model.compile(loss=loss, optimizer=optimizer)
    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=2, mode='min')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,patience=3, min_lr=0.0001)
    history = model.fit(Xt, Yt, validation_data=(Xval,Yval),
    epochs=epoch, batch_size=batchs, verbose=2, callbacks=[ reduce_lr])

    if not(savename == ""):
        model.save(f'./models/{savename}.keras')

    return history





def hiddenstate_dep(hiddenstate_size, actf, loss, optimizer):
    k=10
    Xt,Yt,Xval,Yval = trainnval_set(k, 200000,50000,mtm=False)
    cmap = plt.get_cmap("tab10")
    
    for i in range(len(hiddenstate_size)):
        y = SNN(Xt, Yt, Xval, Yval, hiddenstate_size[i], actf, loss, optimizer,mtm=False,k=10).history["val_loss"]

        epoch_arr = np.arange(len(y))
        linestyle = ['-', '--', '-.', ':'][i % 4]
        plt.plot(epoch_arr, y ,label=f'H={hiddenstate_size[i]}, {linestyle.strip()}', c=cmap(i), linestyle=linestyle)
        

        del y
        tf.keras.backend.clear_session()
        gc.collect() #garbage collector collect

    plt.xlabel('Epohe')
    plt.ylabel(f'val MSE')
    plt.yscale('log')
    plt.title(f'optimizer={optimizer}, \n batch_size={32}, hidden layer={0}, activation=tanh ')
    plt.legend()
    
    plt.savefig('./plots/H_dep_lstm_mtm.png')

def k_dep(k_arr, H_arr, actf, loss, optimizer):
    cmap = plt.get_cmap("tab10")
    
    for i in range(len(k_arr)):
        Xt,Yt,Xval,Yval = trainnval_set(k_arr[i], 200000,50000, mtm=False)
        y = lstm(Xt, Yt, Xval, Yval, H_arr, actf, loss, optimizer, mtm=False, k=1).history["val_loss"]

        epoch_arr = np.arange(len(y))
        linestyle = ['-', '--', '-.', ':'][i % 4]
        plt.plot(epoch_arr, y ,label=f'k={k_arr[i]}, {linestyle.strip()}', c=cmap(i), linestyle=linestyle)
        

        del y
        tf.keras.backend.clear_session()
        gc.collect() #garbage collector collect

    plt.xlabel('Epohe')
    plt.ylabel(f'val MSE')
    plt.yscale('log')
    plt.title(f'optimizer={optimizer}, \n batch_size={32}, hidden layer={0}, activation=tanh, H={H_arr[0]} ')
    plt.legend()
    
    plt.savefig('./plots/k_dep_rnn2_mto.png')

def actdep_dep(actf_arr, H_arr, loss, optimizer):
    cmap = plt.get_cmap("tab10")
    kopt = 16
    Xt,Yt,Xval,Yval = trainnval_set(kopt, 200000,50000, False)
    for i in range(len(actf_arr)):
        y = SNN(Xt, Yt, Xval, Yval, H_arr,actf_arr[i], loss, optimizer, mtm=False, k=1).history["val_loss"]

        epoch_arr = np.arange(len(y))
        linestyle = ['-', '--', '-.', ':'][i % 4]
        plt.plot(epoch_arr, y ,label=f'act_f={actf_arr[i]}, {linestyle.strip()}', c=cmap(i), linestyle=linestyle)
        

        del y
        tf.keras.backend.clear_session()
        gc.collect() #garbage collector collect

    plt.xlabel('Epohe')
    plt.ylabel(f'val MSE')
    plt.yscale('log')
    plt.title(f'optimizer={optimizer}, \n batch_size={32},H={H_arr[0]} hidden layer={0}, k=16')
    plt.legend()
    
    plt.savefig('./plots/actf_dep_rnn_mto.png')


def layer_num_dep(H_arr,actf, loss, optimizer):
    cmap = plt.get_cmap("tab10")
    kopt = 32
    Xt,Yt,Xval,Yval = trainnval_set(kopt, 200000,50000, mtm=False)
    for i in range(len(H_arr)):
        y = lstm(Xt, Yt, Xval, Yval, H_arr[i],actf[i], loss, optimizer, mtm=False).history["val_loss"]

        epoch_arr = np.arange(len(y))
        linestyle = ['-', '--', '-.', ':'][i % 4]
        plt.plot(epoch_arr, y ,label=f'layer_num={len(H_arr[i])}, {linestyle.strip()}', c=cmap(i), linestyle=linestyle)
        

        del y
        tf.keras.backend.clear_session()
        gc.collect() #garbage collector collect

    plt.xlabel('Epohe')
    plt.ylabel(f'val MSE')
    plt.yscale('log')
    plt.title(f'optimizer={optimizer}, \n batch_size={32}, activ_fun={actf[0][0]}, H={H_arr[0][0]}, k={kopt}')
    plt.legend()
    
    plt.savefig('./plots/layernum_dep_lstm_mto.png')

#NOT WORKING!
def grid_search(actf, loss, optimizer,k,mtm,epoch=100,batchs=32,filename=""):
    Xt,Yt,Xval,Yval = trainnval_set(k, 300,150,mtm=mtm)
    print(Xt.shape, Yt.shape )
    # Save function parameters to a text file
    with open("./models/"+filename, 'w') as f:
        f.write("k: %d\n" % k)
        f.write(f'actf:  {actf}')
        f.write(f'optimizer:{optimizer}')
        f.write("epochs: %d\n" % epoch)
        f.write("batch_size: %d\n" % batchs)
        f.write(f'loss: {loss}')
    #divide data
    def SNN(hidd_arr=(32,32,32), lr=5e-4):
       
        inpt_shp = (None, 3) if mtm==False else (k,3)
        model = tf.keras.models.Sequential()
        seqsw = True if mtm==True or len(hidd_arr)>1 else False

        model.add(SimpleRNN(hidd_arr[0],activation = actf, input_shape=inpt_shp, return_sequences=seqsw))
        for i in range(1,len(hidd_arr)):
            if i==(len(hidd_arr)-1) and mtm== False:
                model.add(SimpleRNN(hidd_arr[i],activation = actf, return_sequences=False))
               
            else:
                model.add(SimpleRNN(hidd_arr[i],activation = actf, return_sequences=True))
        
        if mtm==False:
            model.add(Dense(3)) 
        else:
            
            model.add(tf.keras.layers.TimeDistributed(Dense(3)))
        
        model.compile(loss=loss, optimizer=optimizer(learning_rate=lr))
    
        return model

    param_grid = {
        "model__hidd_arr":[(64, 32, 16), (16, 32, 64), (32,32,32), (16,16,16), (32,16,16), (32,16,16)],
        "model__lr": [1e-3,5e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    }
    
    est = KerasRegressor(model=SNN, verbose=2, epochs=epoch, batch_size=batchs)
    gs = GridSearchCV(estimator=est, param_grid=param_grid,verbose=0,cv=5
                      ,scoring='neg_mean_squared_error', n_jobs=6)
    
    
    gs_res = gs.fit(Xt, Yt)
   
    #means = gs_res.cv_results_['mean_test_score']

    #stds = gs_res.cv_results_['std_test_score']
    #params = gs_res.cv_results_['params']
    # Save results to a text file
    
    with open("./models/"+filename, 'a') as f:
        f.write("Best: %f using %s\n" % (gs_res.best_score_, gs_res.best_params_))
        f.write("Mean scores and standard deviations:\n")
        for i, (mean, std, params) in enumerate(zip(gs_res.cv_results_['mean_test_score'],
                                                    gs_res.cv_results_['std_test_score'],
                                                    gs_res.cv_results_['params'])):
            f.write("Fold %d: %f +/- %f\n" % (i+1, mean, std))
            f.write("Params: %s\n" % params)

    
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
        fold_data_list.append({'score': abs(fold_score), 'hidd_arr': fold_params['model__hidd_arr'], 'lr': fold_params['model__lr'], 'fold_num': fold_num})

    dict_list = fold_data_list
    
    learning_rates = sorted(list(set(item['lr'] for item in dict_list)))
    architectures = sorted(list(set(item['hidd_arr'] for item in dict_list)))
    
    # Create a matrix to store scores
    score_matrix = np.zeros((len(learning_rates), len(architectures)))

    for item in dict_list:
        lr_idx = learning_rates.index(item['lr'])
        arch_idx = architectures.index(item['hidd_arr'])
        score_matrix[lr_idx, arch_idx] = item['score']
    

    plt.figure(figsize=(6, 4))
    im = plt.imshow(score_matrix, cmap='viridis', aspect='auto', interpolation='none',
                    norm=LogNorm(vmin=score_matrix.min(), vmax=score_matrix.max()))

    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            plt.text(j, i, f'{score_matrix[i, j]:.2e}', ha='center', va='center', color='white')
    # Set the tick labels for x-axis (architectures)
    plt.xticks(np.arange(len(architectures)), architectures, rotation = 45)
    plt.xlabel('Arhitecture')

    # Set the tick labels for y-axis (learning rates)
    plt.yticks(np.arange(len(learning_rates)), ["{:.0e}".format(lr) for lr in learning_rates])
    plt.ylabel('Learning Rate')

    # Add color bar
    cbar = plt.colorbar(im, label='log(MSLE)')
    plt.subplots_adjust(bottom=0.3)
    # Add title with parameters
    title='epoch=100, batch_size=32, Nfold=5,k=, activ_f = tanh'
    plt.title(title)
    plt.savefig('./plots/'+savefile)

def predict_mto(model, tend, k, diff=False):
    tmax = tend # kako dolgo simulacijo pozenemo (v cas. enotah L63 sistema)
    dt = 0.01 # mora biti enak dt za pri treniranju!
    n = int(tmax/dt) + 1 # st. korakov v simulaciji
    m = k # dolzina vhoda pred napovedovanjem
    Ntest = 1000 # na koliko razlicnih zacetnih pogojih testiramo
    data_lorenz_NN = np.zeros((Ntest, m + n-1, 3)) # sem shranjujemo
    dn = int(1/dt) # zacetni pogoji v test setu naj bodo po toliko narazen
    y0 = []
    # Podatke za trening in validacijo smo vzeli iz zacetka generiranih podatkov,
    # podatke za test pa bomo iz njihovega konca
    for zacetni in range(1,Ntest+1):
        y0.append(data[-zacetni*dn - m: -zacetni*dn])

    # prvih m elementov je seveda enakih vhodni resnici
    # (zato jih pri risanju spustimo!)
    data_lorenz_NN[:, 0:m, :] = y0
    # Dejanska napoved
    i = 0
    t = dt
    
    while t < tmax:
        norm_input = scaler(data_lorenz_NN[:, i: i + m, :])
        norm_output = model.predict(norm_input, batch_size=Ntest, verbose=0)
        
        if diff==True:
            norm_output = norm_input[:,m-1,:] + norm_output 
        
        new_state = scaler(norm_output, inverse=True)
        
        data_lorenz_NN[:, i + m, :] = new_state.copy()
    
        t += dt
        i += 1
    
    # Pred analizo rezultatov pobrisemo prvih m stanj iz seznama,
    # saj so ta zgolj prepisana resnica
    data_lorenz_NN = data_lorenz_NN[:, m:, :]
    return data_lorenz_NN

def predict_mtm(model, tend, k):
    tmax = tend # kako dolgo simulacijo pozenemo (v cas. enotah L63 sistema)
    dt = 0.01 # mora biti enak dt za pri treniranju!
    n = int(tmax/dt) + 1 # st. korakov v simulaciji
    m = k # dolzina vhoda pred napovedovanjem
    Ntest = 1000 # na koliko razlicnih zacetnih pogojih testiramo
    data_lorenz_NN = np.zeros((Ntest, m + n-1, 3)) # sem shranjujemo
    dn = int(1/dt) # zacetni pogoji v test setu naj bodo po toliko narazen
    y0 = []
    # Podatke za trening in validacijo smo vzeli iz zacetka generiranih podatkov,
    # podatke za test pa bomo iz njihovega konca
    for zacetni in range(1,Ntest+1):
        y0.append(data[-zacetni*dn - m: -zacetni*dn])

    # prvih m elementov je seveda enakih vhodni resnici
    # (zato jih pri risanju spustimo!)
    data_lorenz_NN[:, 0:m, :] = y0
    # Dejanska napoved
    i = 0
    t = dt
    while t <= tmax:
        norm_input = scaler(data_lorenz_NN[:, i: i + m, :])
        norm_output = model.predict(norm_input, batch_size=Ntest, verbose=0)
      
        new_state = scaler(norm_output, inverse=True)
        
        data_lorenz_NN[:,i+m: i + 2*m, :] = new_state.copy()
        t += m*dt
        i += m
    
    
    # Pred analizo rezultatov pobrisemo prvih m stanj iz seznama,
    # saj so ta zgolj prepisana resnica
    data_lorenz_NN = data_lorenz_NN[:, m:, :]
    return data_lorenz_NN

def calceE_t(predict_data, real_data, k):
    dt = 0.01
    dn = int(1/dt)
    Ntest = 1000

    Nt = len(predict_data[0,:,0])
    #t = np.arange(0,tend, Nt )
    Et = np.zeros(Nt)

    for tj in range(Nt):
        Ninitial = len(predict_data[:,0,0])
       
        lsum = 0
        fst = 0
        for i in range(Ninitial):

            lsum +=  np.sum((predict_data[i,tj,:]-real_data[i,tj,:])**2)
            
            
    
        Et[tj] = np.sqrt((1.0/Ninitial) * lsum)

    return Et

def integrate_all(init_cond, t_end, dt=0.01, params = (10,28,8/3) ):
    nt = int(t_end/dt) + 1
    solved = np.zeros((len(init_cond),nt,3 ))
    solver_ATM = ode(lorenz63).set_integrator('dopri5')
    for init in range(len(init_cond)):
        # Nastavimo integrator
        t=0.
        solver_ATM.set_f_params(params)
        solver_ATM.set_initial_value(init_cond[init,:], 0.)
        i = 0
        while t < t_end:
            y_ATM = solver_ATM.integrate(t+dt)
            solved[init,i,:] = y_ATM[:]
            i += 1
            t += dt
    return solved


def plot_et_k(models,t_end, dt=0.01):
    k = [4,32,4,32]
    dn = int(1/dt)
    cmap = plt.get_cmap("tab10")
    linestyle = ['-', '--', '-.', ':']
    init_con_ind = [-i*dn-1 for i in range(1,1000+1)]
    predicted = predict_mtm(models[0], t_end, k[0])
    solved = integrate_all(data[init_con_ind], t_end, dt=0.01, params = (10,28,8/3) )
    res = calceE_t(predicted, solved,k[0])
    plt.plot(np.linspace(0,t_end,len(res)), res, linewidth=1, label=f'LSTM_mtm,{linestyle[0].strip()}',c=cmap(0), linestyle=linestyle[0])
    predicted = predict_mtm(models[1], t_end, k[1])
    solved = integrate_all(data[init_con_ind], t_end, dt=0.01, params = (10,28,8/3) )
    res = calceE_t(predicted, solved,k[1])
    plt.plot(np.linspace(0,t_end,len(res)), res, linewidth=1, label=f'RNN_mtm,{linestyle[1].strip()}',c=cmap(1), linestyle=linestyle[1])
    predicted = predict_mto(models[2], t_end, k[2])
    solved = integrate_all(data[init_con_ind], t_end, dt=0.01, params = (10,28,8/3) )
    res = calceE_t(predicted, solved,k[2])
    plt.plot(np.linspace(0,t_end,len(res)), res, linewidth=1, label=f'RNN_mto,{linestyle[2].strip()}',c=cmap(2), linestyle=linestyle[2])
    predicted = predict_mto(models[3], t_end, k[3])
    solved = integrate_all(data[init_con_ind], t_end, dt=0.01, params = (10,28,8/3) )
    res = calceE_t(predicted, solved,k[3])
    plt.plot(np.linspace(0,t_end,len(res)), res, linewidth=1, label=f'LSTM_mto,{linestyle[3].strip()}',c=cmap(3), linestyle=linestyle[3])

    #plt.xlim([0,3])
    plt.xlabel('t')
    plt.ylabel('E(t)')
    plt.legend()
    plt.title('')
    plt.savefig('./plots/optimal_model_et.png')


#def plot_Et_n_dynm(predict, real_data):
    # Generate some sample data
    tau_values = np.linspace(0, 10, 100)
    E_values = np.sin(tau_values)

    t_values = np.linspace(0, 10, 100)
    x_values = np.sin(t_values)
    y_values = np.cos(t_values)
    z_values = np.sin(2 * t_values)

    # Assuming you have predicted values for x, y, and z
    predicted_x_values = np.cos(t_values)
    predicted_y_values = np.sin(t_values)
    predicted_z_values = np.cos(2 * t_values)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot E(tau)
    axs[0].plot(tau_values, E_values, label='E(tau)')
    axs[0].set_title('Plot of E(tau)')
    axs[0].set_xlabel('tau')
    axs[0].set_ylabel('E')
    axs[0].legend()

    # Plot x(t), y(t), z(t) and their predicted values
    axs[1].plot(t_values, x_values, label='x(t)')
    axs[1].plot(t_values, y_values, label='y(t)')
    axs[1].plot(t_values, z_values, label='z(t)')
    axs[1].plot(t_values, predicted_x_values, linestyle='dashed', label='Predicted x(t)')
    axs[1].plot(t_values, predicted_y_values, linestyle='dashed', label='Predicted y(t)')
    axs[1].plot(t_values, predicted_z_values, linestyle='dashed', label='Predicted z(t)')
    axs[1].set_title('Components x(t), y(t), z(t) and Predicted Components')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Values')
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

if __name__=='__main__':

    #dt = 0.01
    #tmax = 10000.
    params = (10,28,8/3)
    data = np.loadtxt('./intdata.txt', delimiter=',')
    normalized_data = scaler(data)
    # EARLY STOP IS OUT!

    #hiddenstate_dep([[4],[8],[16],[32],[64]], ['tanh'],"mean_squared_error", 'adam')
    #k_dep([4,8,16,32,64], [16], ['tanh'],"mean_squared_error", 'adam')
    #actdep_dep([['tanh'], ['sigmoid'],['relu'], ['softplus'] ], [16],"mean_squared_error", 'adam' )  

    #layer_num_dep([[32],[32,32],[32,32,32],[32,32,32,32]],[['tanh'],['tanh','tanh'],
    #              ['tanh','tanh','tanh'],['tanh','tanh','tanh','tanh'] ], "mean_squared_error", 'adam')

    #grid_search("tanh", "mean_squared_error", optimizers.Adam,k=10,mtm=True,epoch=100,batchs=32,filename="grid_sh_mtm")
    #plot_min_scores("./models/grid_sh_mtm", "gs_rnn_mtm.png")

    

    #WHEN MAKING PREDICTION MODEL turn on EARLYSTOPPING

    
    #train rnn mtm with different k: , will leave 10000 pst to test
    #optimal lstm mto: H16,k32,tanh
    #optimal mtm RNN: H16, k32, tanh, 3 layer
    #optimal mtm lstm: H32, k?, tanh, 4 layer
    #kk=32
    #Xt,Yt,Xval,Yval = trainnval_set(kk, 200000,50000, mtm=True)
    #lstm(Xt, Yt, Xval, Yval, [32,32,32,32], ['tanh','tanh','tanh','tanh'],'mean_squared_error','adam',epoch=200,mtm=True, k=kk, savename=f'lstm_mtm_k={kk}')
    #optimal lstm mto: H16,k32,tanh, 2 layers
    #Xt,Yt,Xval,Yval = trainnval_set_diff(32, 600000,100000)
    #lstm(Xt, Yt, Xval, Yval, [16,16], ['tanh','tanh'],'mean_squared_error','adam',epoch=100, savename=f'lstm_mto_optimal_diff_100.keras')
    
    new_model = tf.keras.models.load_model('./models/lstm_mto_optimal_diff_100.keras.keras')
    new_model1 = tf.keras.models.load_model('./models/lstm_mto_optimal.keras')
    
    k=32
    t_end=10.24
    predicted = predict_mto(new_model, t_end, k, diff=True)
    
    
    dn = int(1/0.01)
    init_con_ind = [-i*dn-1 for i in range(1,1000+1)]
    
    
    solved = integrate_all(data[init_con_ind], t_end, dt=0.01, params = (10,28,8/3) )
    #print(predicted[0,:,:], solved[0,:,:])

    #f = plot_3d_curve_with_subplots(solved[0,:,:],predicted[0,:,:], 10.24)
    #f.savefig('./plots/lstm_mto_final_predict_diff.png')
    
    
    res = calceE_t(predicted, solved,k)
    plt.clf()

    predicted1 = predict_mto(new_model1, t_end, k, diff=False)
    res1 = calceE_t(predicted1, solved,k)
    plt.plot(np.linspace(0,t_end,len(res)), res, linewidth=1, label='lstm_mto_diff')
    plt.plot(np.linspace(0,t_end,len(res1)), res1, linewidth=1, label='lstm_mto')
    plt.xlabel('t')
    plt.ylabel('E(t)')
    #plt.xlim([0,3])
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/optimal_lstm_et_diff.png')
    
    #models= []
    #kar=[4,8,16]
    #for i in range(3):
    #    models.append(tf.keras.models.load_model(f'./models/lstm_mtm_k={kar[i]}.keras'))
    
    #plot_et_k(models,10.24, dt=0.01)


    #WHEN MAKING PREDICTION MODEL turn on EARLYSTOPPING

    
    #train rnn mtm with different k: , will leave 10000 pst to test
    #optimal lstm mto: H16,k32,tanh, 2 layers
    #optimal mtm RNN: H16, k32, tanh, 3 layer
    #optimal mto RNN: H16, k4 is better! or 32?, tanh, 2 layer
    #optimal mtm lstm: H32, k4, tanh, 4 layer

    # on 600k and 100k train and val
    #Xt,Yt,Xval,Yval = trainnval_set(4, 100000,10000, mtm=False)
    #SNN(Xt, Yt, Xval, Yval, [16,16], ['tanh','tanh'],'mean_squared_error','adam',epoch=100,mtm=False, k=1, savename=f'rnn_mto_optimal_2_k4_small')

    #models = [tf.keras.models.load_model('./models/lstm_mtm_optimal.keras'),tf.keras.models.load_model('./models/rnn_mtm_optimal.keras'),
    #          tf.keras.models.load_model('./models/rnn_mto_optimal_2_k4.keras'), tf.keras.models.load_model('./models/lstm_mto_optimal.keras')]
    
    #plot_et_k(models, 10.24)