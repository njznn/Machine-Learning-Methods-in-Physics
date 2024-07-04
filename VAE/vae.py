import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import QuantileTransformer
from sklearn import metrics
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score





def plot_label_clusters(z, y_train):
    # Display a 2D plot of the digit classes in the latent space

    plt.figure(figsize=(6, 6))

    # Use a colormap with 10 distinct colors
    cmap = plt.cm.get_cmap('tab10')
    for label in range(0, 10):
        indices = np.where(y_train == label)
        plt.scatter(z[indices, 0], z[indices, 1],s=3, color=cmap(label), label=f'{label}')

    plt.xlabel("z0")
    legend_points = [plt.scatter([], [], s=20, color=cmap(label), label=f'{label}') for label in range(10)]
    plt.legend(handles=legend_points, title='Števka', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.ylabel("z1")
    plt.savefig("./plots/latent_space_dist.png")

def plot_latent_space(decoder, n=20, figsize=15):
    # display a n*n 2D manifold of digits
    decoder.load_weights('./models/dec_weights100.h5')
    digit_size = 28
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z0")
    plt.ylabel("z1")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig('./plots/from_latent.png')


def plot_hists(data, labels, bins=100):
    
    fig, axes = plt.subplots(1, 4, figsize=(9, 3))
    komp_values = [0, 1, 2, 3]
    titles = [r'$m_{j1}$', r'$(\tau_2 / \tau_1)_1$', r'$(\tau_3 / \tau_2)_1$', r'$m_{d_{j1}}$']
    indbck = np.where(labels == 0)[0]
    indsig = np.where(labels == 1)[0]
    print(indsig)

    for i, komp in enumerate(komp_values):
        # Choose the appropriate subplot
        # Create histograms
        axes[i].hist(data[:, komp], bins=bins, alpha=1, label='S+N', histtype='bar', color='black')
        axes[i].hist(data[indbck, komp], bins=bins, alpha=1, label='N',  histtype='bar', color='blue')
        axes[i].hist(data[indsig, komp], bins=bins, alpha=1, label='S',  histtype='bar', color='red')
        
        #axes[i].set_yscale('log')


        # Add labels and title
        axes[i].set_xlabel(titles[i])
        axes[i].set_ylabel('N')
        #formatter = ticker.ScalarFormatter(useMathText=True)
        #formatter.set_powerlimits((0, 0))  # Do not use scientific notation for small values
        #axes[i].yaxis.set_major_formatter(formatter)
        axes[i].tick_params(axis='y', which='both', labelsize=8)
        if i==3:
            axes[i].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig('./plots/histplot_reconst.png')

def plot_histogram_grid(matrices,labels=0, bins=100):
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['font.size'] = 18
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    lab = [r'$z_{mean}$', r'$z_{logvar}$', r'$z^2_{mean}$']
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))  # Do not use scientific notation for small values
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]

            data = matrices[i,j, :]
            if j==2:
                data = matrices[i,j,:]**2
                #ax.set_yscale('log')
            ax.hist(data, bins=bins, alpha=0.8,  color='red')
            
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(axis='y', which='both', labelsize=8)
            #ax.hist(data[np.where(labels==0)], bins=bins, alpha=0.8, density=True,color='blue')
            #ax.hist(data[np.where(labels==1)], bins=bins, alpha=0.8, density=True,color='red')
            if (i==2):
                ax.set_xlabel(lab[j])
            

    legend_handles = [Patch(color='blue', alpha=0.8), Patch(color='red', alpha=0.8)]
    #fig.legend(legend_handles, ['N', 'S'], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize='medium')
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig('./plots/histmatrix_bb_new.png')

def plot_auc_curve(y_true, y_scores_list):

    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)
    lab = [r'$z_{mean}$', r'$z_{logvar}$', r'$z^2_{mean}$']
    for j in range(3):
        ax = axes[j]
        for i in range(len(y_scores_list)):
            if j==2:
                fpr, tpr, _ = metrics.roc_curve(y_true, y_scores_list[i][0]**2)
                roc_auc = metrics.roc_auc_score(y_true, y_scores_list[i][0]**2)
            else:
                fpr, tpr, _ = metrics.roc_curve(y_true, y_scores_list[i][j])
                roc_auc = metrics.roc_auc_score(y_true, y_scores_list[i][j])

            
            ax.plot( fpr,tpr, label=f'AUC{i}'+'= {:.2f}'.format(roc_auc))
            #ax.set_yscale('log')
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax.set_xlabel('FPR')
            if j==0:
                ax.set_ylabel('TPR')
            ax.legend(loc='lower right')
        ax.set_title(lab[j])

    plt.savefig('./plots/auc_L.png')

def CrystalBallopt(x, A, mCB, sCB):
    #optimal:
    aL = 1.4709725540155203
    nL = 3.7533888887587894
    aR = 1.6714153694042544
    nR = 9.654308950547579
    
    condlist = [
        (x - mCB) / sCB <= -aL,
        (x - mCB) / sCB >= aR,
    ]
    funclist = [
        lambda x: A
        * (nL / np.abs(aL)) ** nL
        * np.exp(-(aL**2) / 2)
        * (nL / np.abs(aL) - np.abs(aL) - (x - mCB) / sCB) ** (-nL),
        lambda x: A
        * (nR / np.abs(aR)) ** nR
        * np.exp(-(aR**2) / 2)
        * (nR / np.abs(aR) - np.abs(aR) + (x - mCB) / sCB) ** (-nR),
        lambda x: A * np.exp(-((x - mCB) ** 2) / (2 * sCB**2)),
    ]
    return np.piecewise(x, condlist, funclist)

def CrystalBallopt2(x, A, mCB, sCB):
    #optimal:
    aL = 1.6
    nL = 3.7533888887587894
    aR = 0.2
    nR = 1.6
    
    condlist = [
        (x - mCB) / sCB <= -aL,
        (x - mCB) / sCB >= aR,
    ]
    funclist = [
        lambda x: A
        * (nL / np.abs(aL)) ** nL
        * np.exp(-(aL**2) / 2)
        * (nL / np.abs(aL) - np.abs(aL) - (x - mCB) / sCB) ** (-nL),
        lambda x: A
        * (nR / np.abs(aR)) ** nR
        * np.exp(-(aR**2) / 2)
        * (nR / np.abs(aR) - np.abs(aR) + (x - mCB) / sCB) ** (-nR),
        lambda x: A * np.exp(-((x - mCB) ** 2) / (2 * sCB**2)),
    ]
    return np.piecewise(x, condlist, funclist)


def extr_signal(zlogvar, xdata,invmass,  numdata):
    
    zlogvar = np.ravel(zlogvar)
    zlogvar = zlogvar**2
    sortedind = np.argsort(zlogvar)
    sortedind = sortedind[::-1] #largest indices, as signal is on right
    mj1 = xdata[sortedind, 0]
    
    mj2 = xdata[sortedind,4]
    minv = invmass[sortedind]
    return np.array([minv[:numdata], mj1[:numdata], mj2[:numdata] ])

def plot_histograms(minv, mj1, mj2):
    fig, axs = plt.subplots(1, 3, figsize=(9, 4))

    # Plot histogram for minv
    #minv = minv[minv>= 3300]
    #minv = minv[minv<=4000]
    axs[0].hist(minv, bins=40, color='blue', alpha=0.8)
    bin_values, bins = np.histogram(minv, bins=40)
    axs[0].set_ylabel('N=1000')
    axs[0].set_xlabel('m[GeV]')
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers
    
# Calculate bin errors (if you have uncertainties associated with each bin)
    bin_errors = np.sqrt(bin_values)
    cbopt = lambda x, A, mCB, sCB: CrystalBallopt(x,A,mCB, sCB)
    # Fit
    popt, pcov = curve_fit(
        cbopt,
        bin_centers,
        bin_values, 
        
        p0=[40, 3500, 50],
    )
    perr = np.sqrt(np.diag(pcov))
    A, mCB, sCB= popt
    alpcopt = lambda x,a:  a*cbopt(x, A, mCB, sCB)
    popt, pcov = curve_fit(
        alpcopt,
        bin_centers,
        bin_values, 
        
        p0=[50],
    )
    a= popt
    
    axs[0].plot(bin_centers, cbopt(bin_centers, A, mCB, sCB), 'r-')
    axs[0].set_title(r'$m_Z=$'+f'{round(mCB)}'+' GeV' + r'$\pm$' +f'{round(sCB,0)}' + r' GeV')
    
    # Plot histogram for mj1

    bin_values1, bins1, _ = axs[1].hist(mj1, bins=40, color='blue', alpha=0.8)
    bin_centers1 = (bins1[:-1] + bins1[1:]) / 2  # Compute bin centers
    bin_errors1 = np.sqrt(bin_values1)
    cbopt = lambda x, A, mCB, sCB: CrystalBallopt(x,A,mCB, sCB)
    # Fit
    popt1, pcov1 = curve_fit(
        cbopt,
        bin_centers1,
        bin_values1, 
        p0=[60, 680, 40],
    )
    perr = np.sqrt(np.diag(pcov1))
    A1, mCB1, sCB1= popt1
    axs[1].plot(bin_centers1, cbopt(bin_centers1, A1, mCB1, sCB1), 'r-')
    axs[1].set_title(r'$m_X=$'+f'{round(mCB1)}'+' GeV' + r'$\pm$' +f'{round(np.abs(sCB1),0)}' + r' GeV')
    axs[1].set_xlabel('m[GeV]')
    #axs[1].set_xlim([0,900])
    
    
    axs[2].hist(mj2, bins=40, color='blue', alpha=0.8)
    bin_values2, bins2 = np.histogram(mj2, bins=40)
    bin_centers2 = (bins2[:-1] + bins2[1:]) / 2  # Compute bin centers
    bin_errors2 = np.sqrt(bin_values2)
    cbopt = lambda x, A, mCB, sCB: CrystalBallopt(x,A,mCB, sCB)
    # Fit
    popt2, pcov2 = curve_fit(
        cbopt,
        bin_centers2,
        bin_values2, 
        p0=[110, 400, 20],
    )
    perr = np.sqrt(np.diag(pcov2))
    A2, mCB2, sCB2= popt2
    axs[2].plot(bin_centers2, cbopt(bin_centers2, A2, mCB2, sCB2), 'r-')
    axs[2].set_title(r'$m_Y=$'+f'{round(mCB2)}'+' GeV' + r'$\pm$' +f'{round(sCB2,0)}' + r' GeV')
    axs[2].set_xlabel('m[GeV]')
    
    
    plt.savefig('./plots/bla.png')


def calculate_auc(predictions, labels):
    auc = roc_auc_score(labels, predictions)
    return auc


if __name__=='__main__':
    #(x_train, y_train), _ = mnist.load_data()
    #x_train = x_train.astype('float32')/255.0
    #x_train = x_train.reshape((len( x_train ), np.prod(x_train.shape[1:])))
    #original_dim = np.prod(x_train.shape[1:])
    x_train = np.load('./Podatki_7naloga/blackbox_jetobs.npy')
    x_train = x_train.astype('float32')
    #x_labels = np.load('./Podatki_7naloga/lhco_H_labels.npy')
    #x_invmass = np.load('./Podatki_7naloga/blackbox_invmass.npy')
    
    f_scaler = QuantileTransformer(output_distribution ='uniform')
    x_train_transformed = f_scaler.fit_transform(x_train)
    #x_train_reconstructed = f_scaler.inverse_transform(x_train_transformed )
    original_dim = np.prod(x_train_transformed.shape[1:])

    
    #enkoder
    inputs = keras.Input(shape =(original_dim ,))
    h1 = keras.layers.Dense(64, activation='selu')(inputs)
    h2 = keras.layers.Dense(64, activation='selu')(h1)
    h3 = keras.layers.Dense(64, activation='selu')(h2)
    z_mean = keras.layers.Dense(1, activation=None)(h3)
    z_log_var = keras.layers.Dense(1, activation=None)(h3)

    def sampling(args):
        z_mean,z_log_var = args
        epsilon = K.random_normal(shape=( K.shape(z_mean)[0], 1), mean =0.0,
                                stddev =1.0 )
        return z_mean + K.exp(0.5* z_log_var )* epsilon


    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean , z_log_var , z], name ='encoder')

    #dekoder:
    latent_inputs = keras.Input(shape=(1,),name='z_sampling')
    x = keras.layers.Dense(64, activation ='selu')(latent_inputs)
    x1 = keras.layers.Dense(64, activation ='selu')(x)
    x2 = keras.layers.Dense(64, activation ='selu')(x1)
    outputs = keras.layers.Dense(original_dim)(x2)
    decoder = keras.Model(latent_inputs , outputs , name ='decoder')
    outputs = decoder(encoder(inputs)[2])

    
    vae = keras.Model(inputs , outputs , name ='vae')
    rec_loss = keras.losses.mean_squared_error(inputs,outputs) # *x, where x = 1/beta
    rec_loss *= 5000 #this is 1/beta
    kl_loss = -0.5*K.sum( 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(rec_loss + kl_loss)
    vae.add_loss(vae_loss)
    #opt = keras.optimizers.Adadelta()
    vae.compile(optimizer = 'adadelta')
    batch_size = 1000
    
    #hist = vae.fit(x_train_transformed,x_train_transformed,epochs=100,batch_size=batch_size)
    #encoder.save_weights('./models/sm_enc_bb_lr001_weights100.h5')
    #decoder.save_weights('./models/sm_dec_bb_lr001_weights100.h5')
    
    
    #load weights:
    encoder.load_weights('./models/sm_enc_bb1_weights100.h5')
    #decoder.load_weights('./models/dec_weights100.h5')
    res1 = encoder.predict(x_train_transformed)
    encoder.load_weights('./models/sm_enc_bb2_weights100.h5')
    #decoder.load_weights('./models/dec_weights100.h5')
    res2 = encoder.predict(x_train_transformed)
    
    #decoder.load_weights('./models/sm_dec_H2_weights100.h5')
    #dec2 = decoder.predict(res2[0])
    #xpredict = f_scaler.inverse_transform(dec2) 
   
    #plot_hists(xpredict, x_labels)
    encoder.load_weights('./models/sm_enc_bb3_weights100.h5')
    #decoder.load_weights('./models/dec_weights100.h5')
    res3 = encoder.predict(x_train_transformed)
    
    
    mat = np.array([res1, res2, res3])
    #auc = metrics.roc_auc_score(x_labels, res1[0])
    #fpr, tpr,thresholds = metrics.roc_curve(x_labels, res1[0])
    #plot_auc_curve(x_labels, [res1, res2, res3])

    plot_histogram_grid(mat)
    #plot_latent_space(decoder)
    #plot_label_clusters(z_mean, y_train)
    #plot_hists(x_train_reconstructed, x_labels)
    
    #res = extr_signal(res2[0], x_train, x_invmass,2000)
    
    #plot_histograms(res[0], res[1], res[2])
    
    """
    auc_z_mean_values = []
    auc_z_log_var_values = []
    auc_z_mean_squared_values = []
    epochs=100
    for epoch in range(epochs):
        # Fit the VAE model
        hist = vae.fit(x_train_transformed, x_train_transformed, epochs=1, batch_size=batch_size, verbose=1)

        # Get encoder predictions for the training set
        z_mean_pred= encoder.predict(x_train_transformed)[0]
        z_mean_pred = -z_mean_pred
        # Calculate AUC for z_mean
        auc_z_mean = calculate_auc(np.ravel(z_mean_pred), x_labels)  # Assuming you have labels for your training set
        auc_z_mean_values.append(1-auc_z_mean)

        # Calculate AUC for z_log_var
        z_log_var_pred= encoder.predict(x_train_transformed)[1]
        auc_z_log_var = calculate_auc(z_log_var_pred.flatten(), x_labels)
        auc_z_log_var_values.append(auc_z_log_var)

        # Calculate AUC for z_mean^2
        z_mean_squared_pred = np.square(z_mean_pred)
        auc_z_mean_squared = calculate_auc(z_mean_squared_pred.flatten(), x_labels)
        auc_z_mean_squared_values.append(auc_z_mean_squared)

        print(f'Epoch {epoch + 1}/{epochs} - AUC z_mean: {auc_z_mean:.4f}, AUC z_log_var: {auc_z_log_var:.4f}, AUC z_mean^2: {auc_z_mean_squared:.4f}')

    # Plot AUC values over epochs
    plt.figure(figsize=(6, 6))
    if auc_z_mean_values[-1] < 0.5:
        auc_z_mean_values = 1-auc_z_mean_values
        auc_z_mean_squared_values = 1- auc_z_mean_squared_values
    plt.plot(range(1, epochs + 1), auc_z_mean_values, label=r'$AUC z_{mean}$')
    plt.plot(range(1, epochs + 1), auc_z_log_var_values, label=r'$AUC z_{log var}$')
    plt.plot(range(1, epochs + 1), auc_z_mean_squared_values, label=r'$AUC z_{mean}^2$', color='black', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.savefig('./plots/bla.png')
    """
    

