from cProfile import label
from email.mime.application import MIMEApplication
import numpy as np
import matplotlib.pyplot as plt
from numpy import kaiser, linalg as LA, sort
from sklearn.cluster import KMeans
from sympy import cosine_transform, sequence
from matplotlib.colors import Normalize
from pyjet import DTYPE_PTEPM, cluster
from numba import njit
import time
from scipy.optimize import curve_fit





def KMeanss(X, n_clusters, max_iter=100):
    # Randomly initialize centroids
    
    idx = np.random.choice(X.shape[0], n_clusters, replace=False)
    centroids = X[idx, :]
    for _ in range(max_iter):
        # Assign clusters based on closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

        # Check for convergence
        if np.allclose(new_centroids, centroids):
            break
        
        if _ == max_iter:
            "Max iteration exceded!"
            break


        centroids = new_centroids

    return centroids, labels

def kmeans_multiple_runs(X, n_clusters, n_runs):
    centroids_list = []

    for _ in range(n_runs):
        centroids, _ = KMeanss(X, n_clusters)  # Use your custom KMeanss function
        centroids_list.append(centroids)

    return np.array(centroids_list)

def plot_2d_histogram(centroids):
    n_runs, n_clusters, n_features = centroids.shape

    plt.figure(figsize=(8, 6))

    # Plot each set of centroids from different runs with different colors
    for i in range(n_runs):
        flattened_centroids = centroids[i].reshape(-1, n_features)
        plt.hist2d(flattened_centroids[:, 0], flattened_centroids[:, 1], bins=30, cmap='viridis', alpha=0.5, norm=Normalize(vmin=0, vmax=n_runs))

    plt.colorbar(label='Number of occurrences')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Histogram of Centroid Positions with Different Runs')
    plt.tight_layout()
    plt.savefig("./plots/centr.png")

def plotgauss(data):

    column1 = data[:, 0]
    column2 = data[:, 1]
    
    # Create a new figure for 3D plotting with smaller size
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D histogram with finer structure (increased number of bins)
    hist, xedges, yedges = np.histogram2d(column1, column2, bins=50)
    
    # Construct arrays for the anchor positions of the bars
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    
    # Construct arrays with the dimensions for the bars
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    
    # Plot 3D bar diagram
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', cmap='viridis')
    
    # Set labels and title
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r'$\phi$')
    ax.set_zlabel('N')
    
    # Display the plot
    plt.tight_layout()
    
    plt.savefig("./plots/gauss.png")

def plot_kmeans_results(X, labels, centroids, hmass, maxptind):
    """
    Plot the results of K-means clustering.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), data points.
    - labels: numpy array of shape (n_samples,), cluster labels.
    - centroids: numpy array of shape (n_clusters, n_features), centroid positions.
    """
    plt.figure(figsize=(6, 4))

    # Plot data points with color-coded clusters
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        if np.any(label==maxptind):
            
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'Cluster {label + 1}')
            plt.scatter(centroids[label, 0], centroids[label, 1], c='black', marker='x', s=100)
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color='green')


    # Plot centroids
    #plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')

    plt.xlabel(r"$\eta$")
    plt.ylabel(r'$\phi$')
    plt.title(r'$m_{H}=$' + f'{round(hmass, 2)}'+' MeV, K=10')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/clhiggs10.png")
    
def convert_to_npy(data):
        datamat = np.zeros((len(data), 4))
        for i in range(len(data)):
            for j in range(4):
                datamat[i,j] = data[i][j]
    
        return datamat



class Higgs:
    def __init__(self, data):
        self.data=data
        self.pt = data[:,0]
        self.eta = data[:,1]
        self.fi = data[:,2]
        self.mass = data[:,3]
        self.labels = None
        self.clusters=0
        self.hmass = 0
        self.maxptind = None
        self.times = np.array([])

    def momentums(self):
        px = self.pt*np.cos(self.fi)
        py = self.pt*np.sin(self.fi)
        pz = self.pt*np.sinh(self.eta)
        pab = self.pt*np.cosh(self.eta)

        return np.array([pab, px, py, pz]).T
    
    def momentumss(self, cetv):
        px = cetv[0]*np.cos(cetv[2])
        py = cetv[0]*np.sin(cetv[2])
        pz = cetv[0]*np.sinh(cetv[1])
        pab = cetv[0]*np.cosh(cetv[1])

        return np.array([pab, px, py, pz]).T


    def KMeanss(self, n_clusters, max_iter=300):
        # Randomly initialize centroids
        X = np.array([self.eta, self.fi]).T
        
        idx = np.random.choice(X.shape[0], n_clusters, replace=False)
      
        centroids = X[idx, :]
        for _ in range(max_iter):
            # Assign clusters based on closest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

            # Check for convergence
            if np.allclose(new_centroids, centroids):
                break
            
            if _ == (max_iter-1):
                print("Max iteration exceded!")
                break

            centroids = new_centroids
        self.labels = labels
        self.clusters = n_clusters

        return centroids
    
    def make_hist(self, niter, K, method):
        higgs_mass_values = []
    
        for i in range(niter):
            res.KMeanss(K)
            res.calc_higgs(method)
            higgs_mass_values.append(res.hmass)

        filtered_values = [x for x in higgs_mass_values if x <= 200]  # Filter out values greater than 200

        average_mass = np.mean(filtered_values)  # Compute average
        std_deviation = np.std(filtered_values)
        print(average_mass)

        plt.figure(figsize=(6, 4))

        plt.hist(filtered_values, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel(r'$m_{H}$')
        plt.yscale('log')
        plt.ylabel('N')
        plt.title('K=13,'+ " " +r'$m_{H} =$'+f'{round(average_mass, 1)}'+ r'$\pm$' + f'{round(std_deviation, 1)}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./plots/higgs_mass_histogram13.png")
        
    def infrared_safety_kmeans(self, npout, K):
        higgs_mass_values = []
        for i in range(npout-K):
            self.KMeanss(K)
            self.calc_higgs('Kmeans')
            higgs_mass_values.append(self.hmass)
            minind = np.where(self.pt==np.min(self.pt))
            self.pt = np.delete(self.pt, minind)
            self.fi = np.delete(self.fi, minind)
            self.eta = np.delete(self.eta, minind)

        return higgs_mass_values
    
    def infrared_safety_high(self, npout, R):
        higgs_mass_values = []
        exctime = np.array([])
        for i in range(npout):
            
            self.calc_higgs('high', R)
            higgs_mass_values.append(self.hmass)
            minind = np.where(self.pt==np.min(self.pt))
            self.pt = np.delete(self.pt, minind)
            self.fi = np.delete(self.fi, minind)
            self.eta = np.delete(self.eta, minind)

        return higgs_mass_values
    
    def infrared_safety_pyjet(self, npout, R):
        higgs_mass_values = []
        for i in range(npout):
            print(i)
            self.calc_higgs('pyjet', R)
            higgs_mass_values.append(self.hmass)
            minind = np.where(self.pt==np.min(self.pt))
            self.pt = np.delete(self.pt, minind)
            self.fi = np.delete(self.fi, minind)
            self.eta = np.delete(self.eta, minind)

        return higgs_mass_values
    
    def hierarhic_group(self, R, p):
        protolist = np.array([self.pt, self.eta, self.fi, np.zeros(len(self.pt))]).T
        rlist = []
        while protolist.shape[0] != 0:
            di = protolist[:,0]**(2*p)
            N=len(protolist)
            dij = []
            dijmin = np.inf
            min_i = 0
            min_j=0
            for i in range(N-1):
                dtemp = np.array([])
                for j in range(i+1, N):
                    dd = np.min([di[i], di[j]])*((protolist[i,1]-protolist[j,1])**2 + (protolist[i,2]-protolist[j,2])**2)/R**2
                    if dd< dijmin:
                        dijmin = dd
                        min_i = i
                        min_j=j
                    dtemp = np.append(dtemp, dd)
                dij.append(dtemp)
            
            
            minarr = [np.min(di), dijmin]

            if minarr[0]> minarr[1]:
                ptk = protolist[min_i,0] + protolist[min_j,0]
                etak = (protolist[min_i,0]*protolist[min_i,1] + 
                        protolist[min_j,0]*protolist[min_j,1])/ptk
                fik = (protolist[min_i,0]*protolist[min_i,2] + 
                        protolist[min_j,0]*protolist[min_j,2])/ptk
                protolist=np.delete(protolist, [min_i, min_j], axis=0)
                
                a=np.array([ptk, etak, fik, 0])
                protolist = np.vstack((protolist, a))

            else:
                ind_i = np.where(di==np.min(di))[0][0]
                rlist.append(protolist[ind_i,:])
                protolist = np.delete(protolist,ind_i, axis=0)
        
        
        return rlist
        
    def jetstonpy(self, jets):
        pt = np.array([jet.pt for jet in jets])
        eta = np.array([jet.eta for jet in jets])
        fi = np.array([jet.phi for jet in jets])
        m = np.array([jet.mass for jet in jets])
        
        
        pvec = np.array([pt, eta, fi, m]).T
        return pvec

    def calc_higgs(self, method, R=0.6, p=1):
        if method=='Kmeans':
            ptsum = [np.sum(self.pt[np.where(self.labels==i)]) for i in range(self.clusters)]
            
            maxptind = np.array([], dtype=int)
            maxptind = np.append(maxptind, np.where(ptsum==np.max(ptsum)))
            ptsum[maxptind[0]] = 0
            maxptind = np.append(maxptind, np.where(ptsum==np.max(ptsum)))
            self.maxptind = maxptind

            cetv = self.momentums()
            
            cetv1 = np.sum(cetv[np.where(self.labels==maxptind[0])], axis=0)
            cetv2 = np.sum(cetv[np.where(self.labels==maxptind[1])], axis=0)
            self.hmass = np.sqrt((LA.norm(cetv1[1:]) + LA.norm(cetv2[1:]))**2 - LA.norm(cetv1[1:]+cetv2[1:])**2)

        elif method=='high':
            start_time = time.time()  # Get the current time before executing the function
            realspl= self.hierarhic_group(R,p=1)
             # Execute the function with provided arguments
            end_time = time.time()  # Get the current time after executing the function
    
            execution_time = end_time - start_time 
            self.times = np.append(self.times,execution_time)
            realspl=convert_to_npy(realspl)
            
            maxptind = np.array([], dtype=int)
            
            maxptind = np.append(maxptind, np.where(realspl[:,0]==np.max(realspl[:,0])))
            mmax = realspl[maxptind[0],:].copy()
            realspl[maxptind[0],:] = np.zeros(4)
            maxptind = np.append(maxptind, np.where(realspl[:,0]==np.max(realspl[:,0])))
            self.maxptind = maxptind
            realspl[maxptind[0]] = mmax

            
            cetv1 = self.momentumss(realspl[maxptind[0], :])
            cetv2 = self.momentumss(realspl[maxptind[1], :])
            self.hmass = np.sqrt((LA.norm(cetv1[1:]) + LA.norm(cetv2[1:]))**2 - LA.norm(cetv1[1:]+cetv2[1:])**2)

        elif method=='pyjet':
            #data = np.array([self.pt, self.eta, self.fi, self.mass], dtype=DTYPE_PTEPM).T
            data = np.load('h_bb_sorted.npy', allow_pickle=True)
            print(data)
            event = np.array(data, dtype=DTYPE_PTEPM)
            sequence=cluster(event, ep=False, R=R, p=p)
            incl_jets= sequence.inclusive_jets()
            
            
            p1 = np.array([incl_jets[0].px,incl_jets[0].py,incl_jets[0].pz])
            p2 = np.array([incl_jets[1].px,incl_jets[1].py,incl_jets[1].pz])
            self.hmass = np.sqrt((incl_jets[0].e + incl_jets[1].e)**2 - LA.norm(p1+p2)**2)

            


def plot_higgs_mass_values(higgs_mass_values):

    x_values = np.arange(len(higgs_mass_values))

    # Plotting
    plt.figure(figsize=(6, 5))
    plt.scatter(x_values, higgs_mass_values, s=10, color='black')  # Plot the values
    plt.axhline(y=125.25, linestyle='--', label=r'125.25 GeV', c='black')  # Plot the constant line

    # Set labels and title
    plt.xlabel('N_dropout')
    plt.ylabel(r'$m_{H} \ [GeV]$')
    plt.title('R=0.6')
    #plt.ylim([0,200])
    plt.legend()  # Add legend

    # Show grid
    #plt.yscale('log')
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.savefig("./plots/higgs_inf_kf_pyjet.png")

def pyjet_higgs(event, R, p=1):
    sequence=cluster(event, ep=False, R=R, p=p)
    incl_jets= sequence.inclusive_jets()
    
    
    p1 = np.array([incl_jets[0].px,incl_jets[0].py,incl_jets[0].pz])
    p2 = np.array([incl_jets[1].px,incl_jets[1].py,incl_jets[1].pz])
    hmass = np.sqrt((incl_jets[0].e + incl_jets[1].e)**2 - LA.norm(p1+p2)**2)
    return hmass, incl_jets

def infrared_safety_pyjet(npout,R):
    data = np.load('h_bb_sorted.npy', allow_pickle=True)[0]
    event = np.array(data, dtype=DTYPE_PTEPM)
    higgs_mass_values = []
    for i in range(npout):

        # Cluster and find inclusive jets
        sequence = cluster(event, ep=False, R=R, p=1)
        incl_jets = sequence.inclusive_jets()
        
        
        
        
        # Recalculate the invariant mass using the two highest-pt jets
        p1 = np.array([incl_jets[0].px, incl_jets[0].py, incl_jets[0].pz])
        p2 = np.array([incl_jets[1].px, incl_jets[1].py, incl_jets[1].pz])
        
        hmass = np.sqrt((incl_jets[0].e + incl_jets[1].e)**2 - LA.norm(p1 + p2)**2)
        higgs_mass_values.append(hmass)
        pt = np.array([i[0] for i in event])
        minpt = np.argmin(pt)
        event = np.concatenate((event[:minpt], event[int(minpt+1):]))


    return higgs_mass_values

def calc_time_inf(npout):
    data = np.load('h_bb_sorted.npy', allow_pickle=True)[0]
    event = np.array(data, dtype=DTYPE_PTEPM)
    tdata = convert_to_npy(data)
    res = Higgs(tdata)
    higgs_mass_values = []
    timepyjet = np.array([])
    for i in range(npout):
        start_time = time.time() 
        # Cluster and find inclusive jets
        sequence = cluster(event, ep=False, R=0.6, p=1)
        incl_jets = sequence.inclusive_jets()
        
        # Recalculate the invariant mass using the two highest-pt jets
        p1 = np.array([incl_jets[0].px, incl_jets[0].py, incl_jets[0].pz])
        p2 = np.array([incl_jets[1].px, incl_jets[1].py, incl_jets[1].pz])
        
        hmass = np.sqrt((incl_jets[0].e + incl_jets[1].e)**2 - LA.norm(p1 + p2)**2)
        higgs_mass_values.append(hmass)
        pt = np.array([i[0] for i in event])
        minpt = np.argmin(pt)
        event = np.concatenate((event[:minpt], event[int(minpt+1):]))
        end_time = time.time()  # Get the current time after executing the function

    
        execution_time = end_time - start_time 
        timepyjet = np.append(timepyjet, execution_time)
    res.infrared_safety_high(npout,0.6)
    mytime = res.times
    return timepyjet, mytime

def plot_time(array1, array2):
    fig, ax = plt.subplots()
    
    # Plot arrays
    ax.plot(array1, label='my')
    ax.plot(array2, label='pyjet')
    ax.set_yscale('log')
    
    # Add labels and title
    ax.set_xlabel('N_dropout')
    ax.set_ylabel('t[s]')
    # Add legend
    ax.legend()
    ax.grid()
    
    # Show plot
    plt.savefig("./plots/higgs_time.png")

def plot_higgs_mass_histogram():
    # Step 1: Load the data
    data = np.load('h_bb_sorted.npy', allow_pickle=True)
    
    hmass = np.array([])

    
    # Step 2: Loop through each event and calculate the Higgs mass
    print(len(data))
    for i in range(len(data)):
        event = np.array(data[i], dtype=DTYPE_PTEPM)
        mass = pyjet_higgs(event, R=0.6)[0]
        if mass < 500:
            hmass = np.append(hmass, mass)
    
    
    # Step 3: Plot histogram
    bin_values, bins, _ = plt.hist(hmass, bins=70, alpha=0.7, color='blue', edgecolor='black')

    # Step 4: Add line for Higgs mass 125.25 GeV
    plt.axvline(x=125.25, color='red', linestyle='--', label='Higgs mass 125.25 GeV')
    

    print(len(hmass[hmass<500])/len(hmass))
    # Step 5: Calculate average and standard deviation
    average_mass = np.mean(hmass)
    std_deviation = np.std(hmass)

    # Step 6: Add lines for average mass ± standard deviation
    plt.axvline(x=average_mass, color='green', linestyle='--', label=r"$\langle m_{H} \rangle $" + f'= ({average_mass:.2f}' + r'$ \pm $' +f'{round(std_deviation, 2)})' +  ' GeV')
    plt.axvline(x=average_mass + std_deviation/2, color='orange', linestyle='--', label=r"$\langle m_{H} \rangle $" +r"$\pm$"+ f' \u03C3')
    plt.axvline(x=average_mass - std_deviation/2, color='orange', linestyle='--')

    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers

# Calculate bin errors (if you have uncertainties associated with each bin)
    bin_errors = np.sqrt(bin_values)
    cbopt = lambda x, A, mCB, sCB: CrystalBallopt(x,A,mCB, sCB)
    # Fit
    popt, pcov = curve_fit(
        cbopt,
        bin_centers,
        bin_values, 
        sigma=bin_errors,
        
        p0=[133, 124.5, 3.0],
    )
    perr = np.sqrt(np.diag(pcov))
    A, mCB, sCB= popt
    
    alpcopt = lambda x,a:  a*cbopt(x, A, mCB, sCB)
    popt, pcov = curve_fit(
        alpcopt,
        bin_centers,
        bin_values, 
        sigma=bin_errors,
        
        p0=[200],
    )
    a= popt
    plt.plot(bin_centers, alpcopt(bin_centers, a), 'r-', label='CB' + r"$\langle m_{H} \rangle $" + f'= ({mCB:.2f}' + r'$ \pm $' +f'{round(sCB, 2)})' +  ' GeV')
    
    
    # Step 7: Display the plot
    plt.xlabel(r'$m_{H}(GeV)$')
    plt.ylabel('N')
    #plt.title('Histogram of Higgs Masses')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/higgs_hist_last_500_94_cb.png")

def CrystalBallopt(x, A, mCB, sCB):
    #optimal:
    aL = 1.4709725540155203
    aR =1.6714153694042544
    nL = 3.7533888887587894
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

if __name__=='__main__':

    #1 naloga:
    #data = np.load('gauss.npy', allow_pickle=True)
    #print(data)

    """
    kmeans = KMeans(n_clusters=2, max_iter=100, n_init=10)  
    kmeans.fit(data)
    #print(kmeans.cluster_centers_)
    #plotgauss(data)
    res = KMeanss(data,n_clusters=5)
    
    #plot_kmeans_results(data, res[1], res[0])
    res = kmeans_multiple_runs(data, 5,100)
    """
    #2.naloga:
    #on first event:
    data = np.load("h_bb_sorted.npy", allow_pickle =True)[0]
    
    data = np.array(data)
    
    
    
    tdata = convert_to_npy(data)
    """
    plt.figure(figsize=(6, 5))
    
    #res.hierarhic_group(1,0.6)
    #res.calc_higgs('high')
    R_values = np.arange(0.2, 1.0, 0.2)
    
    for R in R_values:
        res = Higgs(tdata)
        print(R)
        # Apply the function for each R value (assuming p is some constant)
        results = abs(np.array(res.infrared_safety_high(238,R))-np.array(infrared_safety_pyjet(238,R)))  # Replace 'your_p_value' with the actual value of p
        # Plotting the results (this is a placeholder; adjust the plotting based on your result format)
        plt.plot(results, label=f'R = {round(R, 2)}')
    #plt.axhline(y=125.25, linestyle='--', label=r'125.25 GeV')  # Plot the constant line
    plt.xlabel('N_dropout')
    plt.ylabel(r'$|m_{H}-m_{H,pyjet}| \ [GeV]$')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/higgs_inf_kf_R_diff.png")
    """
    #res = calc_time_inf(237)
    #plot_time(res[1], res[0])
    
    #res = Higgs(tdata)
    #res.calc_higgs('pyjet')
    #print(res.hmass)
    #hminf = res.infrared_safety_pyjet(239,R=0.6)
    #print(hminf)
    #hminf = infrared_safety_pyjet(238,0.6)
    #print(hminf)
    #plot_higgs_mass_values(hminf)
    #centr = res.KMeanss(10)
    #res.calc_higgs()
    #res.make_hist(2000, 13)
    #plot_kmeans_results(tdata[:,1:3], res.labels, centr, res.hmass, res.maxptind)

    #higgs all data:
    #data = np.load('h_bb_sorted.npy', allow_pickle=True)
    plot_higgs_mass_histogram()
    
