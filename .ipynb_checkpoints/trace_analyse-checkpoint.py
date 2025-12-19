#import criticality as crfn
import sys
# sys.path.insert(1, '/Users/dominicburrows/Dropbox/PhD/Analysis/my_scripts/Github/multiscale_dev_dynamics/')
# #import metastability as mfn
# sys.path.insert(1, '/Users/dominicburrows/Dropbox/PhD/Analysis/my_scripts/Github/seizure_dynamics/')
# import dynamics as dfn
# sys.path.insert(1, '/Users/dominicburrows/Dropbox/PhD/Analysis/my_scripts/Github/empirical_dynamic_modelling/')
# import EDM as efn

#CHECK
#================================    
class trace_analyse: 
#================================    
    """
    Class to analyse trace datasets. 
    
    """
    
    #========================
    def __init__(self, name, trace, dff, bind, coord):
    #========================
        self.name = name # dataset name
        self.trace = trace # Raw traces
        self.dff = dff # Normalised fluorescence
        self.bind = bind # Binarised traces
        self.coord = coord # Cell coordinates
        print('Loaded ' + name)
        
        

    #====================================
    def criticality_stats(self, n_neigh, n_bins, mini, maxi):
    #====================================
        
        """
        This functions runs all criticality analysis on your data.
        
   
    Inputs:
        n_neigh (int): number of closest neigbours to find
        n_bins (int): number of bins to use for correlation function
        mini (int): first bin
        maxi (int): last bin
    
        """
        import numpy as np
        from sklearn.metrics.pairwise import euclidean_distances

        
        
        self.nnb = crfn.neighbour(self.coord, n_neigh) #Calculate nearest neighbours
        print('Nearest neighbours found')
        
        self.av, self.pkg = crfn.avalanche(self.nnb, self.bind) #Calculate avalanches
        print('Avalanches calculated')
        
        self.llr_s, self.llr_d = crfn.LLR(self.av, 2000) #Calculate loglikelihood ratio
        self.exp_s, self.exp_d = crfn.power_exponent(self.av, 2000) #Calculate power law exponents
        self.dcc = crfn.DCC(self.av) #Calculate exponent relation
        print('Avalanche statistics calculated')
        
        self.br = crfn.branch(self.pkg, self.av) #Calculate branching ratio
        print('Branching ratio calculated')
        
        
        dist = euclidean_distances(self.coord) #Calculate euclidean distance matrix between all cells
        corr = np.corrcoef(self.trace) #Calculate correlation matrix
        self.corrdis = crfn.corrdist(corr, dist, n_bins, mini, maxi)
        print('Correlation function calculated')
        
        return(self)
    
    
    #====================================
    def firing_stats(self, denominator, cutoff):
    #====================================
        
        """
        This functions calculates all firing statistics on data.
        
   
    Inputs:
        denominator (int): denominator to convert into rate
        cutoff (int): threshold for short vs long range correlations in microns

        """
        
        import numpy as np
    
        self.fr = firing_rate(self.bind, denominator) #Calculate firing rates
        print('Firing rate calculated')
    
        self.fa = firing_amp(self.dff, self.bind) #Calculate firing amplitude
        print('Firing amplitude calculated')
        
        self.fd = firing_dur(self.bind) #Calculate firing duration
        print('Firing duration calculated')
        
        self.s_corr, self.l_corr = short_long_corr(self.trace, self.coord, cutoff) #Calculate firing rates
        print('Correlation calculated')
    
        self.dim = linear_dimensionality(np.cov(self.trace)) #Calculate dimensionality
        print('Dimensionality calculated')
        
        return(self)
    
    #====================================
    def dyn_stats(self, bind_transformed):
    #====================================
        self.n_states, self.p_state, self.m_dwell, self.null_m_dwell, self.v_dwell = meta(bind_transformed)
        print('metastability calculated')
        
        self.dist = state_dist_normbycell(self.dff)
        print('state distance calculated')
        
        # self.le = LE(self.trace)
        # print('LE calculated')
                
        
#================================================
def select_region(trace, dff, bind, coord, region):
#================================================
    
    """
    This function slices data to include only those within a specific brain region.

    Inputs:
        trace (np array): cells x timepoints, raw fluorescence values
        dff (np array): cells x timepoints, normalised fluorescence
        bind (np array): cells x time, binarised state vector
        coord (np array): cells x XYZ coordinates and labels
        region (str): 'all', 'Diencephalon', 'Midbrain', 'Hindbrain' or 'Telencephalon'
    
    Returns:
        sub_trace (np array): cells x timepoints, raw or normalised fluorescence values for subregion
        sub_bind (np array): cells x time, binarised state vector for subregion
        sub_coord (np array): cells x XYZ coordinates for subregion
    
    
    """
    
    import numpy as np

    if coord.shape[0] != trace.shape[0]:
        print('Trace and coordinate data not same shape')
        return()


    if region == 'all':
        locs = np.where(coord[:,4] != 'nan')

    else: 
        locs = np.where(coord[:,4] == region)

    sub_coord = coord[locs][:,:3].astype(float)
    sub_trace, sub_dff, sub_bind = trace[locs], dff[locs], bind[locs]


    return(sub_trace, sub_dff, sub_bind, sub_coord)



#===============================
def firing_rate(bind, denominator):
#===============================
    """
    This function calculate the median firing rate over all neurons. 
    
    Inputs:
        bind (np array): cells x time, binarised state vector
        denominator (int): denominator to convert into rate
        
    Returns:
        fr (float): median firing rate over all neurons
    
    """
    import numpy as np
    
    fr = np.median(np.sum(bind, axis = 1)/denominator)
    
    return(fr)


#===============================
def firing_amp(dff, bind):
#===============================
    """
    This function calculate the median normalised firing amplitude over all neurons. 
    NB this functions treats each spike as independent. 
    
    Inputs:
        dff (np array): cells x timepoints, normalised fluorescence
        bind (np array): cells x time, binarised state vector
        
    Returns:
        fa (float): median firing amplitude over all neurons
    
    """
    import numpy as np
    
    fa = np.median(dff[bind == 1])
    
    return(fa)


#===============================
def firing_dur(bind):
#===============================
    """
    This function calculate the mean firing event duration across all neurons. 
    
    Inputs:
        bind (np array): cells x time, binarised state vector
        
    Returns:
        fd (float): mean firing event duration over all neurons
    
    """
    import numpy as np
    import more_itertools as mit
    
    n_trans = []
    for i in range(bind.shape[0]): #Loop through each neuron
        si = np.where(bind[i] == 1)[0] #Find spike index
        n_trans = np.append(n_trans,[len(list(group)) for group in mit.consecutive_groups(si)]) #Group continuous values together and find their length
    fd = np.mean(n_trans) 
    
    return(fd)


#===============================
def short_long_corr(trace, coord, cutoff):
#===============================
    """
    This function calculate the median pairwise correlation across all neurons above and below a given distance range. 
    This function ignores all self correlations and negative correlations. 
    
    Inputs:
        trace (np array): cells x timepoints, raw fluorescence values
        coord (np array): cells x XYZ coordinates and labels
        cutoff (int): threshold for short vs long range correlations in microns
        
    Returns:
        corr_s (float): median short range correlation over all neurons
        corr_l (float): median long range correlation over all neurons
    
    """
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    
    #Short + Long range pairwise Correlation
    dist = euclidean_distances(coord) 
    corr = np.corrcoef(trace)

    # Take upper triangular of matrix and flatten into vector
    corr = np.triu(corr, k=0) 
    dist = np.triu(dist, k=0)
    corr_v = corr.flatten()
    dist_v = dist.flatten()

    # Convert all negative correlations to 0
    corr_v = [0 if o < 0 else o for o in corr_v]
    corr_v = np.array(corr_v)
    dist_v[np.where(corr_v == 0)] = 0 #Convert all negative correlations to 0s in distance matrix

    # Order by distances
    unq = np.unique(dist_v)
    dist_vs = np.sort(dist_v)
    corr_vs = np.array([x for _,x in sorted(zip(dist_v,corr_v))])

    # Remove all 0 distance values = negative correlations and self-correlation
    dist_ = dist_vs[len(np.where(dist_vs == 0)[0]):]
    corr_ = corr_vs[len(np.where(dist_vs == 0)[0]):]

    corr_s = np.median(corr_[dist_ < cutoff])
    corr_l = np.median(corr_[dist_ > cutoff])
    return(corr_s, corr_l)

#===============================
def linear_dimensionality(data):
#===============================
    """
    This function calculate the dimensionality as a measure of the equal/unequal weighting across all eigenvalues.
    
    Inputs:
        data (np array): covariance matrix - make sure this is the correct way around! 
        
    
    Returns:
        dim (float): dimensionality
    
    """
    import numpy as np
    
    v = np.linalg.eigh(data)[0]
    dim = (np.sum(v)**2)/np.sum((v**2))
    
    return(dim)


def meta(data):
    import numpy as np

    #Empirical data
    all_clust, sub_clust = affprop(data) #cluster with affinity prop on empirical data
    emp_sim = Sim_loop(data, all_clust, sub_clust) #calculate similarity between clustered states

    #Generate null data
    rpks = np.zeros((data.shape))
    for t in range(data.shape[0]):
        temp_pks = data[t]
        np.random.shuffle(temp_pks) 
        rpks[t] = temp_pks

    null_all_clust, null_sub_clust = affprop(rpks) #cluster with affinity prop on null data
    null_sim = Sim_loop(rpks, null_all_clust, null_sub_clust) #calculate similarity between clustered states
    tot_states = len(emp_sim)
    n_states = sum(emp_sim > np.max(null_sim))#np.percentile(null_sim, 0.9))

    #check on nulled confirmed
    fin_clust = sub_clust[emp_sim > np.max(null_sim)] #np.percentile(null_sim, 0.9)] 
                   #Find the clusters that occur above chance
    p_state, m_dwell, v_dwell = state_stats(fin_clust, all_clust) #Calculate state transition statistics

    if len(fin_clust)>0:
        null_m_dwell = null_states(fin_clust, data) #Calculate the mean dwell time with random dynamics
    else: null_m_dwell=None

    return(n_states, p_state, m_dwell, null_m_dwell, v_dwell)


#find first PC of data for embedding
def LE(input_data, n_components=3):
    from sklearn import decomposition

    pca = decomposition.PCA(n_components)
    fit = pca.fit(input_data)
    time_series = fit.components_[0]

    tau = 1
    E = find_E(time_series, tau, 'fnn')
    embed = takens_embed(E,tau,time_series)
    le = LE_embed(embed, tau)
    return(le)


def meta(data):
    import numpy as np

    #Empirical data
    all_clust, sub_clust = affprop(data) #cluster with affinity prop on empirical data
    emp_sim = Sim_loop(data, all_clust, sub_clust) #calculate similarity between clustered states

    #Generate null data
    rpks = np.zeros((data.shape))
    for t in range(data.shape[0]):
        temp_pks = data[t]
        np.random.shuffle(temp_pks) 
        rpks[t] = temp_pks

    null_all_clust, null_sub_clust = affprop(rpks) #cluster with affinity prop on null data
    null_sim = Sim_loop(rpks, null_all_clust, null_sub_clust) #calculate similarity between clustered states
    tot_states = len(emp_sim)
    n_states = sum(emp_sim > np.percentile(null_sim, 0.9))

    #check on nulled confirmed
    fin_clust = sub_clust[emp_sim > np.percentile(null_sim, 0.9)] #Find the clusters that occur above chance
    p_state, m_dwell, v_dwell = state_stats(fin_clust, all_clust) #Calculate state transition statistics

    if len(fin_clust)>0:
        null_m_dwell = null_states(fin_clust, data) #Calculate the mean dwell time with random dynamics
    else: null_m_dwell=None

    return(n_states, p_state, m_dwell, null_m_dwell, v_dwell)

#Cluster with affinity propagation
#==============================
def affprop(data):
#==============================
    """
    This function performs affinity propagation on state vectors. 
    
    Inputs:
        data (np array): cellsxtimepoints, state vectors
        
    Returns:
        all_c (np array): 1d vector of cluster labels for each time point
        sub_c (np array): 1d vector of all unique cluster labels, that label more than a single time point

    """
    from sklearn.cluster import AffinityPropagation
    import numpy as np
    
    
    cluster = AffinityPropagation(damping = 0.5, max_iter = 200, convergence_iter = 15).fit(data)
    unq,counts = np.unique(cluster.labels_, return_counts = True)
    all_c = cluster.labels_
    sub_c = unq[counts > 1] #Remove clusters that have only a singular member
    return(all_c, sub_c)

#Similarity
#==============================
def Similarity(curr_clust):
#==============================
    """
    This function calculates the mean similarity between state vecotrs belonging to a cluster.
    
    Inputs:
        curr_clust (np array): all state vectors belonging to this cluster
        
    Returns:
        mean_sim (float): the mean similarity

    """
    import numpy as np
    
    ijdot = np.inner(curr_clust, curr_clust)
    self_dot = np.apply_along_axis(np.max,0,ijdot)
    idot = np.reshape(np.repeat(self_dot, ijdot.shape[0]), ijdot.shape)
    jdot = np.reshape(np.repeat(self_dot, ijdot.shape[0]), ijdot.shape).T
    sim_mat = np.triu(ijdot / (idot + jdot - ijdot))
    np.fill_diagonal(sim_mat,0)
    mean_sim = np.mean(sim_mat[np.nonzero(sim_mat)])
    return(mean_sim)

#=========================================
def Sim_loop(data, all_clust, sub_clust):
#==========================================
    """
    This function loops through all clusters in a dataset and finds the mean similarity for each cluster. 
    
    Inputs:
        data (np array): cells x timepoints
        all_clust (np array): 1d vector of cluster labels for each time point
        sub_clust (np array): 1d vector of all unique cluster labels, that label more than a single time point

        
    Returns:
        sim_list (list): list of all similarities for each cluster

    """
    import numpy as np
    
    sim_list = list(range(len(sub_clust)))
    
    #Loop through all clusters with more than 1 member
    for i in range(len(sub_clust)):
        curr_clust = data[np.where(all_clust == sub_clust[i])[0]] #Find all time frames belonging to current cluster
        sim_list[i] = Similarity(curr_clust) #Calculate mean similarity for this cluster
    return(sim_list)


#==========================================
def state_stats(fin_clust, all_clust):
#==========================================
    """
    This function calculates the probability and mean dwell times of each state. 
    
    Inputs:
        fin_clust (np array): 1d vector of state cluster labels occur with above chance probability
        all_clust (np array): 1d vector of cluster labels for each time point
        
    Returns:
        p_state (np.array):  1d vec containing probabilities of each state
        m_dwell (np.array): 1d vec containing the mean dwell time for each state
        full_vec (list): contains all durations in between every single state transition 
        
    """

    import more_itertools as mit
    import numpy as np

    p_state, m_dwell = np.zeros(len(fin_clust)),np.zeros(len(fin_clust)) 
    
    full_vec = list(range(len(fin_clust)))
    for i in range(len(fin_clust)):
        
        #calculate probabilities of each state
        p_state[i] = len(np.where(all_clust == fin_clust[i])[0])/len(all_clust)
        
        #find all periods with the same state over consecutive time frames
        dur_list = [list(group) for group in mit.consecutive_groups(np.where(all_clust == fin_clust[i])[0])]
        vec = []
        
        for t in range(len(dur_list)):
            vec = np.append(vec, len(dur_list[t]))
        m_dwell[i] = np.mean(vec)
        full_vec[i] = vec
    return(p_state, m_dwell, full_vec)


#==========================================
def null_states(fin_clust, data):
#==========================================
    """
    This function calculates the mean dwell time in a system with a given number of states and random dynamics. 
    
    Inputs:
        fin_clust (np array): 1d vector of state cluster labels occur with above chance probability
        data (np array): cells x timepoints
        
    Returns:
        null_m_dwell (np.array): 1d vec containing the mean dwell time for each state

    """

    import random
    import more_itertools as mit
    import numpy as np

    all_states = np.arange(1,len(fin_clust)+1)
    rand_states = np.array(random.choices(all_states, k = data.shape[0]))
    dur_list = [list(group) for group in mit.consecutive_groups(rand_states)]
    vec = []
    for t in range(len(dur_list)):
        vec = np.append(vec, len(dur_list[t]))
    null_m_dwell = np.mean(vec)
    return(null_m_dwell)


#=================
def cosine_sim(data):
#================
    """
    This functions calculates the cosine similarity from one point in time to to the next in state space. 
    
    Inputs:
        data (np array): cells x timeframes

    Returns:
        dist (np array): 1d vector, distance distribution
    """
    import numpy as np
    dist = np.zeros((data.shape[1])-1)
    for i in range(dist.shape[0]):
        data_t0 = data[:,i]
        data_t1 = data[:,i+1]
        dp = sum(data_t0 * data_t1) #dot product
        # Magnitude of each vector (norm)
        norm_t0 = np.linalg.norm(data_t0)
        norm_t1 = np.linalg.norm(data_t1)
        
        # Handling division by zero if either vector is zero (adding a small constant epsilon)
        epsilon = 1e-10  # A small number to avoid division by zero
        mag_prod = max(norm_t0 * norm_t1, epsilon)
        
        dist[i] = dp / mag_prod
    return(dist)

#======================================================================================
def state_dist_normbycell(data):
#======================================================================================
    """
    This functions calculates the euclidean distance from one point in time to to the next in state space. 
    
    Inputs:
        data (np array): cells x timeframes

    Returns:
        dist (np array): 1d vector, distance distribution
    """
    import numpy as np
    dist = np.zeros((data.shape[1])-1)
    for i in range(dist.shape[0]):
        data_t0 = data[:,i]
        data_t1 = data[:,i+1]
        
        dist[i] = np.linalg.norm(data_t0 - data_t1) / data.shape[0]#euclidean distance distribution
    return(dist)


#======================================================================================
def magnitude_distance(data):
#======================================================================================
    """
    This functions calculates the euclidean distance from one point in time to to the next in state space. 
    
    Inputs:
        data (np array): cells x timeframes

    Returns:
        dist (np array): 1d vector, distance distribution
    """
    import numpy as np
    dist = np.zeros((data.shape[1])-1)
    for i in range(dist.shape[0]):
        data_t0 = np.linalg.norm(data[:,i])
        data_t1 = np.linalg.norm(data[:,i+1])
        
        dist[i] = ((data_t0 - data_t1)**2 ) / data.shape[0] #euclidean distance distribution
    return(dist)


import admin_functions as adfn


#==============================================
#EMBED DATA
#==============================================

#==============================================
def Lorenz(x, y, z, sigma, r, b):
#==============================================

    """
    This function plots the lorenz attractor system, a simplified model of convection rolling - long rolls of counter-rotating air that are oriented approximately parallel to the ground. 
    
    Inputs:
       x (int/float): rate of convective overturning
       y (int/float) horizontal temperature difference
       z (int/float): departure from linear vertical temperature gradient
       s (int/float): prandtl parameter
       r (int/float): rayleigh parameter
       b (int/float): b parameter
    Returns:
       x_d, y_d, z_d (float): values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_d = sigma*(y - x)
    y_d = r*x - y - x*z
    z_d = x*y - b*z
    return x_d, y_d, z_d


#==============================================
def takens_embed(m, tau, data):
#==============================================
    """
    This function takes a singular time series as input and performs lagged coordinate embedding to generate a reconstructed attractor with dimension m and time lag tau. 
    
    Inputs:
        m (int): embedding dimension
        tau (int): time lag into past
        data (np array): 1d vector time series for reconstruction  
    
    Returns:
        data_embed (np array): m x t array of time x embedding dimension
    
    """
    import numpy as np
    data_embed = np.zeros((data.shape[0] - ((m-1)*tau), m))
    
    #loop through each dimension
    for i in range(0, m):
        
        if i == m-1:
            data_embed[:,(m-1)-i] = data[(i*tau):]
        
        else:
            data_embed[:,(m-1)-i] = data[(i*tau):-1* ((m*tau)-((i+1)*tau))]

    return(np.array(data_embed))


#==============================================    
def LE_embed(data, tau):
#==============================================    
    """
    This calculates the lyapunov exponent on an embedded dataset. 
    
    Inputs:
        data (np array): embedded timmeseries
        tau (int): time lag

    
    Returns:
        LE (float): lyapunov exponent
    
    """

    import numpy as np
    from sklearn.neighbors import NearestNeighbors 

    le = np.zeros((data.shape[0]-1))
    
    #Find nearest neighbours
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    
    #Loop through each point
    for i in range(1,data.shape[0]-1):
        sum_ = 0
        sum_count = 0
        
        #Loop through each neighbour to i
        for e in range(indices.shape[0]):
            dj0 = np.linalg.norm(data[indices[e][0]] - data[indices[e][1]]) #Distance at time 0
            
            if dj0 > 0: #if distance is 0 at start ignore
                sep = indices[e][1] - indices[e][0] #Time separation at t0

                #Avoid time points that go past end 
                if e+i < indices.shape[0]:
                    d1i_ind = indices[e+i][0]
                    d2i_ind = d1i_ind+sep
                    if d2i_ind< data.shape[0]:
                        dji = np.linalg.norm(data[d1i_ind] - data[d2i_ind]) #Distance at time i

                        if np.abs(dji/dj0) == 0: #if distance at end is 0 add 0
                            sum_ += 0
                        else:
                            sum_ += np.log(np.abs(dji/dj0))
                        sum_count +=1
             
        if sum_count == 0:
            break
        le[i-1] = (1/ i) *(sum_/sum_count)
                    
    return(le)






#==============================================
#PARAMETER ESTIMATION
#==============================================

#==============================================
def MI(data, delay, n_bins):
#==============================================    
    """
    This function calculates the mutual information of a time series and a delayed version of itself. MI quantifies the amount of information obtained about 1 variable, by observing the other random variable. In terms of entropy, it is the amount of uncertainty remaining about X after Y is known. So we are calculating the amount of uncertainty about time series xi and xi + tau shifted, across a range of taus. To calculate MI for 2 time series, we bin the time series data into n bins and then treat each time point as an observation, and calculate MI using joint probabilities of original time series xi and delayed xi + tau. 
    
    Inputs:
        data (np array): 1d vector timeseries
        delay (int): time lag
        n_bins (int): number of bins to split data into
    
    Returns:
        MI (float): mutual information
    
    """
    
    import math
    import numpy as np
    
    
    MI = 0
    xmax = np.max(data) #Find the max of the time series
    xmin = np.min(data) #Find the min of the time series
    delay_data = data[delay:len(data)] # generate the delayed version of the data - i.e. starting from initial delay
    short_data = data[0:len(data)-delay] #shorted original data so it is same length as delayed data
    size_bin = abs(xmax - xmin) / n_bins #size of each bin
    
    #Define dicts for each probability
    P_bin = {} # probability that data lies in a given bin
    data_bin = {} # data lying in a given bin
    delay_data_bin = {} # delayed data lying in a given bin
    
    
    prob_in_bin = {} #
    condition_bin = {} 
    condition_delay_bin = {}
    
    #Simple function for finding range between values of time series
    def find_range(time_series, xmin, curr_bin, size_bin):
        values_in_range = (time_series >= (xmin + curr_bin*size_bin)) & (time_series < (xmin + (curr_bin+1)*size_bin))
        return(values_in_range)

    #Loop through each bin
    for h in range(0,n_bins):
        
        #calculate probability of a given time bin, unless already defined
        if h not in P_bin:
            data_bin.update({h:  find_range(short_data, xmin, h, size_bin)})
            P_bin.update({h: len(short_data[data_bin[h]]) / len(short_data)})            
            
        #populate probabilities for other time bins 
        for k in range(0,n_bins):
            if k not in P_bin:
                data_bin.update({k: find_range(short_data, xmin, k, size_bin)})
                P_bin.update({k: len(short_data[data_bin[k]]) / len(short_data)})                            
                
            #to calculate the joint probability we need to find the time points where the lagged data lie in a given bin
            if k not in delay_data_bin:
                delay_data_bin.update({k: find_range(delay_data, xmin, k, size_bin) })
                                
            # Find the joint probability, that OG time series lies in bin h and delayed time series lies in bin k
            Phk = len(short_data[data_bin[h] & delay_data_bin[k]]) / len(short_data)

            if Phk != 0 and P_bin[h] != 0 and P_bin[k] != 0:
                MI += Phk * math.log( Phk / (P_bin[h] * P_bin[k]))
    return(MI)

    
#==============================================    
def FNN(data,tau,m, thresh):
#==============================================    
    """
    This function calculates how many nearest neighbours are false neighbours, in an embedded timeseries. Specifically, false nearest neighbours are defined as nearest neighbours to each point in E dimensional embedded space whose distances in E+1 dimensional space are greater than a defined threshold. 
    
    Inputs:
        data (np array): 1d vector timeseries
        tau (int): time lag
        m (int): embedding dimension
        thresh (int): false nn threshold
    
    Returns:
        n_false_NN (int): number of false nns
    
    """
    from sklearn.neighbors import NearestNeighbors 
    import numpy as np

    embed_data_1 = takens_embed(m, tau, data)
    embed_data_2 = takens_embed(m+1, tau, data)
    embed_data_1 = embed_data_1[:embed_data_2.shape[0]] #Shorten embedded time series to match eachother
    
    #Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(embed_data_1)
    distances, indices = nbrs.kneighbors(embed_data_1)

    #two data points are nearest neighbours if their distance is smaller than the standard deviation
    sigma = np.std(distances.flatten())

    n_false_NN = 0
    
    #if distance between points is nonzero, less than the std AND distance between nn in next dimension / previous dimension > threshold = nn is false
    for i in range(embed_data_2.shape[0]):
        if (0 < distances[i,1]) and (distances[i,1] < sigma) and (np.linalg.norm(embed_data_2[i] - embed_data_2[indices[i][1]])/distances[i][1]) > thresh:
            n_false_NN  += 1;
    return n_false_NN 



#====================================
def simplex_project(data, E, tau, t):
#====================================

    """
    This function performs simplex projection over t time steps into the future. Briefly, it splits a time series in library and prediction sets.
    It then embeds the library manifold in E dimensions, using E time lags. For each point p (embedded in E dim space) in the 
    prediction timeseries, the algorithm finds the E+1 nearest neighbours in the library manifold which forms a simplex around 
    the point p. The algorithm then predicts the position of point p at t using the positions of each neighbour at t exponentially weighted by 
    the distances from p at t0. See: Sugihara et al. 'Nonlinear Forecasting as a way of distinguishing chaos from measurement error in time series'.
    
    
    Inputs:
        data (np array): 1d vector of time series to perform simplex projection. 
        E (int): embedding dimension
        tau (int): time delay to use for embedding
        t (int): how many time steps ahead to predict
        
    
    Returns:
        corr (float): correlation coefficient between observed and predicted
        x_tp_m (np array): a vector of observations
        x_tp_pred_m (np array): a vector of predictions
        
    """
    from scipy import spatial
    import numpy as np
    import pandas as pd
    
    
    #==========================================
    def shift_nn(shape, lib_m, dist_mat, nn_ind, nn_ind_tp, curr_dist, nn_num, num, t): 
    #========================================== 
    #This function deals with points that go off manifold at t - finds next nearest neighbours
    
        nn_off = np.sum(nn_ind_tp >= shape) #Number of nearest neighbours that go off manifold at time t
        nn_rem = sorted(range(len(curr_dist)), key=lambda k: curr_dist[k])[nn_num:] #return indeces of ordered remaining nearest neighbours not currently included

        count = 0
        new_nn_tp_l, new_nn_l = [],[]
        for nn in nn_rem: #loop through each remaining neighbour
            if nn + t < shape: #if index of nn + t is on manifold, add to list and count
                new_nn_l = np.append(new_nn_l, nn) #add indeces of new neighbours at t0
                new_nn_tp_l = np.append(new_nn_tp_l, nn+t) #add indeces of new neighbours at tp
                count +=1 
                if count == nn_off: #Stop loop once you have enough neighbours
                    break

        nn_on = nn_ind[nn_ind_tp < shape] #Indeces of OG nearest neighbours at t0 that stay on the manifold into t
        nn_tp_on = nn_ind_tp[nn_ind_tp < shape] #Indeces of OG nearest neighbours at t that stay on the manifold at t

        new_nn_ind = (np.append(nn_on, new_nn_l)).astype(int) #add nearest neighbour points at t0 that stay on manifold up to tp, to new points
        new_nn_ind_tp = (np.append(nn_tp_on, new_nn_tp_l)).astype(int) #add nearest neighbour points at tp that stay on manifold, to new points


        nn = lib_m[new_nn_ind] #positions of nearest neighbours in library, to current point in pred at t0
        nn_dist = dist_mat[num, new_nn_ind]  #distances of each nn to our pred point
        w_mat = np.exp(-1*(nn_dist/np.min(nn_dist))) #matrix of weights for each nn

        return(new_nn_ind_tp, w_mat)
    

    # split data in half into library and prediction
    lib = data[:data.shape[0]//2]
    pred = data[data.shape[0]//2:]

    # Build manifold with given E and tau
    lib_m = takens_embed(E, tau, lib)
    pred_m = takens_embed(E, tau, pred)

    x_tp_m = np.zeros(pred_m.shape[0]) #Matrix to enter values you are trying to predict
    x_tp_pred_m = np.zeros(pred_m.shape[0]) #Matrix to values you have predicted
    x_tp_m[:] = np.nan #Make all nan to deal with empty values
    x_tp_pred_m[:] = np.nan


    #find the E+1 nearest neighbours in library
    dist_mat = spatial.distance_matrix(pred_m, lib_m) #compute distances between all points
    nn_num = E+1 #how many nearest neighbours to find


    #Loop through each point in pred
    for num in range(pred_m.shape[0]-t):

        # Find nearest neighbours in library for each pred_m point
        current_point = pred_m[num]
        curr_dist = dist_mat[num]
        nn_ind = sorted(range(len(curr_dist)), key=lambda k: curr_dist[k])[:nn_num] #return indeces of nearest neighbours in library

        #Calculate weights for simplex projection - weights are calculated from nn distance at t0
        nn = lib_m[nn_ind] #positions of nearest neighbours in library, to current point in pred at t0
        nn_dist = dist_mat[num, nn_ind]  #distances of each nn to our pred point
        w_mat = np.exp(-1*(nn_dist/np.min(nn_dist))) #matrix of weights for each nn 

        # Where do nn end up at t + n
        nn_ind_tp = np.array(nn_ind) + t #find indeces of neighbours in the future for simplex projection

        #Deal with points that go off manifold at t - find next nearest neighbours 
        if sum(nn_ind_tp >= lib_m.shape[0]) >0:                
            #Replace neighbours that go off manifold at t with neighbours that dont, and recalculate weights
            nn_ind_tp, w_mat = shift_nn(lib_m.shape[0], lib_m, dist_mat, np.array(nn_ind), nn_ind_tp, curr_dist, nn_num, num, t)

        nn_tp = lib_m[nn_ind_tp] # locations of neighbours in future

        #Simplex project - how much do the positions of neighbours relative to point of interest change over time 
        #use weights from t 0
        #use neighbour points from t + n
        x_tp = pred_m[num][0] #Point I am trying to predict 
        x_tp_pred = 0
        for nn_i in range(w_mat.shape[0]): #Loop through each nn and sum over the weight*position at tp
            x_tp_pred+= (w_mat[nn_i]/np.sum(w_mat))*nn_tp[nn_i]
        x_tp_pred = x_tp_pred[0] #project back into 1d space

        x_tp_m[num] = x_tp #true 
        x_tp_pred_m[num+t] = x_tp_pred  #estimated - NB you are estimating the future value at t, not the original
        

    my = {'Obs': x_tp_m, 'Pred': x_tp_pred_m}
    my_df = pd.DataFrame(data=my) 
    corr = my_df.corr()['Obs']['Pred']
        
        
    return(corr, [x_tp_m, x_tp_pred_m])

#==============================================    
def find_tau(data, mode):
#==============================================    
    """
    This function estimates tau for lagged coordinate embedding, using different approaches. mi = find the tau that provides the first minima of the MI - this provides most independent information to initial time series without completely losing the time series. ac = find the tau at which the autocorrelation drops below 1/e. 
    
    Inputs:
        data (np array): 1d vector timeseries
        mode (str): 'mi' or 'ac'
    
    Returns:
        tau (int): estimated tau for embedding
    
    """
    
    if mode == 'mi':
    
        import numpy as np
        from scipy.signal import argrelextrema

        MI_list = []
        for i in range(1,50):
            MI_list = np.append(MI_list,[MI(data,i,50)])

        tau = argrelextrema(MI_list, np.less)[0][0] + 1 #find the first minima of MI function
        return(tau)
    
    if mode == 'ac':
        import numpy as np
        
        auto = adfn.autocorr(data, 50)
        tau = np.min(np.where(auto < 1/np.e)) #find the tau at which the autocorrelation drops below 1/e
        return(tau)


#==============================================    
def find_E(data, tau, mode):
#==============================================    
    """
    This function estimates the embedding dimension E for lagged coordinate embedding, using different approaches. 
    fnn = find the E that approaches 0 false nearest neighbours - what embedding unfolds the manifold so that nearest neighbours become preserved.
    simplex = runs simplex projection over a range of E values with a given tau, and returns the E with greatest correlation between the real variable and predicted. 
    
    Inputs:
        data (np array): 1d vector timeseries
        tau (int): delay for embedding
        mode (str): 'fnn' or 'simplex'
    
    Returns:
        E (int): estimated number of dimensions to use for embedding
    
    """
    
    if mode == 'fnn':
        import numpy as np
        from scipy.signal import argrelextrema

        nFNN = []
        for i in range(1,15):
            nFNN.append(FNN(data,tau,i, 10) / len(data))

        E = np.min(np.where(np.array(nFNN) < 0.003 )) + 1
        return(E)
    
    if mode == 'simplex':
        import numpy as np
        
        E_range = 20 
        t = 1 
        
        corr_l = [0]*E_range
        for E in range(1, E_range+1):
            corr_l[E-1] = simplex_project(data, E, tau, t)[0]
            #print('Done E = ' + str(E))

        E_max = np.where(corr_l == np.max(corr_l))[0][0] + 1
        return(E_max)


    
    
#==============================================
#Cross mapping
#==============================================

#====================================
def crossmap(lib_m, pred_m):
#====================================
   
    """
    This function performs cross map predictions from one manifold to another. Briefly, the algorithm takes two different manifold and uses on to predict
    the other - if manifold Y can accurately predict manifold X, then Y contains information about X within it and thus X must cause Y. For each point in 
    manifold X, we find the nearest neighbours to that point and then locate the same nearest neighbours (labelled by their time indeces) on manifold Y. We 
    then use the locations of nn in Y and the distances between point of interest p on X and its nearest neighbours in X, to predict where point p ends up in Y. 
    The prediction will be accurate if the local structure of the manifold is converved across manifold X and Y - i.e. nearest neighbours to p on X are also nearest
    neighbours to p on Y. 
    
    
    Inputs:
        lib_m (np array): t x E embedded time series, used to make the prediction.
        pred_m (np array): t x E embedded time series, used as the observed dataset to compare with prediction. 
        
    
    Returns:
        x_m (np array): t x E embedded time series, used as the observed dataset to compare with prediction.
        x_pred_m (np array): t x E embedded time series, the predicted manifold. 
        
    """
    import numpy as np
    from scipy import spatial
    
    #make sure each manifold is the same length
    mini = np.min([pred_m.shape[0], lib_m.shape[0]])
    pred_m = pred_m[:mini,:]
    lib_m = lib_m[:mini,:]
    
    x_m = np.zeros(pred_m.shape[0]) #Matrix to enter values you are trying to predict
    x_pred_m = np.zeros(pred_m.shape[0]) #Matrix to values you have predicted
    x_m[:], x_pred_m[:] = np.nan, np.nan #Make all nan to deal with empty values

    #find the E+1 nearest neighbours in library
    dist_mat = spatial.distance_matrix(lib_m, lib_m) #compute distances between all points against themselves
    nn_num = lib_m.shape[1]+1 #how many nearest neighbours to find
    
    #Loop through each time step in lib
    for t in range(lib_m.shape[0]):
        # Find nearest neighbours in library for current point in library
        current_point = lib_m[t]
        curr_dist = dist_mat[t]
        nn_ind = sorted(range(len(curr_dist)), key=lambda k: curr_dist[k])[:nn_num+1][1:] #return indeces 

        nn = lib_m[nn_ind] #positions of nearest neighbours in library, to current point in lib
        nn_pred = pred_m[nn_ind] #positions of points in pred, labelled by indeces of nearest neighbours in lib to point in lib

        #Reconstruct pred point
        #Use weights calculated from distances between lib point and its nearest neighbours in lib
        #Use coordinates of pred points sharing time indeces with lib nearest neighbours

        #CALCULATE WEIGHTS
        nn_dist = dist_mat[t, nn_ind]  #distances of each nn to our pred point
        w_mat = np.exp(-1*(nn_dist/np.min(nn_dist))) #matrix of weights for each nn 

        #SUM OVER ALL PRED POINTS
        x_ = pred_m[t][0] # Value I am trying to predict
        x_pred = 0 # Predicted value
        for nn_i in range(w_mat.shape[0]): #Loop through each nn in lib and sum over the weight*position in pred
            x_pred+= (w_mat[nn_i]/np.sum(w_mat))*nn_pred[nn_i]
        x_pred = x_pred[0] #project back into 1d space

        #Populate vectors
        x_m[t] = x_
        x_pred_m[t] = x_pred
        
    return(x_m, x_pred_m)


#==============================================
def CCM_range(l_range, cause, effect):
#==============================================
    
    """
    This function performs convergent cross mapping between two manifolds: a causative variable (prediction manifold) - one we are testing 
    to see if it causes the other; an effected variable (library manifold) - one we are testing to see if it is caused by the other. CCM 
    is performed over a range of library sizes to check for convergence - the property that if the supposed causative variable actually causes
    the supposed effected variable the correlation between CCM predictions and observed manifold values should increase as more points are 
    added. 
    
    Inputs:
        l_range (np array): 1d vector of library sizes to test CCM
        cause (dict): dictionary for the causative variable, containing the data and parameters
        effect (dict): dictionary for the effected variable, containing the data and parameters
    
    Returns:
        corr_l (list): list containing CCM correlation values as you increase library 
        true_l (list): list containing observed prediction manifold as you increase library 
        pred_l (list): list containing predicted prediction manifold as you increase library 
    
    """    

    import random
    from scipy import stats
    import numpy as np
    
    lib, lib_E, lib_tau = effect['data'], effect['E'], effect['tau'] # This is the variable that will be used to predict -  the effected variable.
    pred, pred_E, pred_tau = cause['data'], cause['E'], cause['tau'] # This is the variable that will be predicted - the causative variable. 

    #Embed data
    lib_m = takens_embed(lib_E, lib_tau, lib) 
    pred_m = takens_embed(pred_E, pred_tau, pred) 

    #Initialise data output structures
    true_l, pred_l = [0]*len(l_range), [0]*len(l_range)
    corr_l = [0]*len(l_range)

    smallest = np.min([lib_m.shape[0], pred_m.shape[0]]) #find smallest array - may be different sizes if tau and E are different
    
    #Cross map as you increase library size
    for e,l in enumerate(l_range):
        t_l = random.sample(range(smallest),l) #Randomly sample
        lib_m_sub, pred_m_sub = lib_m[t_l], pred_m[t_l]
        true, pred = crossmap(lib_m_sub, pred_m_sub) #Run cross map on subsampled data
        true_l[e], pred_l[e] = true,pred
        corr_l[e] = stats.pearsonr(true, pred)[0]
    return(corr_l, true_l, pred_l)