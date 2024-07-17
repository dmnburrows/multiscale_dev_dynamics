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
        
        self.dist = dfn.state_dist(self.dff)
        print('state distance calculated')
        
        self.le = LE(self.trace)
        print('LE calculated')
                
        
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
    E = efn.find_E(time_series, tau, 'fnn')
    embed = efn.takens_embed(E,tau,time_series)
    le = efn.LE_embed(embed, tau)
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