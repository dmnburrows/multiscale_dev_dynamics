#Import packages
#---------------------------------------
import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import glob
import admin_functions as adfn
import trace_analyse as tfn
from tqdm import tqdm

fdata = "/home/dburrows/DATA/BLNDEV-WILDTYPE/"


#structural connectivity
smat = pd.read_csv(f'{fdata}/structural_connectivity_kunst.csv', index_col=0)

import numpy as np
import pandas as pd
import re

dff_l  = np.sort(glob.glob(f"{fdata}/*regdeltaff*"))
bind_l  = np.sort(glob.glob(f"{fdata}/*regbin*"))
coord_l  = np.sort(glob.glob(f"{fdata}/*labcoord_new*"))
nnb_l  = np.sort(glob.glob(f"{fdata}/*nnb*"))

def collapse_lr_mean(smat: pd.DataFrame,
                     left_tag="_left",
                     right_tag="_right") -> pd.DataFrame:
    """
    Collapse a hemisphere-split region×region matrix into base-region×base-region
    by averaging over the 2x2 (L/R) blocks. Handles missing L or R for any region.
    """
    # Ensure rows/cols align and are same ordering
    smat = smat.copy()
    assert (smat.index.equals(smat.columns)), "Expected square matrix with matching row/col labels."

    # Map each label -> base region name
    def base_name(x: str) -> str:
        if x.endswith(left_tag):
            return x[:-len(left_tag)]
        if x.endswith(right_tag):
            return x[:-len(right_tag)]
        return x  # if already unsuffixed

    row_base = pd.Index([base_name(x) for x in smat.index], name="region")
    col_base = pd.Index([base_name(x) for x in smat.columns], name="region")

    # Build collapsed matrix by block-averaging with groupby
    tmp = smat.copy()
    tmp.index = row_base
    tmp.columns = col_base

    # groupby over duplicate row/col names and take mean over all entries in each block
    collapsed = tmp.groupby(level=0).mean().T.groupby(level=0).mean().T

    return collapsed

smat_lr = collapse_lr_mean(smat)
# --- after smat_lr = collapse_lr_mean(smat) ---

coarse = {
    # hypothalamus (3 -> 1)
    "Rostral_hypothalamus": "Hypothalamus",
    "Intermediate_hypothalamus": "Hypothalamus",
    "Caudal_hypothalamus": "Hypothalamus",

    # reticular formation (3 -> 1)
    "Anterior_reticular_formation": "Reticular_formation",
    "Intermediate_reticular_formation": "Reticular_formation",
    "Posterior_reticular_formation": "Reticular_formation",

    # medulla stripes (5 -> 1)
    "MO_stripe_1": "Medulla",
    "MO_stripe_2": "Medulla",
    "MO_stripe_3": "Medulla",
    "MO_stripe_4": "Medulla",
    "MO_stripe_5": "Medulla",
}

# relabel (anything not in dict keeps its own name), then block-mean rows+cols
smat_coarse = (
    smat_lr.rename(index=lambda x: coarse.get(x, x), columns=lambda x: coarse.get(x, x))
          .groupby(level=0).mean()
          .T.groupby(level=0).mean().T)

df = smat_coarse.copy()
df = df.loc[df.columns, df.columns].copy()
np.fill_diagonal(df.values, 0)
smat_coarse = df

import numpy as np

def prob_dist(D_um, S, lam_um=80.0, p0=0.05, beta=3.0,
                              s_scale="max", zero_diag=True):
    """
    p = p0 * exp(-D/lam) * (1 + beta * S_norm)
    - distance sets locality
    - S only scales probabilities (esp. long-range), doesn't create connections alone
    """
    D = np.asarray(D_um, float)
    S = np.asarray(S, float)
    S = np.clip(S, 0, None)

    if s_scale == "max":
        denom = np.nanmax(S) + 1e-12
    else:  # e.g. s_scale=99 for percentile
        denom = np.nanpercentile(S, float(s_scale)) + 1e-12
    S_norm = np.clip(S / denom, 0, 1)

    P = p0 * np.exp(-D / lam_um) * (1.0 + (beta * S_norm))
    P = np.clip(P, 0.0, 1.0)

    if zero_diag:
        np.fill_diagonal(P, 0.0)
    return P

def nnb(curr=None, smat_coarse=None, lam_um = 180, p0=0.6, beta=40):
    # build new probabilistic nnb
    
    # start with local probability -> some exponential decay.
    # sample each neuron -> based on connections, setup exponential; sum together
    xyz = curr[:,:3].astype(float) * [.8, .8, 15] #define distance of each pixel in x,y,z
    #weight each neuron pair for sConn
    atlas_ind = smat_coarse.columns #order of indeces
    labels = curr[:,-1]
    labels = np.asarray(labels, dtype=object)  # shape (N,)
    df= pd.DataFrame(data=np.append(np.arange(0,len(atlas_ind)),np.nan), index=np.append(atlas_ind, 'Nan'))
    curr_index = df.loc[labels] #indeces of each of my labels to map back to atlas
    
    M = smat_coarse.to_numpy()          # (R, R)
    R = M.shape[0]
    # pad with an extra row/col of zeros (or np.nan if you prefer)
    Mpad = np.zeros((R+1, R+1), dtype=M.dtype)
    Mpad[:R, :R] = M
    
    idx = curr_index[0].to_numpy()         # (N,) floats with nan
    valid = np.isfinite(idx)
    
    idx_i = np.where(valid, idx.astype(np.int64), R)   # missing -> R (the padded slot)
    smat_w = Mpad[idx_i[:, None], idx_i[None, :]] # (N, N)
    # add in local probability
    local_p = np.zeros((xyz.shape[0], xyz.shape[0]))
    coords = np.asarray(xyz[:,:3].astype(float), dtype=np.float64)   # (N,3)
    pre_coord = coords[:,None]
    post_coord = coords[None,:]
    D = np.linalg.norm(pre_coord - post_coord, axis=2)  # (N,N)

    p_mat = prob_dist(D, smat_w, lam_um=lam_um, p0=p0, beta=beta,
                              s_scale="max", zero_diag=True)

    return(p_mat)

import numpy as np
from tqdm import tqdm

#=======================================================================
def avalanche(nnb, bind, min_size=3, seed_min_neigh=2, backfill_stop=30, show_pbar=True):
#=======================================================================
    binarray = (bind > 0).astype(np.uint8)
    nnbarray = (nnb > 0)
    N, T = binarray.shape
    pkg = np.zeros((N, T), dtype=np.int32)
    marker = 0

    t_iter = range(T - 1)
    if show_pbar:
        t_iter = tqdm(t_iter, total=T-1, desc="Avalanches", leave=True)

    for t in t_iter:
        cid = np.where(binarray[:, t] > 0)[0]
        if cid.size == 0:
            continue

        for c in cid:
            if pkg[c, t] == 0:
                neigh = np.where(nnbarray[c, :])[0]
                if np.intersect1d(neigh, cid).size > seed_min_neigh:
                    marker += 1
                    pkg[c, t] = marker

            neighbour = np.where(nnbarray[c, :])[0]
            neighbouron = np.intersect1d(cid, neighbour)
            if neighbouron.size == 0:
                continue

            where0 = np.where(pkg[neighbouron, t] == 0)[0]

            if where0.size < neighbouron.size:
                oldav = np.unique(pkg[neighbouron, t])
                realav = oldav[oldav > 0]
                if realav.size == 0:
                    continue

                firstav = int(realav.min())
                uniteav = np.where(np.isin(pkg[:, t], realav))[0]
                pkg[uniteav, t] = firstav
                pkg[c, t] = firstav

                convertav = realav[realav != firstav]
                back = min(t, backfill_stop)
                for av_id in convertav:
                    av_id = int(av_id)
                    for dt in range(1, back + 1):
                        fill = np.where(pkg[:, t - dt] == av_id)[0]
                        if fill.size:
                            pkg[fill, t - dt] = firstav
            else:
                if pkg[c, t] > 0:
                    pkg[neighbouron[where0], t] = pkg[c, t]

        # propagate to t+1
        n_av = np.unique(pkg[:, t])
        n_av = n_av[n_av > 0]
        if n_av.size == 0:
            continue

        cid2 = np.where(binarray[:, t + 1] > 0)[0]
        if cid2.size == 0:
            continue
        active2 = np.zeros(N, dtype=bool)
        active2[cid2] = True

        for n in n_av:
            cgroup = np.where(pkg[:, t] == n)[0]
            if cgroup.size == 0:
                continue
            nbr = np.where(nnbarray[cgroup, :])[1]
            if nbr.size == 0:
                continue
            nbr = np.unique(nbr)
            cand = nbr[active2[nbr]]
            cand = cand[pkg[cand, t + 1] == 0]
            if cand.size:
                pkg[cand, t + 1] = int(n)

    ids, counts = np.unique(pkg, return_counts=True)
    mask = ids > 0
    ids = ids[mask].astype(np.int32)
    avsize = counts[mask].astype(np.int64)

    if ids.size == 0:
        return np.zeros((2, 0), dtype=np.int64), pkg

    framesvec = np.array([(pkg == i).any(axis=0).sum() for i in ids], dtype=np.int32)
    keep = avsize >= min_size
    av = np.vstack((avsize[keep], framesvec[keep]))
    return av, pkg



import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

reg_list = ['Habenula', 'Hypothalamus', 'Medulla', 'Optic tecum - SPV', 'Pallium', 'Subpallium', 'Thalamus']
rng = np.random.default_rng(42)

# ---- worker ----
import numpy as np
from joblib import Parallel, delayed

# =========================
# GRANULAR avalanches ONLY
# =========================
reg_list = ['Habenula', 'Hypothalamus', 'Medulla', 'Optic tecum - SPV',
            'Pallium', 'Subpallium', 'Thalamus']

def process_one_fish_gran(i, dff_l, bind_l, coord_l, reg_list,
                          smat_coarse, lam_um=30, p0=0.1, beta=0.5, seed=0):
    # Load
    trace = np.load(dff_l[i])                  # (N,T)  (your script loads dff here)
    dff   = np.load(dff_l[i])
    bind  = np.load(bind_l[i])
    coord = np.load(coord_l[i], allow_pickle=True)

    name = adfn.save_name(dff_l[i])

    rng = np.random.default_rng(seed)          # IMPORTANT: per-worker RNG

    fish_res = {}
    for reg in reg_list:
        # IMPORTANT: take [0] so locs is a 1D integer index
        locs = np.where(coord[:, 5] == reg)[0]
        if locs.size == 0:
            continue                           # avoids empty S/D -> nanmax crash inside prob_dist()

        sub_coord = coord[locs]                # KEEP ALL COLUMNS (nnb() uses last col labels)
        sub_bind  = bind[locs]

        # sanity: avalanche expects (N,T)
        if sub_bind.ndim != 2 or sub_bind.shape[0] != locs.size:
            continue

        # build probabilistic neighbourhood
        sub_nnb = nnb(curr=sub_coord, smat_coarse=smat_coarse,
                      lam_um=lam_um, p0=p0, beta=beta)

        if sub_nnb.size == 0:
            continue

        # sample adjacency once per fish/region
        cmat = (rng.random(sub_nnb.shape, dtype=np.float32) < sub_nnb)

        # run avalanches (disable progress bar inside multiprocessing)
        av, pkg = avalanche(cmat, sub_bind, show_pbar=False)

        fish_res[reg] = {"av": av, "pkg": pkg}

    return name, fish_res


# ---- run in parallel + assemble + save once ----
n_jobs = 10

out = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
    delayed(process_one_fish_gran)(
        i, dff_l, bind_l, coord_l, reg_list,
        smat_coarse, 30, 0.1, 0.5, 42 + i
    )
    for i in range(len(dff_l))
)

results = dict(out)
np.save(fdata + "/gran_avalanches.npy", results, allow_pickle=True)
print("Saved:", fdata + "/gran_avalanches.npy")
