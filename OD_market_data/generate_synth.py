# === Imports (from mean.py / utils.py) ===
import torch
import numpy as np
import numpy.linalg as linalg
from scipy.stats import ortho_group
import scipy.stats as st
import scipy as sp
import random
import utils

device = utils.device

# === Verbatim from mean.py ===
def generate_sample(n, feat_dim):
    #create sample with mean 0 and variance 1
    X = torch.randn(n, feat_dim, device=device)
    #X = X/(X**2).sum(-1, keepdim=True)
    return X

def corrupt(feat_dim, n_dir, cor_portion, opt):

    prev_dir_l = []
    #noise_norm = opt.norm_scale*np.sqrt(feat_dim)    
    #noise_m = torch.from_numpy(ortho_group.rvs(dim=feat_dim).astype(np.float32)).to(device)
    #chunk_sz = n_cor // n_dir
    #cor_idx = torch.LongTensor(list(range(n_cor))).to(utils.device).unsqueeze(-1)
    
    noise_idx = 0
    #generate n_dir number of norms, sample in interval [kp, sqrt(d)]
    #for testing, to achieve high acc for tau0 & tau1: noise_norms = np.random.normal( np.sqrt(feat_dim), 1. , (int(np.ceil(n_c
    
    #min number of samples per noise dir
    n_noise_min = 520   
    end = 0
    noise_vecs_l = []
    chunk_sz = (feat_dim-1) // n_dir
    for i in range(n_dir):
        
        cur_n = int(n_noise_min * 1.1**i)
        cur_noise_vecs = 0.1 *torch.randn(cur_n, feat_dim).to(utils.device)
                
        cur_noise_vecs[:, i*chunk_sz] += np.sqrt(n_dir/np.clip(cor_portion, 0.01, None))
        #noise_vecs[start:end, noise_idx] += 1./np.clip(cor_portion, 0.01, None)         
        
        cur_noise_vecs[cur_n//2:] *= (-1)
        ###corrupt1d(X, prev_dir_l, cor_idx[start:end], noise_vecs[start:end])        
        noise_vecs_l.append(cur_noise_vecs)
        
    #noise_vecs = 0.1 *torch.randn(n_cor, feat_dim, device=X.device)
    noise_vecs = torch.cat(noise_vecs_l, dim=0)
    cor_idx = torch.LongTensor(list(range(len(noise_vecs)))).to(utils.device)
    n = int(len(noise_vecs)/(cor_portion/(1-cor_portion)))
    X = generate_sample(n, feat_dim)
    X = torch.cat((noise_vecs, X), dim=0)
    
    if len(X) < feat_dim:
        print('Warning: number of samples smaller than feature dim!')
    opt.true_mean = torch.zeros(1, feat_dim, device=utils.device)
    '''
    idx = torch.zeros(n_dir, n_points, device=X.device)
    src = torch.ones(1, n_cor, device=X.device).expand(n_dir, -1)
    
    idx.scatter_add_(1, cor_idx, src)
    idx = idx.sum(0)
    cor_idx = torch.LongTensor(range(n_points))[idx.view(-1)>0].to(X.device)
    '''
    return X, cor_idx, noise_vecs

# === Minimal setup & one-shot generation (verbatim pattern from mean.py::generate_and_score) ===
opt = utils.parse_args()

# feature-dim and sample count defaults used in synthetic runs
opt.n = 10000
opt.feat_dim = 128
opt.feat_dim = 512

n = opt.n
feat_dim = opt.feat_dim
n_repeat = 20
opt.p = 0.2  # default total portion corrupted
# number of top dirs for calculating tau0
opt.n_top_dir = 1

n_dir_l = [20]  # a representative synthetic setting in the script
dataset_name = 'syn'

if dataset_name == 'syn':
    for n_dir in n_dir_l:
        cur_data_l = []
        for _ in range(n_repeat):
            X, cor_idx, noise_vecs = corrupt(feat_dim, n_dir, opt.p, opt)
            n = len(X)
            
            X = X - X.mean(0)
            if opt.fast_jl:
                X = utils.pad_to_2power(X)
            cur_data_l.append([X, cor_idx])

# At this point:
# - cur_data_l is a list of [X, cor_idx] pairs
# - X is your synthetic tabular dataset (torch.FloatTensor)
# - cor_idx indexes the outliers (first block inserted by corrupt())
