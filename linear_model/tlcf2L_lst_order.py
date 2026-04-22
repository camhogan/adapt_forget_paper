#
# Linear student-teacher model with latent structure
#
# Learning of arbitrary number of tasks 
#
# Numerical simulations on order dependence
#
import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from itertools import permutations 

from tlcf2L_lst_model import generate_tasks, calc_dW, fnorm2

#import matplotlib.pyplot as plt


def calc_perm(idxs):
	P = len(idxs)
	Pmat = np.zeros((P, P))
	for j, jidx in enumerate(idxs):
		Pmat[j, jidx] = 1
		#Pmat[jidx, j] = 1
	return jnp.array(Pmat)


def calc_perm_typicality(CA):
	P = len(CA); Pones = jnp.ones((P))
	ts = jnp.dot( CA, Pones ) - Pones

	tidxs = jnp.argsort(ts)
	Perm_ptoc = calc_perm( tidxs ) #small to large
	Perm_ctop = calc_perm( jnp.flip(tidxs) ) #large to small
	
	return Perm_ptoc, Perm_ctop


def generate_Mperms(P):
	Mperms = []
	perms = permutations( range(P) )
	for perm in list(perms):
		Mperms.append( np.zeros((P, P)) )
		#print(perm)
		for pidx in range(P):
			Mperms[-1][perm[pidx], pidx] = 1
	return Mperms


#calculate the path length
def calc_plength(Ctmp):
	plength = 0.0
	for i in range( len(Ctmp)-1 ):
		plength += Ctmp[i,i+1]
	return plength


def calc_Pflip(P):
	Pflip = np.zeros( (P,P) )
	for q in range(P):
		Pflip[q,P-q-1] = 1
	return jnp.array(Pflip)


def calc_perm_path_length(CA, Mperms):
	P = len(CA)
	min_Mperm = np.zeros((P,P)); max_Mperm = np.zeros((P,P))
	min_length = 0; max_length = 0
	
	for midx, Mperm in enumerate(Mperms):
		CAp = jnp.dot( Mperm, jnp.dot(CA, Mperm.T) )
		plength = calc_plength(CAp)
		if midx == 0:
			min_length = plength; min_Mperm = Mperm 
			max_length = plength; max_Mperm = Mperm
		else:
			if plength < min_length:
				min_length = plength; min_Mperm = Mperm 
			if plength > max_length:
				max_length = plength; max_Mperm = Mperm
	Pflip = calc_Pflip(P)
	Perm_minpaths = [min_Mperm, jnp.dot(Pflip, min_Mperm)]
	Perm_maxpaths = [max_Mperm, jnp.dot(Pflip, max_Mperm)]
	
	return Perm_minpaths, Perm_maxpaths
	

def run_exp(params, key):
	num_epochs = params['num_epochs']
	learning_rate = params['learning_rate']
	Nsess = params['Nsess']
	Nx = params['Nx']; Ny = params['Ny']
	
	key, abkey = random.split(key, num=2)
	Aseq, Bseq = generate_tasks(abkey, params)	
	
	W = jnp.zeros((Ny, Nx)) #zero initialization
	
	errors = np.zeros((Nsess, Nsess*num_epochs))
	for t in range( Nsess*num_epochs ):
		task_idx = t//num_epochs
		W = W - learning_rate*calc_dW(W, Aseq[task_idx], Bseq[task_idx])

		for nidx in range(Nsess):
			errors[nidx, t] = fnorm2( Bseq[nidx] - jnp.dot(W, Aseq[nidx]) )/Ny

	return errors


def generate_feature_correlations(pkey, Nsess):
	# 0 <= rho <= 1
	#rhos = random.uniform(pkey, (Nsess, Nsess))
	# -1 <= rho <= 1
	rhos = -jnp.ones((Nsess, Nsess)) + 2*random.uniform(pkey, (Nsess, Nsess))
	
	rho_ut = jnp.triu(rhos) - jnp.diag( jnp.diag(rhos) ) #upper triangle components (without diagonal)
	CA = rho_ut + rho_ut.T + jnp.identity(Nsess)
	lmbds, vs = jnp.linalg.eigh(CA)
	
	while jnp.min(lmbds) < 0.0: #correlation matrix needs to be positive semi-definite
		newkey, pkey = random.split(pkey)
		rhos = -jnp.ones((Nsess, Nsess)) + 2*random.uniform(pkey, (Nsess, Nsess))
		rho_ut = jnp.triu(rhos) - jnp.diag( jnp.diag(rhos) )
		CA = rho_ut + rho_ut.T + jnp.identity(Nsess)
		lmbds, vs = jnp.linalg.eigh(CA)
		
	return CA


def calc_perms_rnd(pRkey, ctrl_size, Mperms):
	Perms_rnd = []
	rnd_pidxs = random.choice( pRkey, np.arange(len(Mperms)), (ctrl_size,), replace=False )
	for pidx in rnd_pidxs:
		Perms_rnd.append( Mperms[pidx] )
	return Perms_rnd


def simul1(params):
	fstr = 'data/tlcf2L_lst_order_simul1_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_ctrls' + str(params['ctrl_size']) + '_ik' + str(params['ik']) + ".txt"
	fw = open(fstr, 'w')
	
	key = random.PRNGKey( np.random.choice(range(999999)) + ik )
	pAkey, pRkey, key = random.split(key, num=3)
	
	Nsess = params['Nsess']
	
	params['CA'] = generate_feature_correlations(pAkey, Nsess)
	params['CB'] = np.ones((Nsess, Nsess)) #generate_feature_correlations(pBkey, Nsess)
	
	CAo = params['CA'].copy(); CBo = params['CB'].copy()
	ltmps = ""
	for q1idx in range(params['Nsess']):
		for q2idx in range(params['Nsess']):
			ltmps += str(CAo[q1idx, q2idx]) + " " + str(CBo[q1idx, q2idx]) + " "
	fw.write( ltmps + "\n" )
	
	num_epochs = params['num_epochs']
	
	Mperms = generate_Mperms(len(CAo))
	Perm_ptoc, Perm_ctop = calc_perm_typicality(CAo)
	Perm_minpaths, Perm_maxpaths = calc_perm_path_length(CAo, Mperms)
	Perms_rnd = calc_perms_rnd(pRkey, params['ctrl_size'], Mperms)

	Perms = [Perm_ptoc, Perm_ctop] + Perm_minpaths + Perm_maxpaths + Perms_rnd
	
	for Perm in Perms: 
		params['CA'] = jnp.dot(Perm, jnp.dot(CAo, Perm.T))
		errs = run_exp(params, key)
		
		for tidx in range(Nsess+1):
			ltmps = str(tidx)
			for nidx in range(Nsess):
				if tidx == 0:
					ltmps += " " + str(errs[nidx, 0])
				else:
					ltmps += " " + str(errs[nidx, tidx*num_epochs-1])
			fw.write( ltmps + "\n" )
	

if __name__ == "__main__":
	stdins = sys.argv # standard inputs

	P = int(stdins[1]) # the number of tasks
	ikmax = int(stdins[2]) # simulation id

	#env parameters
	params = {
	'Ns': 30, #dimensionality of feature space
	'Nx': 3000, #3000, #input layer width
	'Ny': 10, #output layer width
	'Nsess': P, #the number of sessions (tasks)
	'learning_rate': 0.001, #learning rate
	'num_epochs': 100, #number of epochs
	'ctrl_size': 30, #number of random permutation per 
	'ikmax': ikmax, #simulation id
	}

	for ik in range(ikmax):
		params['ik'] = ik
		simul1(params)
	
		
		
