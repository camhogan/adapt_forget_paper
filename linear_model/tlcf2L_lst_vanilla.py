#
# Linear student-teacher model with latent structure
#
# Learning of arbitrary number of tasks 
#
# Numerical simulations
#
import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from tlcf2L_lst_model import generate_tasks, calc_dW, fnorm2

#import matplotlib.pyplot as plt


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
	rhos = random.uniform(pkey, (Nsess, Nsess))
	rho_ut = jnp.triu(rhos) - jnp.diag( jnp.diag(rhos) ) #upper triangle components (without diagonal)
	CA = rho_ut + rho_ut.T + jnp.identity(Nsess)
	lmbds, vs = jnp.linalg.eigh(CA)
	
	while jnp.min(lmbds) < 0.0: #correlation matrix needs to be positive semi-definite
		newkey, pkey = random.split(pkey)
		rhos = random.uniform(pkey, (Nsess, Nsess))
		rho_ut = jnp.triu(rhos) - jnp.diag( jnp.diag(rhos) )
		CA = rho_ut + rho_ut.T + jnp.identity(Nsess)
		lmbds, vs = jnp.linalg.eigh(CA)
		
	return CA


#def generate_readout_correlations(rhoB, Nsess):
#	return rhoB*jnp.ones((Nsess, Nsess)) + (1.0-rhoB)*jnp.identity(Nsess)


def simul1(params):
	fstr = 'data/tlcf2L_lst_vanilla_simul1_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_ik' + str(params['ik']) + ".txt"
	fw = open(fstr, 'w')
	
	key = random.PRNGKey( np.random.choice(range(999999)) + ik )
	pAkey, pBkey, key = random.split(key, num=3)
	
	Nsess = params['Nsess']
	
	params['CA'] = generate_feature_correlations(pAkey, Nsess)
	params['CB'] = generate_feature_correlations(pBkey, Nsess)
	
	CA = params['CA']; CB = params['CB']
	ltmps = ""
	for q1idx in range(params['Nsess']):
		for q2idx in range(params['Nsess']):
			ltmps += str(CA[q1idx, q2idx]) + " " + str(CB[q1idx, q2idx]) + " "
	fw.write( ltmps + "\n" )
	
	num_epochs = params['num_epochs']
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
	'ikmax': ikmax, #simulation id
	}

	for ik in range(ikmax):
		params['ik'] = ik
		simul1(params)
	
		
		
