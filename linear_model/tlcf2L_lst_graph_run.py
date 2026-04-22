#
# Linear student-teacher model with latent structure
#
# Learning of tasks having graph structure
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
from tlcf2L_lst_graph import C_chain, C_ring, C_tree, C_leaves


def run_exp(params, key):
	num_epochs = params['num_epochs']
	learning_rate = params['learning_rate']
	Nsess = params['Nsess']
	
	key, abkey = random.split(key, num=2)
	
	Aseq, Bseq = generate_tasks(abkey, params)	
	
	Nx = params['Nx']; Ny = params['Ny']
	W = jnp.zeros((Ny, Nx))
	
	errors = np.zeros((Nsess, Nsess*num_epochs))
	for t in range( Nsess*num_epochs ):
		task_idx = t//num_epochs
		W = W - learning_rate*calc_dW(W, Aseq[task_idx], Bseq[task_idx])

		for nidx in range(Nsess):
			errors[nidx, t] = fnorm2( Bseq[nidx] - jnp.dot(W, Aseq[nidx]) )/Ny

	return errors


def calc_Cperm(Ctmp, perm):
	P = len(Ctmp)
	Pmat = np.zeros((P,P))
	for q in range(P):
		Pmat[ perm[q], q ] = 1
		
	return np.dot( Pmat.T, np.dot(Ctmp, Pmat) )
	

def generate_feature_correlations(pkey, Nsess):
	rhos = random.uniform(pkey, (Nsess, Nsess))
	rho_ut = jnp.triu(rhos) - jnp.diag( jnp.diag(rhos) ) #upper triangle componens (without diagonal)
	CA = rho_ut + rho_ut.T + jnp.identity(Nsess)
	lmbds, vs = jnp.linalg.eigh(CA)
	
	while jnp.min(lmbds) < 0.0: #correlation matrix needs to be positive semi-definite
		newkey, pkey = random.split(pkey)
		rhos = random.uniform(pkey, (Nsess, Nsess))
		rho_ut = jnp.triu(rhos) - jnp.diag( jnp.diag(rhos) )
		CA = rho_ut + rho_ut.T + jnp.identity(Nsess)
		lmbds, vs = jnp.linalg.eigh(CA)
		
	return CA


def generate_readout_correlations(rhoB, Nsess):
	return rhoB*jnp.ones((Nsess, Nsess)) + (1.0-rhoB)*jnp.identity(Nsess)
	

def simul_trajectory(params):
	key = random.PRNGKey(params['ik'])
	pkey, key = random.split(key)
	
	Nsess = params['Nsess']
	
	params['CA'] = generate_feature_correlations(pkey, Nsess)
	params['CB'] = generate_readout_correlations(params['rhoB'], Nsess)
	
	fstr = 'data/tlcf2L_lst_vanilla_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ik' + str(params['ik']) + ".txt"
	fw = open(fstr, 'w')

	CA = params['CA']
	ltmps = ""
	for q1idx in range(params['Nsess']):
		for q2idx in range(params['Nsess']):
			ltmps += str(CA[q1idx, q2idx]) + " "
	fw.write( ltmps + "\n" )
	
	num_epochs = params['num_epochs']
	errs = run_exp(params, key)
	for tidx in range(3*num_epochs):
		fw.write( str(tidx) + " " + str(errs[0,tidx]) + " " + str(errs[1,tidx]) + " " + str(errs[2, tidx]) + "\n" )


def simul_chain(params):
	fstr = 'data/tlcf2L_lst_graph_run_chain_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ilm' + str(params['ilmax']) + '_ik' + str(params['ik']) + ".txt"
	fw = open(fstr, 'w')
	
	key = random.PRNGKey(ik)
	pkey, key = random.split(key)
	
	Nsess = params['Nsess']
	alpha = params['alpha']
	
	CAzero = C_chain(Nsess, alpha)
	params['CB'] = generate_readout_correlations(params['rhoB'], Nsess)

	perms = [] 
	perms.append( jnp.array([0, 1, 2, 3, 4]) )
	perms.append( jnp.array([0, 4, 2, 3, 1]) )
	for il in range(ilmax):
		pkey, key = random.split(key)
		perms.append( random.permutation(pkey, jnp.arange(Nsess)) )
	
	num_epochs = params['num_epochs']
	for pidx in range( len(perms) ):
		params['CA'] = calc_Cperm(CAzero, perms[pidx])
		
		pkey, key = random.split(key)
		errs = run_exp(params, pkey)
		for tidx in range( len(errs[0]) ):
			fw.write( str(tidx) + " " + str( np.sum(errs[:,tidx]) ) + "\n" )


def simul_ring(params):
	fstr = 'data/tlcf2L_lst_graph_run_ring_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ilm' + str(params['ilmax']) + '_ik' + str(params['ik']) + ".txt"
	fw = open(fstr, 'w')
	
	key = random.PRNGKey(ik)
	pkey, key = random.split(key)
	
	Nsess = params['Nsess']
	alpha = params['alpha']
	
	CAzero = C_ring(Nsess, alpha)
	params['CB'] = generate_readout_correlations(params['rhoB'], Nsess)

	perms = [] 
	perms.append( jnp.array([0, 1, 2, 3, 4]) )
	perms.append( jnp.array([0, 2, 3, 4, 1]) )
	for il in range(ilmax):
		pkey, key = random.split(key)
		perms.append( random.permutation(pkey, jnp.arange(Nsess)) )
	
	num_epochs = params['num_epochs']
	for pidx in range( len(perms) ):
		params['CA'] = calc_Cperm(CAzero, perms[pidx])
		
		pkey, key = random.split(key)
		errs = run_exp(params, pkey)
		for tidx in range( len(errs[0]) ):
			fw.write( str(tidx) + " " + str( np.sum(errs[:,tidx]) ) + "\n" )


def simul_tree(params):
	fstr = 'data/tlcf2L_lst_graph_run_tree_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ilm' + str(params['ilmax']) + '_ik' + str(params['ik']) + ".txt"
	fw = open(fstr, 'w')
	
	key = random.PRNGKey(ik)
	pkey, key = random.split(key)
	
	Nsess = params['Nsess']
	alpha = params['alpha']
	
	CAzero = C_tree(Nsess, alpha)
	params['CB'] = generate_readout_correlations(params['rhoB'], Nsess)

	perms = [] 
	perms.append( jnp.array([0, 1, 2, 3, 4, 5, 6]) )
	perms.append( jnp.array([3, 4, 0, 5, 6, 1, 2]) )
	for il in range(ilmax):
		pkey, key = random.split(key)
		perms.append( random.permutation(pkey, jnp.arange(Nsess)) )
	
	num_epochs = params['num_epochs']
	for pidx in range( len(perms) ):
		params['CA'] = calc_Cperm(CAzero, perms[pidx])
		
		pkey, key = random.split(key)
		errs = run_exp(params, pkey)
		for tidx in range( len(errs[0]) ):
			fw.write( str(tidx) + " " + str( np.sum(errs[:,tidx]) ) + "\n" )
	
		

if __name__ == "__main__":
	stdins = sys.argv # standard inputs

	rhoB = float(stdins[1]) # global readout similarity
	alpha = float(stdins[2]) # input similarity between neighboring tasks
	ilmax = int(stdins[3]) # the number of random permutation simulation per configuration
	ikmax = int(stdins[4]) # simulation id

	#env parameters
	params = {
	'Ns': 30, #dimensionality of feature space
	'Nx': 3000, #3000, #input layer width
	'Ny': 10, #output layer width
	'Nsess': 5, #the number of sessions(tasks)
	'learning_rate': 0.001, #learning rate
	'num_epochs': 100, #number of epochs
	'rhoB': rhoB, #global correlation for B
	'alpha': alpha, #input similarity between neighboring tasks
	'ilmax': ilmax, #the number of random permutation simulation per configuration
	'ikmax': ikmax, #simulation id
	}

	for ik in range(ikmax):
		params['ik'] = ik
		
		#simul_chain(params)
		simul_ring(params)
		#simul_tree(params)
	
		
		
