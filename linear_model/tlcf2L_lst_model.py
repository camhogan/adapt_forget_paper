#
# Linear student-teacher model with latent structure
#
# Learning of arbitrary number of tasks 
#
# Vanilla model
#
import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from scipy import special as scisp


def generate_M(key, Ns, Nxy):
	return (1.0/sqrt(Ns))*random.normal(key, (Nxy, Ns))


def generate_task_seq(key, CM, Nm, params):
	P = params['Nsess']; Ns = params['Ns']
	Mseq = []
	
	lmbds, V = jnp.linalg.eigh(CM)
	sq_lmbds = jnp.sqrt(lmbds)
	Msqr = jnp.dot(V, jnp.dot( jnp.diag(sq_lmbds), V.T ) ) #square root of a given matrix
	
	Mind_seq = [] #independent sequence
	mkeys = random.split(key, P)
	for sidx in range(P):
		Mind_seq.append( generate_M(mkeys[sidx], Ns, Nm) )
	for s1idx in range(P):
		Mseq.append( jnp.zeros((Nm, Ns)) )
		for s2idx in range(P):
			Mseq[s1idx] += Msqr[s1idx, s2idx]*Mind_seq[s2idx]

	return Mseq


def generate_tasks(key, params):
	P = params['Nsess']

	akey, bkey, key = random.split(key, 3)
	
	Aseq = generate_task_seq(akey, params['CA'], params['Nx'], params)
	
	if np.sum(params['CB']) >= P*P:
		Bseq = []
		Btmp = generate_M(bkey, params['Ns'], params['Ny'])
		for sidx in range(P):
			Bseq.append( Btmp )
	else:		
		Bseq = generate_task_seq(bkey, params['CB'], params['Ny'], params)
		
	#CA = params['CA']
	#for s1idx in range(P):
	#	for s2idx in range(P):
	#		print( s1idx, s2idx, (1.0/params['Nx'])*jnp.trace( jnp.dot(Aseq[s1idx], Aseq[s2idx].T) ), CA[s1idx, s2idx] )
		
	return Aseq, Bseq


# Calculate squared frobenius norm
def fnorm2(Mtmp):
	fnorm = jnp.linalg.norm( Mtmp, ord='fro' )
	return fnorm*fnorm


@jit
def calc_dW(W, A, B): #vanilla weight update (with GD)
	return -jnp.dot(B - jnp.dot(W, A), A.T) 


