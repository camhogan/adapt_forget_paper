#
# Linear student-teacher model with latent structure
#
# Generation of tasks with graph structure
#
import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from scipy import special as scisp

#chain garph
def C_chain(P, a):
	C = np.zeros((P, P))
	for i in range(P):
		for j in range(P):
			dij = abs(j-i)
			C[i,j] = a**dij
	return C


#ring graph
def C_ring(P, a):
	C = np.zeros((P, P))
	for i in range(P):
		for j in range(P):
			dij = min( abs(j-i), P - abs(j-i) )
			C[i,j] = a**dij
	return C


#generate a tree graph of depth log2(P)
def generate_tree_graph(P):
	G = np.zeros((P, P)) # tree graph
	paidx = 0; chidx = 1
	for i in range(P):
		G[chidx, paidx] = 1
		chidx += 1
		G[chidx, paidx] = 1
		chidx += 1
		paidx += 1
		if chidx >= P:
			break
	return G


#calculate distance matrix from adjacency matrix
def calc_dist_mat(Gtmp):
	P = len(Gtmp)
	D = np.full((P, P), np.inf)
	np.fill_diagonal(D, 0)
	D[Gtmp > 0] = 1
	
	# Flyod-Warshall algorithm
	for k in range(P):
		for i in range(P):
			for j in range(P):
				if D[i,j] >= D[i,k] + D[k,j]:
					D[i,j] = D[i,k] + D[k,j]
	return D
	

# tree graph
def C_tree(P, a):
	depth = int(log2(P+1))
	G = generate_tree_graph(P)
	D = calc_dist_mat(G + G.T)
	
	C = np.zeros((P, P))
	for i in range(P):
		for j in range(P):
			C[i,j] = a**D[i,j]
	
	return C	
	

# tree graph, but only leave nodes are used
def C_leaves(P, a):
	Ptot = 2*P-1 #the total graph size
	G = generate_tree_graph(Ptot)
	D = calc_dist_mat(G + G.T)

	C = np.zeros((P, P))
	for i in range(P):
		for j in range(P):
			C[i,j] = a**D[P-1+i, P-1+j]
	
	return C
	

	
	
	
	
	
