#
# Linear student-teacher model with latent structure
#
# Learning of tasks having graph structure
#
# Theory
#
import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import matplotlib.pyplot as plt
from matplotlib import cm

from itertools import permutations 
from tlcf2L_lst_graph import C_chain, C_ring, C_tree, C_leaves

clrs = []
cnum = 6
for cidx in range(cnum):
	clrs.append( cm.rainbow( (0.5+cidx)/cnum ) )


def calc_ef(CA, CB):
	P = len(CA)
	#CB = np.ones((3,3)) #fully correlated Bs
	lmbds, vs = np.linalg.eigh(CA)
	if np.min(lmbds) < 0.0:
		return np.nan
	else:
		CAUinv = np.linalg.inv( np.triu(CA) )
		CAUinvCA = np.dot(CAUinv, CA)
		return np.trace( np.dot(CB, np.identity(P) - 2*CAUinvCA + np.dot(CAUinvCA, CAUinvCA.T) ) )
	

def calc_Cperm(Ctmp, perm):
	P = len(Ctmp)
	Pmat = np.zeros((P,P))
	for q in range(P):
		Pmat[ perm[q], q ] = 1
		
	return np.dot( Pmat.T, np.dot(Ctmp, Pmat) )


def chain_graph():
	P = 5;
	aseq = np.arange(0.0, 1.005, 0.01)
	CB = np.ones((P, P))

	alen = len(aseq)
	permlen = len( list(permutations( range(P) )) )
	errs = np.full( (permlen, alen), np.inf )

	for aidx in range(alen):
		a = aseq[aidx]
		CA = C_chain(P, a)
		lmbds, vs = np.linalg.eigh(CA)
		if np.min(lmbds) >= 0.0:
			perms = permutations( range(P) )
			pidx = 0
			for perm in list(perms):
				Ctmp = calc_Cperm(CA, perm)
				errs[pidx, aidx] = calc_ef(Ctmp, CB)
				pidx += 1
	
		#print( a, list(permutations( range(P) ))[ np.argmin(errs[:,aidx]) ] )
	#perms = permutations( range(P) )
	#pidx = 0
	#for perm in list(perms):
	#	print( pidx, perm, np.ma.masked_invalid(errs[pidx, :]).mean() )
	#	pidx += 1
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':20})
	
	svfg1 = plt.figure()
	
	for pidx in range(permlen):
		plt.plot(aseq, errs[pidx, :], color = 'gray')
	plt.plot(aseq, np.mean(errs, axis=0), color = 'k', lw=3.0)	
	
	for pidx in range(permlen):
		if list(permutations( range(P) ))[pidx] == (0, 1, 2, 3, 4): 	
			plt.plot(aseq, errs[pidx, :], color = 'C0', lw=3.0)
		elif list(permutations( range(P) ))[pidx] == (0, 4, 2, 3, 1): 	
			plt.plot(aseq, errs[pidx, :], color = 'C1', lw=3.0)
		elif list(permutations( range(P) ))[pidx] == (0, 2, 4, 3, 1): 	
			plt.plot(aseq, errs[pidx, :], color = 'orange', lw=3.0)
		#elif list(permutations( range(P) ))[pidx] == (0, 4, 3, 1, 2): 	
		#	plt.plot(aseq, errs[pidx, :], color = 'C3', lw=3.0)
	plt.xlim(-0.01, 1.01)
	plt.show()
	svfg1.savefig("figs/fig_tlcf2L_lst_graph_chain_graph1_error_P" + str(P) + ".pdf")
	
	
				
def ring_graph():
	P = 5;
	aseq = np.arange(0.0, 1.005, 0.01)
	CB = np.ones((P, P))

	alen = len(aseq)
	permlen = len( list(permutations( range(P) )) )
	errs = np.full( (permlen, alen), np.inf )

	for aidx in range(alen):
		a = aseq[aidx]
		CA = C_ring(P, a)
		lmbds, vs = np.linalg.eigh(CA)
		if np.min(lmbds) >= 0.0:
			perms = permutations( range(P) )
			pidx = 0
			for perm in list(perms):
				Ctmp = calc_Cperm(CA, perm)
				errs[pidx, aidx] = calc_ef(Ctmp, CB)
				pidx += 1

	perms = permutations( range(P) )
	pidx = 0
	for perm in list(perms):
		print( pidx, perm, np.ma.masked_invalid(errs[pidx, :]).mean() )
		pidx += 1
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':20})
	
	svfg1 = plt.figure()
	
	for pidx in range(permlen):
		plt.plot(aseq, errs[pidx, :], color = 'gray')
	plt.plot(aseq, np.mean(errs, axis=0), color = 'k', lw=3.0)	
	
	for pidx in range(permlen):
		if list(permutations( range(P) ))[pidx] == (0, 1, 2, 3, 4): 	
			plt.plot(aseq, errs[pidx, :], color = 'C0', lw=3.0)
		elif list(permutations( range(P) ))[pidx] == (0, 2, 3, 4, 1): 	
			plt.plot(aseq, errs[pidx, :], color = 'C1', lw=3.0)
		elif list(permutations( range(P) ))[pidx] == (0, 2, 4, 1, 3): 	
			plt.plot(aseq, errs[pidx, :], color = 'orange', lw=3.0)
		#elif list(permutations( range(P) ))[pidx] == (0, 4, 3, 1, 2): 	
		#	plt.plot(aseq, errs[pidx, :], color = 'C3', lw=3.0)
	plt.xlim(-0.01, 1.01)
	plt.show()
	svfg1.savefig("figs/fig_tlcf2L_lst_graph_ring_graph1_error_P" + str(P) + ".pdf")


def tree_graph():
	P = 7;
	aseq = np.arange(0.0, 1.005, 0.01)
	CB = np.ones((P, P))

	alen = len(aseq)
	permlen = len( list(permutations( range(P) )) )
	errs = np.full( (permlen, alen), np.inf )

	for aidx in range(alen):
		a = aseq[aidx]
		CA = C_tree(P, a)
		lmbds, vs = np.linalg.eigh(CA)
		if np.min(lmbds) >= 0.0:
			perms = permutations( range(P) )
			pidx = 0
			for perm in list(perms):
				Ctmp = calc_Cperm(CA, perm)
				errs[pidx, aidx] = calc_ef(Ctmp, CB)
				pidx += 1

	
		print( a, list(permutations( range(P) ))[ np.argmin(errs[:,aidx]) ] )
	#perms = permutations( range(P) )
	#pidx = 0
	#for perm in list(perms):
	#	print( pidx, perm, np.ma.masked_invalid(errs[pidx, :]).mean() )
	#	pidx += 1
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':20})
	
	svfg1 = plt.figure()
	
	for pidx in range(0, permlen, 10): #sparse subsamples
		plt.plot(aseq, errs[pidx, :], color = 'gray')
	plt.plot(aseq, np.mean(errs, axis=0), color = 'k', lw=3.0)	

	perms = permutations( range(P) )
	pidx = 0
	for perm in list(perms):
		if perm == (0, 1, 2, 3, 4, 5, 6): 	
			plt.plot(aseq, errs[pidx, :], color = 'C0', lw=3.0)
		elif perm == (3, 4, 5, 6, 0, 2, 1): 	
			plt.plot(aseq, errs[pidx, :], color = 'orange', lw=3.0)
		elif perm == (3, 4, 0, 5, 6, 1, 2): 	
			plt.plot(aseq, errs[pidx, :], color = 'C1', lw=3.0)
		#elif list(permutations( range(P) ))[pidx] == (1, 5, 0, 6, 3, 4, 2): 	
		#	plt.plot(aseq, errs[pidx, :], color = 'green', lw=3.0)
		pidx += 1

	plt.xlim(-0.01, 1.01)
	plt.show()
	svfg1.savefig("figs/fig_tlcf2L_lst_graph_tree_graph1_error_P" + str(P) + ".pdf")


def leaves_graph():
	P = 8;
	aseq = np.arange(0.0, 1.005, 0.01)
	CB = np.ones((P, P))

	alen = len(aseq)
	permlen = len( list(permutations( range(P) )) )
	errs = np.full( (permlen, alen), np.inf )

	for aidx in range(alen):
		a = aseq[aidx]
		CA = C_leaves(P, a)
		lmbds, vs = np.linalg.eigh(CA)
		if np.min(lmbds) >= 0.0:
			perms = permutations( range(P) )
			pidx = 0
			for perm in list(perms):
				Ctmp = calc_Cperm(CA, perm)
				errs[pidx, aidx] = calc_ef(Ctmp, CB)
				pidx += 1

	
		print( a, list(permutations( range(P) ))[ np.argmin(errs[:,aidx]) ] )
	#perms = permutations( range(P) )
	#pidx = 0
	#for perm in list(perms):
	#	print( pidx, perm, np.ma.masked_invalid(errs[pidx, :]).mean() )
	#	pidx += 1
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':20})
	
	svfg1 = plt.figure()
	
	for pidx in range(0, permlen, 100): #sparse subsamples
		plt.plot(aseq, errs[pidx, :], color = 'gray')
	plt.plot(aseq, np.mean(errs, axis=0), color = 'k', lw=3.0)	

	perms = permutations( range(P) )
	pidx = 0
	for perm in list(perms):
		if perm == (0, 1, 2, 3, 4, 5, 6, 7): 	
			plt.plot(aseq, errs[pidx, :], color = 'C0', lw=3.0)
		elif perm == (0, 4, 2, 6, 7, 3, 5, 1): 	
			plt.plot(aseq, errs[pidx, :], color = 'C1', lw=3.0)
		elif perm == (0, 2, 4, 6, 7, 5, 3, 1): 	
			plt.plot(aseq, errs[pidx, :], color = 'orange', lw=3.0)
		pidx += 1
	"""
	for pidx in range(permlen):
		if list(permutations( range(P) ))[pidx] == (0, 1, 2, 3, 4, 5, 6, 7): 	
			plt.plot(aseq, errs[pidx, :], color = 'C0', lw=3.0)
		elif list(permutations( range(P) ))[pidx] == (0, 4, 2, 6, 7, 3, 5, 1): 	
			plt.plot(aseq, errs[pidx, :], color = 'C1', lw=3.0)
		elif list(permutations( range(P) ))[pidx] == (0, 2, 3, 1, 4, 6, 7, 5): 	
			plt.plot(aseq, errs[pidx, :], color = 'orange', lw=3.0)
		#elif list(permutations( range(P) ))[pidx] == (1, 5, 0, 6, 3, 4, 2): 	
		#	plt.plot(aseq, errs[pidx, :], color = 'green', lw=3.0)
	"""
	
	plt.xlim(-0.01, 1.01)
	plt.show()
	svfg1.savefig("figs/fig_tlcf2L_lst_graph_leaves_graph1_error_P" + str(P) + ".pdf")
	

if __name__ == "__main__":
	#stdins = sys.argv # standard inputs
	chain_graph()
	ring_graph()
	#tree_graph()
	#leaves_graph()
	
	
	
