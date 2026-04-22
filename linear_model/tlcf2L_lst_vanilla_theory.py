#
# Linear student-teacher model with latent structure
#
# Learning of arbitrary number of tasks 
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

clrs = []
cnum = 6
for cidx in range(cnum):
	clrs.append( cm.rainbow( (0.5+cidx)/cnum ) )
	
clr2s = []
cnum = 3
for cidx in range(cnum):
	clr2s.append( cm.gnuplot( (0.5+cidx)/cnum ) )
clr2s.append('0.3')


def calc_ef(CA, CB):
	#CB = np.ones((3,3)) #fully correlated Bs
	lmbds, vs = np.linalg.eigh(CA)
	if np.min(lmbds) < 0.0:
		return np.nan
	else:
		CAUinv = np.linalg.inv( np.triu(CA) )
		CAUinvCA = np.dot(CAUinv, CA)
		return np.trace( np.dot(CB, np.identity(3) - 2*CAUinvCA + np.dot(CAUinvCA, CAUinvCA.T) ) )


def calc_ef_pc(CA, CB):
	P = len(CA)
	CAUinv = np.linalg.inv( np.triu(CA) )
	CAUinvCA = np.dot(CAUinv, CA)
	return np.trace( np.dot(CB, np.identity(P) - 2*CAUinvCA + np.dot(CAUinvCA, CAUinvCA.T) ) )
	

def generate_CA3(rho_seq):
	CA = np.identity(3)
	CA[0, 1] = rho_seq[0]; CA[1, 0] = rho_seq[0]
	CA[1, 2] = rho_seq[1]; CA[2, 1] = rho_seq[1]
	CA[2, 0] = rho_seq[2]; CA[0, 2] = rho_seq[2]
	
	return CA


def calc_CAperm(CA, perm):
	P = len(CA)
	Mperm = np.zeros((P,P))
	for pidx in range(P):
		Mperm[perm[pidx], pidx] = 1
	return np.dot( Mperm, np.dot(CA, Mperm.T) )
	
	
def diagram1():
	rhoB = 1.0 #0.5
	CB = (1.0-rhoB)*np.identity(3) + rhoB*np.ones((3,3))
	
	derror = 0.000001
	drho12 = 0.005; 
	drho23 = 0.005
	
	rho12s = np.arange(0.5*drho12, 1.00, drho12)
	rho23s = np.arange(0.5*drho23, 1.00, drho23)
	rho31s = [0.0, 0.2, 0.4, 0.6, 0.8]

	rho12_mesh = np.arange(0.0, 1.0+0.5*drho12, drho12)
	rho23_mesh = np.arange(0.0, 1.0+0.5*drho23, drho23)
	X, Y = np.meshgrid(rho12_mesh, rho23_mesh)

	N12 = len(rho12s)
	N23 = len(rho23s)
	N31 = len(rho31s)
	
	min_efs = np.zeros((N12, N23, N31))
	defs_best_secondbest = np.zeros((N12, N23, N31))	
	opt_seqs = np.zeros((N12, N23, N31))
	
	for k in range(N31):
		rho31 = rho31s[k]
		for i in range(N12):
			rho12 = rho12s[i]
			for j in range(N23):
				rho23 = rho23s[j]
				rho_seq = [rho12, rho23, rho31]
				CA = generate_CA3(rho_seq)
				perms = permutations( range(3) )
				
				eftmps = [];
				h_dist = []
				for perm in list(perms):
					CAperm = calc_CAperm(CA, perm)
					eftmps.append( calc_ef(CAperm, CB) )
					h_dist.append( 2.0 - (CAperm[0,1] + CAperm[1,2]) )
				
				min_efs[i,j,k] = np.min(eftmps)
				opt_seqs[i,j,k] = np.argmin(eftmps)
				
				eftmps_sorted = sorted( eftmps )
				defs_best_secondbest[i,j,k] = eftmps_sorted[1] - eftmps_sorted[0]
				if defs_best_secondbest[i,j,k] < derror:
					#for pidx in range(6):
					for pidx in range(5,-1,-1): 
						if eftmps[pidx] - min_efs[i,j,k] < derror:
							opt_seqs[i,j,k] = pidx; break
				
				if np.isnan(min_efs[i,j,k]):
					opt_seqs[i,j,k] = np.nan

	plt.rcParams.update({'font.size':12})
	svfg1 = plt.figure(figsize=(10,5))
	
	for k in range(N31):
		plt.subplot(2, 3, k+1)
		plt.pcolor(X, Y, opt_seqs[:,:,k].T, vmin = 0.0, vmax = 6.0, cmap='Set2')
		plt.colorbar()
	
	plt.show()
	svfg1.savefig("figs/fig_tlcf2L_lst_vanilla_theory_ef_task_similarity_nl_diagram1_opt_seq_ham_rhoB" + str(rhoB) + ".pdf")


def calc_alphap(P, m):
	return ( (2-m)/(1-m) )*( (1-m)**P )


def calc_alpham(P, m):
	return -1 + ( m*P/(1-m) + (3-m)/(2-m) )*( (1-m)**P )


#plot coefficient alpha+ and alpha- (from linear perturbation theory)
def plot_alphas():
	
	Ps = [3, 5, 7, 1000]
	ms = np.arange(-0.5, 1.0, 0.01)
	
	Plen = len(Ps)
	mlen = len(ms)
	
	alphaps = np.zeros( (Plen, mlen) )
	alphams = np.zeros( (Plen, mlen) )
	
	for pidx, P in enumerate(Ps):
		for midx, m in enumerate(ms):
			alphaps[pidx, midx] = calc_alphap(P, m)
			alphams[pidx, midx] = calc_alpham(P, m)
	
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	fig1 = plt.figure()
	plt.axhline(0.0, color='k', lw=0.5)
	plt.axvline(0.0, color='k', lw=0.5)
	for pidx in range(Plen-1, -1, -1):
		plt.plot(ms, alphaps[pidx,:], color=clr2s[pidx], lw=2.5)
	plt.xlim(-0.55, 1.05)
	plt.ylim(-0.1, 5.1)
	plt.show()
	fig1.savefig("figs/tlcf2L_lst_vanilla_theory_plot_alphaps_P" + str(Ps[0]) + "-" + str(Ps[-1]) + ".pdf" )
	
	
	fig2 = plt.figure()
	plt.axhline(0.0, color='k', lw=0.5)
	plt.axvline(0.0, color='k', lw=0.5)
	for pidx in range(Plen-1, -1, -1):
		plt.plot(ms, alphams[pidx,:], color=clr2s[pidx], lw=2.5)
	plt.xlim(-0.55, 1.05)
	plt.ylim(-2.05, 1.05)
	plt.show()
	fig2.savefig("figs/tlcf2L_lst_vanilla_theory_plot_alphams_P" + str(Ps[0]) + "-" + str(Ps[-1]) + ".pdf" )


def calc_gplus(P, m, i, j):
	Gpi = - (1-m)**(P+i-1)
	Gpj = - (1-m)**(P+j-1)
	Gpij = ( (3-m)/(2-m) ) * ( (1-m)**(i+j-1) )
	return Gpi + Gpj + Gpij


def calc_gminus(P, m, i, j):
	alpham = calc_alpham(P, m)
	return alpham * ( (1-m)**(P - (j-i)) )

	
def plot_G_decompose(P, m):
	Gtot = np.zeros((P, P))
	Gplus = np.zeros((P, P)); Gminus = np.zeros((P, P))
	
	for i in range(P):
		for j in range(P):
			if j > i:
				Gplus[i,j] = calc_gplus(P, m, i, j)
				Gminus[i,j] = calc_gminus(P, m, i, j)
			else:
				Gplus[i,j] = np.nan; Gminus[i,j] = np.nan
				
			Gtot[i,j] = Gplus[i,j] + Gminus[i,j]
	
	gmax = np.nanmax(np.abs(Gtot))
	
	#plt.style.use("ggplot")
	plt.rcParams.update({'font.size':12})
	plt.ioff()
	
	fig1 = plt.figure()
	plt.subplot(1,3,1)
	plt.matshow(Gtot, fignum=0, vmin=-gmax, vmax=gmax, cmap='bwr')
	plt.xticks([]); plt.yticks([])
	plt.colorbar()
	
	plt.subplot(1,3,2)
	plt.matshow(Gplus, fignum=0, vmin=-gmax, vmax=gmax, cmap='bwr')
	plt.xticks([]); plt.yticks([])
	plt.colorbar()
	
	plt.subplot(1,3,3)
	plt.matshow(Gminus, fignum=0, vmin=-gmax, vmax=gmax, cmap='bwr')
	plt.xticks([]); plt.yticks([])
	plt.colorbar()
		
	plt.show()
	fig1.savefig("figs/fig_tlcf2L_lst_vanilla_theory_plot_G_decomposition_P" + str(P) + "_m" + str(m) + ".pdf")

	
def plot_Gplus():
	P = 7
	ms = np.arange(-0.2, 1.00, 0.01)
	mlen = len(ms)
	
	Gpluseq = np.zeros((mlen, 2*P-3))
	
	for midx, m in enumerate(ms):
		for q in range(2*P-3):
			if (q+3)%2 == 0:
				qL = (q+3)//2; qR = (q+3)//2
			else:
				qL = (q+3)//2; qR = (q+3)//2 + 1
			Gpluseq[midx, q] = calc_gplus(P, m, qL, qR)

	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	clr3s = []
	cnum = 2*P-3
	for cidx in range(cnum):
		clr3s.append( cm.viridis( (0.5+cidx)/cnum ) )
	
	fig1 = plt.figure()
	
	plt.axhline(0.0, color='k', lw=0.5)
	plt.axvline(0.0, color='k', lw=0.5)
	
	for q in range(2*P-3):
		plt.plot(ms, Gpluseq[ :, q], color=clr3s[q], lw=2.0)
	
	plt.xlim(-0.1, 1.01)
	plt.ylim(-0.75, 1.0)
	plt.show()
	
	fig1.savefig("figs/fig_tlcf2L_lst_vanilla_theory_plot_Gplus_P" + str(P) + ".pdf")
	
	
	from matplotlib.collections import PatchCollection
	from matplotlib.patches import Rectangle

	plt.style.use("default")
	fig2 = plt.figure(figsize=(5,5))
	ax2 = fig2.add_subplot()

	for i in range(P+1):
		for j in range(i+1,P+1):
			q = i+j-3
			ax2.add_patch(
				Rectangle( ((j-1)/P, 1-1/P-(i-1)/P) , 1/P, 1/P, facecolor=clr3s[q]) 
			) 
	plt.xticks(())
	plt.yticks(())
	plt.show()
	
	fig2.savefig("figs/fig_tlcf2L_lst_vanilla_theory_plot_Gplus_color_matrix_P" + str(P) + ".pdf")
	


if __name__ == "__main__":
	#stdins = sys.argv # standard inputs

	#diagram1()
	#plot_alphas()
	#plot_G_decompose(7, 0.3)
	plot_Gplus()
