#
# Linear student-teacher model with latent structure
#
# Learning of arbitrary number of tasks 
#
# Plotting
#
import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import matplotlib.pyplot as plt
from matplotlib import cm

clrs = []
cnum = 3
for cidx in range(cnum):
	clrs.append( cm.rainbow( (0.5+cidx)/cnum ) )

clr2s = ['#1f77b4', '#ff7f0e', '#2ca02c']

def calc_ef(CA, CB):
	P = len(CB)
	CAUinv = np.linalg.inv( np.triu(CA) )
	CAUinvCA = np.dot(CAUinv, CA)
	return np.trace( np.dot(CB, np.identity(P) - 2*CAUinvCA + np.dot(CAUinvCA, CAUinvCA.T) ) )


def plot_simul_trajectory(params):
	P = params['Nsess']
	num_epochs = params['num_epochs']
	rhoB = params['rhoB']
	
	CB = rhoB*np.ones((P, P)) + (1 - rhoB)*np.identity(P)
	CA = np.zeros((P, P))
	errs = np.zeros((P, P*num_epochs))
	
	fstr = 'data/tlcf2_lst1_vanilla_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ik' + str(params['ik']) + ".txt"
	
	lidx = 0
	for line in open(fstr, 'r'):
		ltmps = line[:-1].split(" ")
		if lidx == 0:
			for q1idx in range(P):
				for q2idx in range(P):
					CA[q1idx, q2idx] = ltmps[3*q1idx + q2idx]
		else:
			for qidx in range(P):
				errs[qidx, lidx-1] = float(ltmps[1+qidx])
		lidx += 1
	
	print( calc_ef(CA, CB) )
	print( ( errs[0, 3*num_epochs-1] + errs[1, 3*num_epochs-1] + errs[2, 3*num_epochs-1]) )
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	for qidx in range(P):
		plt.plot(errs[qidx], color=clrs[qidx])
	
	plt.show()
	svfg1.savefig("figs/fig_tlcf2_lst1_vanilla_plot_traj1_Nx" + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + "_lr" + str(params['learning_rate'])\
			+ "_nep" + str(params['num_epochs']) + "rhoB" + str(rhoB) + '_ik' + str(params['ik']) + ".pdf")



def plot_simul1(params):
	P = params['Nsess']
	num_epochs = params['num_epochs']
	ikmax = params['ikmax']

	ef_theory = np.zeros((ikmax))
	ef_simul = np.zeros((ikmax))
	rhos = np.zeros((ikmax, P, P))
	
	for ik in range(ikmax):
		CA = np.zeros((P, P))
		CB = np.zeros((P, P))
		errs = np.zeros((P, P+1))
	
		fstr = 'data/tlcf2L_lst_vanilla_simul1_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_ik' + str(ik) + ".txt"
	
		lidx = 0
		for line in open(fstr, 'r'):
			ltmps = line[:-1].split(" ")
			if lidx == 0:
				for q1idx in range(P):
					for q2idx in range(P):
						CA[q1idx, q2idx] = ltmps[2*(P*q1idx + q2idx)]
						CB[q1idx, q2idx] = ltmps[2*(P*q1idx + q2idx) + 1]
			else:
				for qidx in range(P):
					errs[qidx, lidx-1] = float(ltmps[1+qidx])
			lidx += 1

		rhos[ik, :, :] = CA[:,:] 
		ef_theory[ik] = calc_ef(CA, CB)
		ef_simul[ik] = np.sum(errs[:,-1])
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	efmax = np.max(ef_simul)
	
	#svfg1 = plt.figure( figsize=(5.5,5) )
	svfg1 = plt.figure( figsize=(6.4,4.8) )
	#plt.scatter(ef_theory, ef_simul, s=30, color='k')
	plt.scatter(ef_simul, ef_theory, s=30, color='k')
	plt.plot( np.arange(0.0, 1.1*efmax, 0.01), np.arange(0.0, 1.1*efmax, 0.01), color='gray' )
	plt.xlim(0.0, 1.08*efmax)
	plt.ylim(0.0, 1.08*efmax)
	#plt.xticks([0, 0.5, 1, 1.5, 2, 2.5])
	#plt.yticks([0, 0.5, 1, 1.5, 2, 2.5])
	#plt.xticks([0,0.5,1,1.5,2,2.5,3,3.5])
	plt.xticks([0, 1,2,3,4,5])
	plt.show()
	
	svfg1.savefig('figs/tlcf2L_lst_vanilla_simul1_errors_simul-theory_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_ikm' + str(ikmax) + ".pdf")


if __name__ == "__main__":
	stdins = sys.argv # standard inputs

	#env parameters
	params = {
	'Ns': 30, #dimensionality of feature space
	'Nx': 3000, #3000, #input layer width
	'Ny': 10, #output layer width
	'Nsess': 7, #the number of sessions
	'learning_rate': 0.001, #learning rate
	'num_epochs': 100, #number of epochs
	'ikmax': 100, #simulation id
	}

	plot_simul1(params)
	
		
		
