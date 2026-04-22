#
# Linear student-teacher model with latent structure
#
# Learning of tasks having graph structure
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


def plot_simul_graph_trajectory(params, graph_style):
	P = params['Nsess']
	num_epochs = params['num_epochs']
	rhoB = params['rhoB']
	
	ilmax = params['ilmax']
	ikmax = params['ikmax']
	
	def_errs = np.zeros((ikmax, P*num_epochs))
	opt_errs = np.zeros((ikmax, P*num_epochs))
	rnd_errs = np.zeros((ikmax*ilmax, P*num_epochs))
	
	for ik in range(ikmax):
		fstr = 'data/tlcf2L_lst_graph_run_' + graph_style + '_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ilm' + str(params['ilmax']) + '_ik' + str(ik) + ".txt"	

		pidx = 0; tidx = 0
		for line in open(fstr, 'r'):
			ltmps = line[:-1].split(" ")
			if pidx == 0:
				def_errs[ik, tidx] = float(ltmps[1])
			elif pidx == 1:
				opt_errs[ik, tidx] = float(ltmps[1])
			else:
				rnd_errs[ik*ilmax + pidx-2, tidx] = float(ltmps[1])
			tidx += 1
			if tidx == P*num_epochs:
				tidx = 0; pidx += 1
	
	ts = range(P*num_epochs)
	def_errs_mean = np.mean(def_errs, axis=0); def_errs_ste = (1.0/sqrt(ikmax))*np.std(def_errs, axis=0)
	opt_errs_mean = np.mean(opt_errs, axis=0); opt_errs_ste = (1.0/sqrt(ikmax))*np.std(opt_errs, axis=0)
	rnd_errs_mean = np.mean(rnd_errs, axis=0); rnd_errs_ste = (1.0/sqrt(ikmax*ilmax))*np.std(rnd_errs, axis=0) 
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	plt.fill_between(ts, def_errs_mean + def_errs_ste, def_errs_mean - def_errs_ste, color='C0', alpha=0.25)
	plt.fill_between(ts, opt_errs_mean + opt_errs_ste, opt_errs_mean - opt_errs_ste, color='C1', alpha=0.25)
	plt.fill_between(ts, rnd_errs_mean + rnd_errs_ste, rnd_errs_mean - rnd_errs_ste, color='k', alpha=0.25)
	
	plt.plot(def_errs_mean, color='C0', lw=2.0)
	plt.plot(opt_errs_mean, color='C1', lw=2.0)
	plt.plot(rnd_errs_mean, color='k', lw=2.0)

	if graph_style == 'chain':
		plt.xlim(-10, 510); plt.ylim(-0.1, 3.1)
	elif graph_style == 'ring':
		plt.xlim(-10, 510); plt.ylim(-0.1, 3.1)
	elif graph_style == 'tree':
		plt.xlim(-10, 710); plt.ylim(-0.1, 5.1)
	plt.show()
	svfg1.savefig('figs/tlcf2L_lst_graph_run_' + graph_style + '_errors_plot_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ilm' + str(params['ilmax']) + '_ikm' + str(ikmax) + '.pdf')


def plot_simul_chain_trajectory(params):
	P = params['Nsess']
	num_epochs = params['num_epochs']
	rhoB = params['rhoB']
	
	ilmax = params['ilmax']
	ikmax = params['ikmax']
	
	def_errs = np.zeros((ikmax, P*num_epochs))
	opt_errs = np.zeros((ikmax, P*num_epochs))
	rnd_errs = np.zeros((ikmax*ilmax, P*num_epochs))
	
	for ik in range(ikmax):
		fstr = 'data/tlcf2L_lst_graph_run_chain_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ilm' + str(params['ilmax']) + '_ik' + str(ik) + ".txt"	

		pidx = 0; tidx = 0
		for line in open(fstr, 'r'):
			ltmps = line[:-1].split(" ")
			if pidx == 0:
				def_errs[ik, tidx] = float(ltmps[1])
			elif pidx == 1:
				opt_errs[ik, tidx] = float(ltmps[1])
			else:
				rnd_errs[ik*ilmax + pidx-2, tidx] = float(ltmps[1])
			tidx += 1
			if tidx == P*num_epochs:
				tidx = 0; pidx += 1
	
	ts = range(P*num_epochs)
	def_errs_mean = np.mean(def_errs, axis=0); def_errs_ste = (1.0/sqrt(ikmax))*np.std(def_errs, axis=0)
	opt_errs_mean = np.mean(opt_errs, axis=0); opt_errs_ste = (1.0/sqrt(ikmax))*np.std(opt_errs, axis=0)
	rnd_errs_mean = np.mean(rnd_errs, axis=0); rnd_errs_ste = (1.0/sqrt(ikmax*ilmax))*np.std(rnd_errs, axis=0) 
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':20})
	
	svfg1 = plt.figure()
	plt.fill_between(ts, def_errs_mean + def_errs_ste, def_errs_mean - def_errs_ste, color='C0', alpha=0.25)
	plt.fill_between(ts, opt_errs_mean + opt_errs_ste, opt_errs_mean - opt_errs_ste, color='C1', alpha=0.25)
	plt.fill_between(ts, rnd_errs_mean + rnd_errs_ste, rnd_errs_mean - rnd_errs_ste, color='k', alpha=0.25)
	
	plt.plot(def_errs_mean, color='C0', lw=2.0)
	plt.plot(opt_errs_mean, color='C1', lw=2.0)
	plt.plot(rnd_errs_mean, color='k', lw=2.0)

	plt.ylim(-0.1, 3.1)
	plt.xlim(-10, 510)
	plt.show()
	svfg1.savefig('figs/tlcf2L_lst_graph_run_chain_errors_plot_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ilm' + str(params['ilmax']) + '_ikm' + str(ikmax) + '.pdf')


def plot_simul_tree_trajectory(params):
	P = params['Nsess']
	num_epochs = params['num_epochs']
	rhoB = params['rhoB']
	
	ilmax = params['ilmax']
	ikmax = params['ikmax']
	
	def_errs = np.zeros((ikmax, P*num_epochs))
	opt_errs = np.zeros((ikmax, P*num_epochs))
	rnd_errs = np.zeros((ikmax*ilmax, P*num_epochs))
	
	for ik in range(ikmax):
		fstr = 'data/tlcf2L_lst_graph_run_tree_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ilm' + str(params['ilmax']) + '_ik' + str(ik) + ".txt"	

		pidx = 0; tidx = 0
		for line in open(fstr, 'r'):
			ltmps = line[:-1].split(" ")
			if pidx == 0:
				def_errs[ik, tidx] = float(ltmps[1])
			elif pidx == 1:
				opt_errs[ik, tidx] = float(ltmps[1])
			else:
				rnd_errs[ik*ilmax + pidx-2, tidx] = float(ltmps[1])
			tidx += 1
			if tidx == P*num_epochs:
				tidx = 0; pidx += 1
	
	ts = range(P*num_epochs)
	def_errs_mean = np.mean(def_errs, axis=0); def_errs_ste = (1.0/sqrt(ikmax))*np.std(def_errs, axis=0)
	opt_errs_mean = np.mean(opt_errs, axis=0); opt_errs_ste = (1.0/sqrt(ikmax))*np.std(opt_errs, axis=0)
	rnd_errs_mean = np.mean(rnd_errs, axis=0); rnd_errs_ste = (1.0/sqrt(ikmax*ilmax))*np.std(rnd_errs, axis=0) 
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':20})
	
	svfg1 = plt.figure()
	plt.fill_between(ts, def_errs_mean + def_errs_ste, def_errs_mean - def_errs_ste, color='C0', alpha=0.25)
	plt.fill_between(ts, opt_errs_mean + opt_errs_ste, opt_errs_mean - opt_errs_ste, color='C1', alpha=0.25)
	plt.fill_between(ts, rnd_errs_mean + rnd_errs_ste, rnd_errs_mean - rnd_errs_ste, color='k', alpha=0.25)
	
	plt.plot(def_errs_mean, color='C0', lw=2.0)
	plt.plot(opt_errs_mean, color='C1', lw=2.0)
	plt.plot(rnd_errs_mean, color='k', lw=2.0)

	plt.ylim(-0.1, 5.1)
	plt.xlim(-10, 710)
	plt.show()
	svfg1.savefig('figs/tlcf2L_lst_graph_run_tree_errors_plot_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoB' + str(params['rhoB']) + '_ilm' + str(params['ilmax']) + '_ikm' + str(ikmax) + '.pdf')




if __name__ == "__main__":
	stdins = sys.argv # standard inputs

	#env parameters
	params = {
	'Ns': 30, #dimensionality of feature space
	'Nx': 3000, #3000, #input layer width
	'Ny': 10, #output layer width
	'Nsess': 5, #the number of sessions
	'learning_rate': 0.001, #learning rate
	'num_epochs': 100, #number of epochs
	'rhoB': 1.0, # global readout correlation
	'ilmax': 100, # number of random task permutations
	'ikmax': 10, # simulation id
	}

	graph_style = 'chain' # 'chain'/'ring'/'tree'/'leaves'
	plot_simul_graph_trajectory(params, graph_style)
	#plot_simul_tree_trajectory(params)
	#plot_simul1(params)
	
		
		
