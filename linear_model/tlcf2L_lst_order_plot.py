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
cnum = 3 #5
for cidx in range(cnum):
	#clrs.append( cm.rainbow( (0.5+cidx)/cnum ) )
	clrs.append( cm.gnuplot( (0.5+cidx)/cnum ) )

clr2s = ['#1f77b4', '#ff7f0e', '#2ca02c']

def calc_ef(CA, CB):
	P = len(CB)
	CAUinv = np.linalg.inv( np.triu(CA) )
	CAUinvCA = np.dot(CAUinv, CA)
	return np.trace( np.dot(CB, np.identity(P) - 2*CAUinvCA + np.dot(CAUinvCA, CAUinvCA.T) ) )


def osim_data_processing(params):
	P = params['Nsess']
	num_epochs = params['num_epochs']
	ctrl_size = params['ctrl_size']
	ikmax = params['ikmax']

	ef_simul = np.zeros((5, ikmax))
	rhos = np.zeros((ikmax))
	
	for ik in range(ikmax):
		CA = np.zeros((P, P))
		CB = np.zeros((P, P))
		errs = np.zeros((6+ctrl_size, P, P+1))
	
		fstr = 'data/tlcf2L_lst_order_simul1_errors_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_ctrls' + str(params['ctrl_size']) + '_ik' + str(ik) + ".txt"
	
		# file is organized as 
		# periphery-to-core, core-to-periphery, Min-paths on similarity matrix, Max-paths on similarity matrix, random
	
		lidx = 0; tidx = 0; oidx = 0
		for line in open(fstr, 'r'):
			ltmps = line[:-1].split(" ")
			if lidx == 0:
				for q1idx in range(P):
					for q2idx in range(P):
						CA[q1idx, q2idx] = ltmps[2*(P*q1idx + q2idx)]
						CB[q1idx, q2idx] = ltmps[2*(P*q1idx + q2idx) + 1]
			else:
				for qidx in range(P):
					errs[oidx, qidx, tidx] = float(ltmps[1+qidx])
				tidx += 1
				if tidx == P+1:
					tidx = 0; oidx += 1
			lidx += 1

		rhos[ik] = (np.sum(CA) - P)/( P*(P-1) )
		for oidx in range(2): 	
			ef_simul[oidx, ik] = np.sum(errs[oidx, :, -1])
		for oidx in range(2): #summation over two max Ham paths (and sum over two min Ham paths)
			ef_simul[2+oidx, ik] = 0.5*np.sum(errs[2+2*oidx:4+2*oidx, :, -1])	
		ef_simul[4, ik] = (1/ctrl_size)*np.sum(errs[6:,:,-1])
	
	drho = 0.05
	rhomin = -0.5; 
	rhomax = 0.5001
	
	rhoaxs = np.arange(rhomin, rhomax, drho)
	rhocs = 0.5*(rhoaxs[:-1] + rhoaxs[1:])
	rlen = len(rhocs)
	
	bin_cnts = np.zeros((rlen))
	bin_efs = []
	for q in range(5):
		bin_efs.append([])
		for rhidx in range(rlen):
			bin_efs[q].append([])
	
	for ik in range(ikmax):
		rhotmp = rhos[ik]
		if rhomin <= rhotmp and rhotmp < rhomax:
			rhidx = int(floor( (rhotmp - rhomin)/drho ))
			bin_cnts[rhidx] += 1
			for q in range(5):
				bin_efs[q][rhidx].append( ef_simul[q, ik] )
	
	bin_ef_mean = np.zeros((5, rlen))
	bin_ef_se = np.zeros((5, rlen))
	bin_ef_raw_mean = np.zeros((5, rlen))
	bin_ef_raw_se = np.zeros((5, rlen))
	for q in range(5):
		for rhidx in range(rlen):
			bin_er_mean_base = np.mean(bin_efs[-1][rhidx])
			if len(bin_efs[q][rhidx]) > 1:
				bin_ef_mean[q, rhidx] = np.mean(bin_efs[q][rhidx])/bin_er_mean_base
				bin_ef_se[q, rhidx] = np.std(bin_efs[q][rhidx])/( np.sqrt( len(bin_efs[q][rhidx]) )*bin_er_mean_base )
				bin_ef_raw_mean[q, rhidx] = np.mean(bin_efs[q][rhidx])
				bin_ef_raw_se[q, rhidx] = np.std(bin_efs[q][rhidx])/np.sqrt( len(bin_efs[q][rhidx]) )
			else:
				bin_ef_mean[q, rhidx] = np.nan; bin_ef_se[q, rhidx] = np.nan; 
				bin_ef_raw_mean[q, rhidx] = np.nan; bin_ef_raw_se[q, rhidx] = np.nan; 
	
	bin_ef_rate = np.zeros((2, rlen))
	bin_ef_rate_se = np.zeros((2, rlen))
	for rhidx in range(rlen):
		brhlen = len(bin_efs[0][rhidx]) 
		if brhlen > 1:
			for ik in range(brhlen):
				if bin_efs[0][rhidx][ik] < bin_efs[1][rhidx][ik]:
					bin_ef_rate[0, rhidx] += 1/brhlen
				if bin_efs[2][rhidx][ik] < bin_efs[3][rhidx][ik]:
					bin_ef_rate[1, rhidx] += 1/brhlen
			for q in range(2):
				alpha = bin_ef_rate[q, rhidx]*brhlen
				beta = brhlen - alpha
				if alpha > 0.0 and beta > 0.0:
					bin_ef_rate_se[q,rhidx] = np.sqrt( alpha*beta/( (alpha+beta)*(alpha+beta)*(alpha+beta+1) ) )
				else:
					bin_ef_rate_se[q,rhidx] = 0.0
		else:
			for q in range(2):
				bin_ef_rate[q, rhidx] = np.nan
				bin_ef_rate_se[q, rhidx] = np.nan
		
	return rhocs, bin_ef_mean, bin_ef_se, bin_ef_raw_mean, bin_ef_raw_se, bin_ef_rate, bin_ef_rate_se


def plot_simul1(params):
	params['Nsess'] = 5 #7
	params['ctrl_size'] = 30#100
	
	rhocs, bin_ef_mean, bin_ef_se, bin_ef_raw_mean, bin_ef_raw_se, bin_ef_rate, bin_ef_rate_se = osim_data_processing(params)

	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	plt.axvline(0.0, color='k', lw=0.5)
	for bidx, clr in zip([0,1,4], ['red', 'blue', 'black']):
		if bidx != 4:
			plt.fill_between(rhocs, bin_ef_mean[bidx]+bin_ef_se[bidx], bin_ef_mean[bidx]-bin_ef_se[bidx], color=clr, alpha=0.25)
		plt.plot(rhocs, bin_ef_mean[bidx], 'o-', color=clr)
	plt.show()
	svfg1.savefig('figs/tlcf2L_lst_order_simul1_rel_errors_periphery-core_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_ctrls' + str(params['ctrl_size']) + '_ikm' + str(params['ikmax']) + ".pdf")
	
	svfg2 = plt.figure()
	plt.axvline(0.0, color='k', lw=0.5)
	for bidx, clr in zip([2,3,4], ['red', 'blue', 'black']):
		if bidx != 4:
			plt.fill_between(rhocs, bin_ef_mean[bidx]+bin_ef_se[bidx], bin_ef_mean[bidx]-bin_ef_se[bidx], color=clr, alpha=0.25)
		plt.plot(rhocs, bin_ef_mean[bidx], 'o-', color=clr)
	plt.show()
	svfg2.savefig('figs/tlcf2L_lst_order_simul1_rel_errors_min-max-rule_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_ctrls' + str(params['ctrl_size']) + '_ikm' + str(params['ikmax']) + ".pdf")

	svfg3 = plt.figure()
	plt.axvline(0.0, color='k', lw=0.5)
	for bidx, clr in zip([0,1,4], ['red', 'blue', 'black']):
		plt.fill_between(rhocs, bin_ef_raw_mean[bidx]+bin_ef_raw_se[bidx], bin_ef_raw_mean[bidx]-bin_ef_raw_se[bidx], color=clr, alpha=0.25)
		plt.plot(rhocs, bin_ef_raw_mean[bidx], 'o-', color=clr)
	#plt.semilogy()
	plt.show()
	svfg3.savefig('figs/tlcf2L_lst_order_simul1_raw_errors_periphery-core_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_ctrls' + str(params['ctrl_size']) + '_ikm' + str(params['ikmax']) + ".pdf")
	
	svfg4 = plt.figure()
	plt.axvline(0.0, color='k', lw=0.5)
	for bidx, clr in zip([2,3,4], ['red', 'blue', 'black']):
		plt.fill_between(rhocs, bin_ef_raw_mean[bidx]+bin_ef_raw_se[bidx], bin_ef_raw_mean[bidx]-bin_ef_raw_se[bidx], color=clr, alpha=0.25)
		plt.plot(rhocs, bin_ef_raw_mean[bidx], 'o-', color=clr)
	#plt.semilogy()
	plt.show()
	svfg4.savefig('figs/tlcf2L_lst_order_simul1_raw_errors_min-max-rule_Nx' + str(params['Nx']) + '_Nsess' + str(params['Nsess']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_ctrls' + str(params['ctrl_size']) + '_ikm' + str(params['ikmax']) + ".pdf")



def plot_simul2(params):
	Ps = [3, 5, 7]
	ctrl_sizes = [6, 30, 100]
	alphas = [0.2, 0.6, 1.0]
	Plen = len(Ps)

	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	plt.axhline(1.0, color='k', lw=2.0)
	for pidx in range( Plen ):
		params['Nsess'] = Ps[pidx]
		params['ctrl_size'] = ctrl_sizes[pidx]
		
		rhocs, bin_ef_mean, bin_ef_se, bin_ef_raw_mean, bin_ef_raw_se, bin_ef_rate = plot_simul1(params)
		for qidx, clr in zip([0,1], ['blue', 'red']):
			plt.plot(rhocs, bin_ef_mean[qidx], 'o-', color=clr, lw=2.0, alpha=alphas[pidx] )
	plt.show()
	
	svfg2 = plt.figure()
	plt.axhline(1.0, color='k', lw=2.0)
	for pidx in range( Plen ):
		params['Nsess'] = Ps[pidx]
		params['ctrl_size'] = ctrl_sizes[pidx]
		
		rhocs, bin_ef_mean, bin_ef_se, bin_ef_raw_mean, bin_ef_raw_se, bin_ef_rate = plot_simul1(params)
		for qidx, clr in zip([2,3], ['blue', 'red']):
			plt.plot(rhocs, bin_ef_mean[qidx], 'o-', color=clr, lw=2.0, alpha=alphas[pidx] )
	plt.show()
	
	svfg3 = plt.figure()
	plt.axhline(1.0, color='k', lw=2.0)
	for pidx in range( Plen ):
		params['Nsess'] = Ps[pidx]
		params['ctrl_size'] = ctrl_sizes[pidx]
		
		rhocs, bin_ef_mean, bin_ef_se, bin_ef_raw_mean, bin_ef_raw_se, bin_ef_rate = plot_simul1(params)
		for qidx, clr in zip([0,1,4], ['blue', 'red', 'black']):
			plt.plot(rhocs, bin_ef_raw_mean[qidx], 'o-', color=clr, lw=2.0, alpha=alphas[pidx] )
	plt.show()
	svfg1.savefig('figs/tlcf2L_lst_order_plot_simul3_error_rate_periphery-core_Nx' + str(params['Nx']) + '_Nsess' + str(Ps[0]) + '-' + str(Ps[-1]) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_ikm' + str(ikmax) + ".pdf")
	
	svfg2 = plt.figure()
	plt.axhline(1.0, color='k', lw=2.0)
	for pidx in range( Plen ):
		params['Nsess'] = Ps[pidx]
		params['ctrl_size'] = ctrl_sizes[pidx]
		
		rhocs, bin_ef_mean, bin_ef_se, bin_ef_raw_mean, bin_ef_raw_se, bin_ef_rate = plot_simul1(params)
		for qidx, clr in zip([2,3,4], ['blue', 'red', 'black']):
			plt.plot(rhocs, bin_ef_raw_mean[qidx], 'o-', color=clr, lw=2.0, alpha=alphas[pidx] )
	plt.show()
	svfg2.savefig('figs/tlcf2L_lst_order_plot_simul3_error_rate_min-max-path_Nx' + str(params['Nx']) + '_Nsess' + str(Ps[0]) + '-' + str(Ps[-1]) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_ikm' + str(ikmax) + ".pdf")


def plot_simul3(params):
	Ps = [3, 5, 7]
	ctrl_sizes = [6, 30, 100]
	alphas = [0.2, 0.6, 1.0]
	Plen = len(Ps)

	rhocss = []
	bin_ef_rates = []
	bin_ef_rate_ses = []
	for pidx in range( Plen ):
		params['Nsess'] = Ps[pidx]
		params['ctrl_size'] = ctrl_sizes[pidx]
		
		rhocs, bin_ef_mean, bin_ef_se, bin_ef_raw_mean, bin_ef_raw_se, bin_ef_rate,  bin_ef_rate_se = osim_data_processing(params)
		rhocss.append(rhocs)
		bin_ef_rates.append(bin_ef_rate)
		bin_ef_rate_ses.append(bin_ef_rate_se)

	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	plt.axvline(0.0, color='k', lw=0.5)
	plt.axhline(0.5, ls='--', color='k', lw=1.0)
	
	for pidx in range( Plen ):
		plt.fill_between(rhocss[pidx], bin_ef_rates[pidx][0]+bin_ef_rate_ses[pidx][0], bin_ef_rates[pidx][0]-bin_ef_rate_ses[pidx][0], color=clrs[pidx], alpha=0.25) 
		plt.plot(rhocss[pidx], bin_ef_rates[pidx][0], 'o-', color=clrs[pidx], lw=2.0)
	plt.show()
	svfg1.savefig('figs/tlcf2L_lst_order_plot_simul3_error_rate_periphery-core_Nx' + str(params['Nx']) + '_Nsess' + str(Ps[0]) + '-' + str(Ps[-1]) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_ikm' + str(params['ikmax']) + ".pdf")
	
	svfg2 = plt.figure()
	plt.axvline(0.0, color='k', lw=0.5)
	plt.axhline(0.5, ls='--', color='k', lw=1.0)

	for pidx in range( Plen ):
		plt.fill_between(rhocss[pidx], bin_ef_rates[pidx][1]+bin_ef_rate_ses[pidx][1], bin_ef_rates[pidx][1]-bin_ef_rate_ses[pidx][1], color=clrs[pidx], alpha=0.25) 
		plt.plot(rhocss[pidx], bin_ef_rates[pidx][1], 'o-', color=clrs[pidx], lw=2.0)
	plt.show()
	svfg2.savefig('figs/tlcf2L_lst_order_plot_simul3_error_rate_min-max-path_Nx' + str(params['Nx']) + '_Nsess' + str(Ps[0]) + '-' + str(Ps[-1]) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_ikm' + str(params['ikmax']) + ".pdf")



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
	'ctrl_size': 30, # the number of trials used for control
	'ikmax': 1000, #simulation id
	}

	plot_simul1(params)
	#plot_simul2(params)
	#plot_simul3(params)
		
