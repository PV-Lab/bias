import numpy
import sys
import h5
import os
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.factory import get_termination, get_reference_directions, get_algorithm
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling
#from pymoo.util.display import SingleObjectiveDisplay
import numpy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from scipy import interpolate
import addict
import emcee
import de
import de_snooker
import emcee.state
import scipy.spatial
from sklearn.cluster import KMeans
import emcee.autocorr as autocorr
import pandas
import datetime
import corner
from NN_training import log_mse
import time
import mpmath as mp
import pandas as pd
import csv

def new_range(flat_chain):
    '''
    new_range : to set range of corner plot

    Parameters
    ----------
    flat_chain : flat chain for plotting

    Returns
    -------
    new range for lower and upper bound
    '''
    lb_data, mid_data, ub_data = numpy.percentile(flat_chain, [5, 50, 95], 0)

    distance = numpy.max(
        numpy.abs(numpy.array([lb_data - mid_data, ub_data - mid_data])), axis=0
    )

    lb_range = mid_data - 2 * distance
    ub_range = mid_data + 2 * distance

    return numpy.array([lb_range, ub_range])


def run_walkers_2(auto_state,sampler,data,cur_path):
    '''
    run_walkers: final mcmc run
    '''

    chain = None
    prob = None
    taus = None
    tau_check = 100
    finished = False
    step = 0
    tol = 52

    state = auto_state
    mother_chain = []
    mother_prob = []

    while not finished:
        state = next(sampler.sample(state, iterations=1))

        p = state.coords
        ln_prob = state.log_prob
        mother_chain.append(p)
        mother_prob.append(ln_prob)

        chain = addChain(chain, p[:,numpy.newaxis,:])
        prob = addChain(prob, ln_prob[:,numpy.newaxis])

        # Check and save the autocorrelation time at every step
        try:
                tau = sampler.get_autocorr_time(tol=tol)
            
                print(f"step {step}  progress {step/(tau*tol)}")

                if not numpy.any(numpy.isnan(tau)):
                    finished = True

        except autocorr.AutocorrError as err:
                tau = err.tau
            
                #print(f"step {step}  progress {step/(tau*tol)}")

        taus = addChain(taus, tau[:,numpy.newaxis])

        if step and step % 500 == 0:
            motherchainT = numpy.swapaxes(mother_chain, 0, 1)
            motherprobT = numpy.swapaxes(mother_prob, 0, 1)

            plot_mixing(motherchainT,cur_path)
            

            data.root.motherchain = motherchainT
            data.root.motherprob = motherprobT
            data.save()
            
            
            if not numpy.any(numpy.isnan(tau)):
                burn = int(numpy.ceil(numpy.max(tau)) * 2)

                chain = chain[:,burn:,:]
                prob = prob[:,burn:]
            #plot_check(state, sampler)          # to plot jv curves from the current position of all walkeres overlayed on top of the jv exp
        
        if step == 300:
            finished = True
        step = step + 1

    mother_chain = numpy.array(mother_chain)
    mother_prob = numpy.array(mother_prob)
    motherchainT = numpy.swapaxes(mother_chain, 0, 1)
    motherprobT = numpy.swapaxes(mother_prob, 0, 1)

    plot_mixing(motherchainT,cur_path)
    print("finished")

    data.root.motherchain = motherchainT
    data.root.motherprob = motherprobT
    data.save()
    return motherchainT, motherprobT

cm_plot = plt.cm.viridis

def get_color(idx, max_colors, cmap):
    return cmap(1.0 * float(idx) / max_colors)

def flatten(chain):
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
    return flat_chain

def plot_mixing(chain,cur_path):
    f_mixing = Path(cur_path)/("mixing")
    if os.path.exists(f_mixing):
        print("file already exists")
    else:
        os.mkdir(f_mixing)
    chain_length = chain.shape[1]
    x = numpy.linspace(0, chain_length - 1, chain_length)
    for i in range(chain.shape[2]):
        plt.figure(figsize=[15,7])
        
        for j in range(chain.shape[0]):
            plt.plot(x, chain[j, :, i], color=get_color(j, chain.shape[0] - 1, cm_plot))
        fig_path = Path(f_mixing)/("mixing_%i.png" %i)
        plt.savefig(fig_path)
        plt.close('all')
        

    flat_chain = flatten(chain)

    fig = corner.corner(
        flat_chain,
        quantiles=(0.05, 0.5, 0.95),
        show_titles=True,
        bins=20,
        range=new_range(flat_chain).T,
        use_math_text=True,
        title_fmt=".2g",
    )
    plt.savefig(Path(f_mixing)/("corner.png"))
    plt.close('all')


def plot_start(auto_state, y_exp_norm, reg, batchsize, scaler, lb, ub,cur_path, data):
    '''
    plot_stat : plot the predicted Y for all the starting points super imposed on the normalized experimental data

    Parameters: 
    auto_state : coords of all the walkers
    y_exp_norm : numpy array of normalized experimental data
    reg : regression model
    scaler : scaler to tansform the parameteres
    lb: lower bound
    ub : upper bound
    cur_path : where the figures will be stores
    '''
    f_high_start = Path(cur_path)/("high_start")
    if os.path.exists(f_high_start):
        print("file already exists")
    else:
        os.mkdir(f_high_start)

    coords = auto_state.coords
    #coords = auto_state
    lb = numpy.log(lb)
    ub = numpy.log(ub)
    theta_trans = numpy.array(coords) * (ub-lb) + lb
    theta_actual = numpy.exp(theta_trans)
    
    theta_norm = theta_transform(theta_trans, scaler)
    jv_predicted_from_theta = reg.predict(theta_norm, batch_size = batchsize)

    data.root.walkers_start_actual = theta_actual
    data.root.jv_predict = jv_predicted_from_theta
    data.save()

    
    for i in range(jv_predicted_from_theta.shape[0]):
        plt.figure(figsize=[10,10])
        plt.plot(jv_predicted_from_theta[i,:], label="nn")
        plt.plot(numpy.squeeze(y_exp_norm), label="target")
        plt.title(f"{theta_actual[i,:]}")
        plt.legend()
        fig_path = Path(f_high_start)/("prob_%i.png" %i)
        plt.savefig(fig_path)
        plt.close('all')
    
    
def auto_high_probability(sampler, start, iterations=100):
    '''
    auto_high_probability : relocates all the walkers to high probability region

    Parameters 
    ----------
    sampler : emcee sampler model
    start : starting point for all the walkers
    interation : number of generations to run the emcee sampler before stopping

    Returns
    -------
    state : high probability starting point for all the walkers
    auto_chain : chain from the mcmc run
    auto_probability : probability of all the positions in the chain

    '''
    auto_chain = None
    auto_probability = None
    
    state = emcee.state.State(start)
    
    log_prob = None
    rstart = None
    
    finished = None
    prev_prob = -1e308

    while not finished:
        chain, probability = auto_high_probability_iterations(sampler, iterations, state)

        # store chain
        auto_chain = addChain(auto_chain, chain)
        auto_probability = addChain(auto_probability, probability)

        best_chain, best_prob = select_best_kmeans(auto_chain, auto_probability)

        #print(f"best_chain {best_chain}")
        #print(f"best_prob {best_prob}")
        
        state.coords = best_chain
        state.log_prob = best_prob
        state.random_state = None
        
        best_prob = numpy.max(best_prob)
        
        change = numpy.abs(best_prob - prev_prob)/numpy.abs(prev_prob)
        min_prob = 1
        if change < 0.001 and  best_prob > min_prob:
            finished=True
        else:
            print(f"Auto high probability has not converged yet, change {change} > 0.001")
            prev_prob = best_prob
        
    return state, auto_chain, auto_probability

def auto_high_probability_iterations(sampler, iterations, state):

    auto_chain = None
    auto_probability = None
    best = -numpy.inf

    for i in range(iterations):
        state = next(
            sampler.sample(
                state,
                iterations=1,
            )
        )

        p = state.coords
        ln_prob = state.log_prob
        random_state = state.random_state

        if any(ln_prob > best):
            best = numpy.max(ln_prob)

        accept = numpy.mean(sampler.acceptance_fraction)

        auto_chain = addChain(auto_chain, p[:, numpy.newaxis, :])
        auto_probability = addChain(auto_probability, ln_prob[:, numpy.newaxis])

        if i%10 == 0:
            print(f"auto run: idx: {i} accept: {accept:.3f} max ln(prob): {best:.5f}")

    sampler.reset()
    return auto_chain, auto_probability

def select_best_kmeans(chain, probability):
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
    flat_probability = numpy.squeeze(probability.reshape(-1, 1))

    # unique
    flat_chain_unique, unique_indexes = numpy.unique(
        flat_chain, return_index=True, axis=0
    )
    flat_probability_unique = flat_probability[unique_indexes]

    '''
    Before 06.16.2025
    # remove low probability
    flat_prob = numpy.exp(flat_probability_unique)
    max_prob = numpy.max(flat_prob)
    min_prob = max_prob / 5  # 10% of max prob cutoff
    '''
    # remove low probability
    flat_prob = (flat_probability_unique)
    max_prob = numpy.max(flat_prob)

    #error = numpy.full_like(512, fill_value=1, dtype=float) * 0.1
    #sigma = 1e-4
    #min_ln_pdf = -0.5*numpy.sum((((error)**2/(sigma**2)) + numpy.log(2*3.14*(sigma**2))), axis = 1)
    min_prob = 1

    #changed on 06.16.2025
    #selected = (flat_prob >= min_prob) & (flat_prob <= max_prob)

    selected = (flat_prob >= min_prob) #& (flat_prob <= max_prob)

    flat_chain = flat_chain_unique[selected]
    flat_probability = flat_probability_unique[selected]
    print(f"selected {flat_probability} points")

    if len(flat_chain) > (2 * chain_shape[0]):
        # kmeans clustering
        km = KMeans(chain_shape[0])
        km.fit(flat_chain)

        dist = scipy.spatial.distance.cdist(flat_chain, km.cluster_centers_)

        idx_closest = numpy.argmin(dist, 0)

        closest = dist[idx_closest, range(chain_shape[0])]

        best_chain = flat_chain[idx_closest]
        best_prob = flat_probability[idx_closest]
    else:
        pop_size = chain.shape[0]
        sort_idx = numpy.argsort(flat_probability_unique)
        sort_idx = sort_idx[numpy.isfinite(sort_idx)]

        best = sort_idx[-pop_size:]

        best_chain = flat_chain_unique[best, :]
        best_prob = flat_probability_unique[best]

    return best_chain, best_prob


def addChain(*args):
    temp = [arg for arg in args if arg is not None]
    if len(temp) > 1:
        return numpy.concatenate(temp, axis=1)
    else:
        return numpy.array(temp[0])


def random_start(nwalkers, ndim, log_probability_vec):
    start_random = numpy.random.random((1.0, 1, (nwalkers, ndim)))
    start_random = numpy.clip(start_random, 0, 1)
    if start_random.shape[0]<nwalkers:
        left = nwalkers - start_random.shape[0]
        more_random_samples = numpy.random.normal(1.0, 1, (nwalkers, ndim))
        more_random_samples = numpy.clip(more_random_samples, 0, 1)
        numpy.concatenate((start_random, more_random_samples[:left, :]))
    print("no of unique random start point:",start_random.shape)
    prob_start_random = log_probability_vec


def unique_start_points(log_probability_vec, y_exp_norm,sigma,reg, batchsize,scaler, lb, ub, nwalkers, ndim):
   
    '''
    unique_start_point

    Parameters 
    ----------
    best_scaled : parameters predicted res.X in pymoo space, which is from 0 to 1.
    log_proability_vec : ln transforms the theta from pymoo space. then performs scaler transform on the ln transformed theta.
                         calculated the predicted y curve using trained surrogate model
    y_exp_norm : normalized experimental data
    sigma : gaussian error factor
    reg : regression model
    scaler : scaler for parameter transform
    lb : upper bound
    ub : lower bound
    
    Returns
    -------
    start : starting point for the walkers
    prob : probability of the starting point
    '''

    lb_ln = numpy.log(lb)
    ub_ln = numpy.log(ub)

    start =  numpy.random.normal(1.0, 0.5, (nwalkers, ndim))
    start = numpy.clip(start, 0, 1)
    if start.shape[0]<nwalkers:
        left = nwalkers - start.shape[0]
        more_samples = numpy.random.normal(1.0, 1, (nwalkers, ndim))
        more_samples = numpy.clip(more_samples, 0, 1)
        numpy.concatenate((start, more_samples[:left, :]))
    print("no of unique start points centred at best point:",start.shape)
    prob = log_probability_vec(start, y_exp_norm,sigma, reg, batchsize, scaler, lb_ln, ub_ln)   

    return start, prob 
    
    

def mcmcsampler(nwalkers, ndim, log_probability_vec, y_exp_norm,sigma,reg, batchsize,scaler, lb, ub):
    lb_ln = numpy.log(lb)
    ub_ln = numpy.log(ub)
    gamma0 = 2.38/ numpy.sqrt(2 * ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_vec, vectorize=True, 
                                    args=(y_exp_norm, sigma,reg,batchsize, scaler, lb_ln, ub_ln),
                                    moves=[(de_snooker.DESnookerMove(), 0.1),
                                        (de.DEMove(), 0.9*0.9),
                                        (de.DEMove(gamma0), 0.9*0.1),])
    return sampler


def plot_best(best_norm, y_exp_norm, reg,batchsize, sub_path):
    '''
    plot_best : plots the predicted y at the best point

    Parameters 
    ----------
    best_norm : the scaler normalized value of the best point predicted by pymoo
    y_exp_norm : normalized experimental data
    reg : regression model
    dir : directory where the plot will be saved

    Returns
    -------
    Nothing
    '''
    y_predict = reg.predict(best_norm, batch_size=batchsize)
    
    err = numpy.sum( (numpy.squeeze(y_predict) - numpy.squeeze(y_exp_norm))**2)
    
    plt.figure(figsize=[10,10])
    plt.plot(numpy.squeeze(y_predict), label="nn")
    plt.plot(numpy.squeeze(y_exp_norm), label="target")
    plt.title(f"{best_norm} err {err}")
    plt.legend()
    plt.savefig(Path(sub_path)/('best_point_predict'))
    #plt.show()
    plt.close('all')
    

def save_best(train_path, best_scaled, best_ln):
    '''
    save_best: saves the best point found by pymoo

    Parameters 
    ----------
    train_path : location of the file where this data is stored
    best_sclaed : the best point in the pymoo space
    best_ln : the best point in natural log sapce

    Returns
    -------
    Nothing

    '''

    td = h5.H5()
    td.filename = train_path
    td.load()
    td.root.best_scaled = best_scaled
    td.root.best_point_ln = best_ln
    td.root.best_point = numpy.exp(best_ln)
    td.save()

def plot_sim(y, y_predict, dir, fname):
    '''
    plot_sim : plot's y_norm and predited y vs arbitrary output

    Parameters 
    ----------
    y : numpy.array of y-axis values. Each row is new result.
    y_predict : numpy array of y-axis of predicted output
    dir : the directory where the plot is stores
    fname : the filename of the sved plot

    Returns
    -------
    Nothing
    '''
    num = y.shape[0]

    fig,ax = plt.subplots(num,1, figsize=[10,15])
    for i in range(num):
        ax[i,].plot(y[i,:])
        ax[i,].plot(y_predict[i,:],'--')
    plt.xlabel('x (a.u.)')
    plt.ylabel('y (a.u.)')
    #plt.show()
    fig.savefig(dir/fname)
    plt.close('all')
    

def check_nn(reg, train_path, dir, fname):

    td = h5.H5()
    td.filename = train_path
    td.load()
    x_test = td.root.X_test
    y_test = td.root.Y_test

    idx = numpy.random.randint(0, x_test.shape[0],5)
    y_predict = reg(x_test[idx,:])
    plot_sim(y_test[idx,:], y_predict, dir, fname)




def load_training_data(train_path):
    '''
    load_training_data : loads training data from hdf5

    Parameters 
    ----------
    train_path : file path to training data file
    exp : identifies which experiment we are using

    returns
    -------
    ub
    lb
    n_var
    y1_max
    y2_min
    y2_max
    y2_min

    '''
    td = h5.H5()
    td.filename = train_path
    td.load()
    ub = td.root.ub
    lb = td.root.lb
    n_var = td.root.n_var
    y_max = td.root.y_max
    y_min = td.root.y_min
    return ub, lb, n_var, y_max, y_min
    

def load_training_data_2(train_path):
    '''
    load_training_data : loads training data from hdf5

    Parameters 
    ----------
    train_path : file path to training data file
    exp : identifies which experiment we are using

    returns
    -------
    X : x-axis of the training data
    Y : y-axis of the training data

    '''
    td = h5.H5()
    td.filename = train_path
    td.load()
    x_train = td.root.X_train
    y_train = td.root.Y_train
    x_test = td.root.X_test
    y_test = td.root.Y_test 
    X = numpy.concatenate((x_train, x_test), axis=0)
    Y = numpy.concatenate((y_train, y_test), axis=0)
    print(X.shape, Y.shape)
    return X, Y

    
def load_exp_data(exp_path):
    '''
    load_exp_data : loads interpolated experimental data from hdf5

    Parameters 
    ----------
    exp_path : file path to experimental file
    exp : identifies which experiment we are using

    returns
    -------
    y_exp_1 : y-axis of 1 or multiple illumination intensities of experiment 1
    y_exp_2 : y-axis of 1 or multiple illumination intensities of experiment 2

    '''

    exp_data = h5.H5()
    exp_data.filename = exp_path.as_posix()
    exp_data.load()
    y_exp = exp_data.root.y_exp_1
    x_exp = exp_data.root.x_exp_1
    print(y_exp.shape)
    return y_exp, x_exp


def y_exp_transform(y_exp, max, min):
    '''
    y_exp_transform : norm of the experimental data using the min max from the simulated dataset

    Parameters 
    ----------
    y : experimental data
    max : max of the simulated data
    min : min of the simulated data

    returns
    -------
    y_exp_norm : norm of the experimetnal data

    '''
    y_exp_norm = (y_exp-min)/(max-min)

    return y_exp_norm


def log_norm_pdf_vec(y,mu,sigma):
    
    '''
    log_norm_pdf_vec : calculates the gaussian error between the predicted jv and the experimental data

    Parameters 
    ----------
    y : jv predicted from theta
    mu : normalized experimental data
    sigma : gaussian error parameter

    returns
    -------
    error : error between the jv predicted from theta and experimental data

    '''
    n = len(y)
    #print("y shape", y.shape)
    #print("mu shape", mu.shape)
    error_jsc = y[:,:40] - mu[:40]
    error_voc = y[:, 384:439] - mu[384:439]
    error_mp = y[:,169:340] - mu[169:340]
    error_end = y[:, 500:] - mu[500:]
    error_tot = numpy.concatenate((error_jsc, error_mp, error_voc, error_end), axis=1)
    #print("error shape", error_tot.shape)
    #error = -0.5 * n * numpy.log(2 * numpy.pi * sigma**2) - numpy.sum((y - mu)**2) / (2 * sigma**2)
    #error = -0.5*numpy.sum((((y-mu)**2/(sigma**2)) + numpy.log(2*3.14*(sigma**2))), axis = 1)
    error = -0.5*numpy.sum((((error_tot)**2/(sigma**2)) + numpy.log(2*3.14*(sigma**2))), axis = 1)
    return error


def error_func(theta,y_exp_norm, reg, batchsize, scaler, lb, ub):
    '''
    error : calculates the  error between the predicted jv and the experimental data

    Parameters 
    ----------
    y : jv predicted from theta
    mu : normalized experimental data
    sigma : gaussian error parameter

    returns
    error : error between the jv predicted from theta and experimental data

    '''
    theta_trans = numpy.array(theta) * (ub-lb) + lb    
    theta_norm = theta_transform(theta_trans, scaler)
    jv_predicted_from_theta = reg.predict(theta_norm, batch_size=batchsize)
    #print("jv_predicted_from_theta shape", jv_predicted_from_theta.shape)
    #print("y_exp_norm shape", y_exp_norm.shape)
    error_jsc = jv_predicted_from_theta[:,:40] - y_exp_norm[:40]
    error_voc = jv_predicted_from_theta[:, 384:439] - y_exp_norm[ 384:439]
    error_mp = jv_predicted_from_theta[:,169:340] - y_exp_norm[169:340]
    error_end = jv_predicted_from_theta[:, 500:] - y_exp_norm[500:]
    error_tot = numpy.abs(numpy.concatenate((error_jsc, error_mp, error_voc, error_end), axis=1))
    error_elimentwise = error_tot
    error_total = numpy.sum(error_elimentwise, axis=1)
    #error_total = numpy.sum(jv_predicted_from_theta - y_exp_norm, axis=1)
    #print(error_total)
    return error_total

def log_probability_vec(theta, y_exp_norm,sigma, reg, batchsize, scaler, lb, ub):
    '''
    log_probability_vec : ln transforms the theta from pymoo space. then performs scaler transform on the ln transformed theta.
                          calculated the predicted jv curve using trained surrogate model
   
    Parameters 
    ----------
    theta : numpy array of the parameters predicted by pymoo. theta is pymoo space (0 to 1)
    y_exp_norm : minmax transformed experimental data
    sigma : gaussian error parameter
    reg : regression model
    scaler : scaler value for normalization
    lb : lower bound
    ub : upper bound

    returns
    -------
    errors : error between the jv predicted from theta and experimental data

    '''

    theta_trans = numpy.array(theta) * (ub-lb) + lb    
    theta_norm = theta_transform(theta_trans, scaler)
    jv_predicted_from_theta = reg.predict(theta_norm, batch_size=batchsize)
    errors = log_norm_pdf_vec(jv_predicted_from_theta, y_exp_norm, sigma)
    return errors



def theta_transform(theta, scaler):
    
    '''
    theta_transform : does transform of theta predicted by pymoo

    Parameters 
    ----------
    theta : numpy array of the parameters predicted by pymoo. theta is pymoo space (0 to 1)
    scaler : scaler value for normalization
    
    returns
    -------
    theta_norm : scaler transformed theta  
    
    '''

    theta_norm = scaler.transform(theta)
    return theta_norm

def theta_inverse_transform(theta_norm, scaler, lb, ub):
    
    '''
    theta_inverse_transform : does inverse transform of theta from NN space to ln space

    Parameters 
    ----------
    theta : numpy array of the parameters represented in the NN space (scaler space). 
    scaler : scaler value for normalization
    
    returns
    -------
    theta_norm : scaler inverse transformed theta  
    
    '''
    lb_ln = numpy.log(lb)
    ub_ln = numpy.log(ub)

    theta_ln = scaler.inverse_transform(theta_norm)  
    theta_scaled = (theta_ln - lb_ln)/(ub_ln-lb_ln)
    return theta_scaled, theta_ln, 


def plot_corner(sub_path, chain, ub_ln, lb_ln):
    flat_chain = flatten(chain)
    flat_chain_trans = (flat_chain * (ub_ln-lb_ln) + lb_ln)
    flat_chain_actual = numpy.exp(flat_chain_trans)
    S_etl = 5e-7 /flat_chain_actual[:,4] #in cm/s
    S_htl = 5e-7 /flat_chain_actual[:,-1] #in cm/s
    chain_modified = flat_chain_actual
    chain_modified[:,0] = numpy.log10(chain_modified[:,0] * 1e4) # abs mobility in cm2/Vs in natural log
    chain_modified[:,3] = numpy.log10(chain_modified[:,3] * 1e4) # ETL mobility in cm2/Vs in natural log
    chain_modified[:,6] = numpy.log10(chain_modified[:,6] * 1e4) # HTL mobility in cm2/Vs in natural log
    chain_modified[:,1] = numpy.log10(chain_modified[:,1] * 1e4) # abs lifetime in natural log
    chain_modified[:,2] = (3.83 - chain_modified[:,2]) * 1e3 # CB offset in meV; EAabs - EAetl
    chain_modified[:,5] = (3.83 + 1.6 - chain_modified[:,5] - 3) * 1e3 # VB offset in meV; EAabs + Egabs - EAhtl - Eghtl
    chain_modified[:,4] = S_etl
    chain_modified[:,-1] = S_htl
    #chain_modified_trans = numpy.log(chain_modified)
    chain_modified_1 = chain_modified[:, (0,1,3,6)] #abs mobility, abs lifetime, ETL mobility, HTL, mobility
    chain_modified_2 = chain_modified[:, (2,4,5,7)] #CB offset, S_ETL, VB offset, S_HTL
    chain_modified_3 = chain_modified[:, 2:]


    label = [r'$\mu_\mathrm{abs} \mathrm{[cm^2/Vs]}$', r'$\tau_\mathrm{abs} \mathrm{[s]}$', r'$\Delta E_\mathrm{C} \mathrm{[meV]}$', r'$log(\mu_{etl} \mathrm{[cm^2/Vs]})$', r'$S_\mathrm{etl} \mathrm{[cm/s]}$', r'$\Delta E_\mathrm{V} \mathrm{[meV]}$', r'$log(\mu_\mathrm{htl}  \mathrm{[cm^2/Vs]})$', r'$S_\mathrm{htl} \mathrm{[cm/s]}$']


    fig = corner.corner(
        chain_modified,
        labels = label,
        label_kwargs={"fontsize": 25},
        quantiles=(0.05, 0.5, 0.95),
        show_titles=True,
        title_kwargs={"fontsize": 16},
        bins=30,
        plot_contours = True,
        range=new_range(chain_modified).T,
        color = "#00316E",
        use_math_text=True,
        max_n_ticks = 4,
        title_fmt=".2f"
        )
    fig.subplots_adjust(right=1.6,top=1.6)
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=20)
    plt.savefig(sub_path/("corner.pdf"),dpi=300,pad_inches=0.2,bbox_inches='tight')
#plt.show()
plt.close()


def predict_jv(theta, reg, batchsize, scaler, ub, lb):
    theta_trans = numpy.array(theta) * (ub-lb) + lb    
    theta_norm = theta_transform(theta_trans, scaler)
    jv_predicted_from_theta = reg.predict(theta_norm, batch_size=batchsize)
    return jv_predicted_from_theta, theta_trans

def plot_jv(y_exp, y_exp_bayes, x_exp, filename,eff_p, eff_e, Voc_p, Voc_e, Jsc_p, Jsc_e, V_mpp_p, V_mpp_e, J_mpp_p, J_mpp_e, Power_p, Power_e, Pmax_p, Pmax_e, FF_p, FF_e, cur_path):
    plt.figure(figsize=(10, 8))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.plot(x_exp, y_exp, color = 'green',alpha=0.45,label='experimental data')
    plt.plot(x_exp, y_exp_bayes, color = 'blue',linestyle='--',alpha=0.45,label='modelled data')
    #plt.plot(x_exp, Power_p, color = 'red', linestyle='--', alpha=0.45, label='Power (modelled)')
    #plt.plot(x_exp, Power_e, color = 'orange', linestyle='--', alpha=0.45, label='Power (experimental)')
    plt.scatter(V_mpp_p, J_mpp_p, color='blue', s=100, zorder=5, label='MPP (modelled)')
    plt.scatter(V_mpp_e, J_mpp_e, color='green', s=100, zorder=5, label='MPP (experimental)')
    # Add a small table with Voc, Jsc, FF, and Pmax values for both modelled and experimental data
    # Ensure all values are scalars (not arrays)
    Voc_p_val = float(numpy.squeeze(Voc_p))
    Voc_e_val = float(numpy.squeeze(Voc_e))
    Jsc_p_val = float(numpy.squeeze(Jsc_p))
    Jsc_e_val = float(numpy.squeeze(Jsc_e))
    FF_p_val = float(numpy.squeeze(FF_p))
    FF_e_val = float(numpy.squeeze(FF_e))
    Pmax_p_val = float(numpy.squeeze(Pmax_p))
    Pmax_e_val = float(numpy.squeeze(Pmax_e))
    eff_p_val = float(numpy.squeeze(eff_p))
    eff_e_val = float(numpy.squeeze(eff_e))

    cell_text = [
        [f"{Voc_p_val:.2f}", f"{Voc_e_val:.2f}"],
        [f"{Jsc_p_val:.2f}", f"{Jsc_e_val:.2f}"],
        [f"{FF_p_val:.2f}", f"{FF_e_val:.2f}"],
        [f"{Pmax_p_val:.2f}", f"{Pmax_e_val:.2f}"],
        [f"{eff_p_val:.2f}", f"{eff_e_val:.2f}"]
    ]
    rows = ['Voc (V)', 'Jsc (mA/cm²)', 'FF', 'Pmax (mW/cm²)', 'Efficiency (%)']
    cols = ['Modelled', 'Experimental']
    # Position the table at centre left
    table = plt.table(cellText=cell_text, rowLabels=rows, colLabels=cols, loc='center left', cellLoc='center', colLoc='center', bbox=[0.05, 0.35, 0.33, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    #plt.text(0.1, -8, '1.0 sun', fontsize=15, color='black')
    #plt.text(0.1, -10, 'Voc = %.2f V' % Voc, fontsize=15, color='black')
    #plt.text(0.1, -12, 'Jsc = %.2f mA/cm2' % Jsc, fontsize=15, color='black')
    #plt.text(0.1, -14, 'Pmax = %.2f mW/cm2' % Pmax, fontsize=15, color='black')
    #plt.text(0.1, -16, 'FF = %.2f' % FF, fontsize=15, color='black')
    #plt.text(0.1, -18, 'eff = %.2f %%' % (eff*100), fontsize=15, color='black')
    plt.legend()
    plt.legend(fontsize=15)
    plt.ylim(-28,0)
    #plt.xlim(0, 1.1)
    plt.xlabel('voltage $V$ (V)', fontsize=20)
    plt.ylabel('current density $J \ \mathrm{(mA/cm^2)}$', fontsize=20)
    plt.title('Illumination depednent $JV$ Data', fontsize=20)
    plt.title(filename, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(Path(cur_path)/'jv_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
 


def parameter_post_processing(best_params_ln, cur_path):
    best_params_ln = best_params_ln.reshape(-1,1)
    best_params_actual = numpy.exp(best_params_ln.reshape(-1,1))
    S_etl = 5e-7 /best_params_actual[4] #in cm/s
    S_htl = 5e-7 /best_params_actual[-1] #in cm/s
    modified_params = best_params_actual.copy()
    modified_params[0] = modified_params[0] * 1e4 # abs mobility in cm2/Vs in natural log
    modified_params[3] = modified_params[3] * 1e4 # ETL mobility in cm2/Vs in natural log
    modified_params[6] = modified_params[6] * 1e4 # HTL mobility in cm2/Vs in natural log
    modified_params[1] = modified_params[1] # abs lifetime in natural log
    modified_params[2] = numpy.round((3.83 - modified_params[2]), 3)  # CB offset in meV; EAabs - EAetl
    modified_params[5] = numpy.round((3.83 + 1.6 - modified_params[5] - 3),3)  # VB offset in meV; EAabs + Egabs - EAhtl - Eghtl
    modified_params[4] = S_etl # ETL surface recombination velocity in cm/s
    modified_params[-1] = S_htl

    # Save modified parameters to CSV
    csv_path = Path(cur_path)/("results_parameters.csv")
    param_names = [
        "abs_mobility_cm2Vs", "abs_lifetime_s", "CB_offset_meV", "ETL_mobility_cm2Vs",
        "S_etl_cm_s", "VB_offset_meV", "HTL_mobility_cm2Vs", "S_htl_cm_s"
    ]
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(param_names)
        writer.writerow(modified_params.flatten())

    return modified_params

def write_csv(cur_path, x_exp, y_exp, y_exp_bayes,
              eff_p, eff_e, Voc_p, Voc_e, Jsc_p, Jsc_e,
              V_mpp_p, V_mpp_e, J_mpp_p, J_mpp_e,
              Power_p, Power_e, Pmax_p, Pmax_e, FF_p, FF_e, ):
    
    csv_path = Path(cur_path) / "results_jv.csv"
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = [
            'x_exp', 'y_exp_p','y_exp_e','Power_p', 'Power_e',
            'eff_p', 'eff_e', 'Voc_p', 'Voc_e', 'Jsc_p', 'Jsc_e',
            'V_mpp_p', 'V_mpp_e', 'J_mpp_p', 'J_mpp_e',
             'Pmax_p', 'Pmax_e', 'FF_p', 'FF_e'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Write all JV points
        n_points = len(x_exp)
        for i in range(n_points):
            row = {
                'x_exp': x_exp[i],
                'y_exp_p': y_exp_bayes[i] if hasattr(y_exp_bayes, '__len__') else y_exp_bayes,
                'y_exp_e': y_exp[i] if hasattr(y_exp, '__len__') else y_exp,
                'Power_p': Power_p[i] if hasattr(Power_p, '__len__') and len(Power_p) == n_points else (Power_p if i == 0 else ''),
                'Power_e': Power_e[i] if hasattr(Power_e, '__len__') and len(Power_e) == n_points else (Power_e if i == 0 else ''),
                'eff_p': eff_p if i == 0 else '',
                'eff_e': eff_e if i == 0 else '',
                'Voc_p': Voc_p if i == 0 else '',
                'Voc_e': Voc_e if i == 0 else '',
                'Jsc_p': Jsc_p if i == 0 else '',
                'Jsc_e': Jsc_e if i == 0 else '',
                'V_mpp_p': V_mpp_p if i == 0 else '',
                'V_mpp_e': V_mpp_e if i == 0 else '',
                'J_mpp_p': J_mpp_p if i == 0 else '',
                'J_mpp_e': J_mpp_e if i == 0 else '',
                'Pmax_p': Pmax_p if i == 0 else '',
                'Pmax_e': Pmax_e if i == 0 else '',
                'FF_p': FF_p if i == 0 else '',
                'FF_e': FF_e if i == 0 else ''
            }
            writer.writerow(row)

def efficiency_func(x_exp, y_exp):
    Voc = x_exp[numpy.argmin(numpy.abs(y_exp))]
    Jsc = y_exp[numpy.argmin(numpy.abs(x_exp))]
    # Extract the part of the curve between 0 and Voc of x value
    mask = (x_exp >= 0) & (x_exp <= Voc)
    x_exp = x_exp[mask]
    y_exp = y_exp[mask]
    y_exp = y_exp.reshape(-1)
    Power = (-y_exp * x_exp)
    max_power_idx = numpy.argmax(Power)
    V_mpp = x_exp[max_power_idx]
    J_mpp = y_exp[max_power_idx]
    Pmax = numpy.max(Power)
    FF = Pmax / (Voc * Jsc)
    eff = Voc * Jsc * FF  # Convert to percentage
    return eff, Voc, Jsc, V_mpp, J_mpp, Power, Pmax, FF

def best_params_post_processing(mother_chain, mother_prob, reg, batchsize, scaler, ub_ln, lb_ln, y_exp, y_max, y_min, x_exp, filename, cur_path):  
    best_prob_mc = numpy.max(mother_prob)
    max_index = numpy.unravel_index(numpy.argmax(mother_prob), mother_prob.shape)
    best_params = mother_chain[max_index[0], max_index[1], :]
    best_params = best_params.reshape(1,-1)

    jv_predicted, best_params_ln = predict_jv(best_params, reg, batchsize,scaler, ub_ln, lb_ln)
    best_parameters_actual = parameter_post_processing(best_params_ln, cur_path)
    jv_predicted= jv_predicted.reshape(-1,1)
    y_exp_bayes = jv_predicted*(y_max-y_min) + y_min
    eff_p, Voc_p, Jsc_p, V_mpp_p, J_mpp_p, Power_p, Pmax_p, FF_p = efficiency_func(x_exp, y_exp_bayes)
    eff_e, Voc_e, Jsc_e, V_mpp_e, J_mpp_e, Power_e, Pmax_e, FF_e = efficiency_func(x_exp, y_exp)
    plot_jv(y_exp, y_exp_bayes, x_exp, filename,eff_p, eff_e, Voc_p, Voc_e, Jsc_p, Jsc_e, V_mpp_p, V_mpp_e,
             J_mpp_p, J_mpp_e, Power_p, Power_e, Pmax_p, Pmax_e, FF_p, FF_e, cur_path)
    write_csv(cur_path, x_exp, y_exp, y_exp_bayes,eff_p, eff_e, Voc_p, Voc_e, Jsc_p, Jsc_e,V_mpp_p, V_mpp_e,
               J_mpp_p, J_mpp_e, Power_p, Power_e, Pmax_p, Pmax_e, FF_p, FF_e)
    return eff_p, eff_p, Voc_p, Voc_e, Jsc_p, Jsc_e, V_mpp_p, V_mpp_e, J_mpp_p, J_mpp_e, Power_p, Power_e, Pmax_p, Pmax_e, FF_p,FF_e, best_prob_mc, max_index, best_params, jv_predicted, best_params_ln, best_parameters_actual, y_exp_bayes



def main(dir_path, name, reg_path, train_path, scaler_path,exp_path, sigma, nwalkers):
    batchsize = 8192
    start_time = time.time()
    identity = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sub_dir = identity + name 
    print(sub_dir)
    sub_path = dir_path/("%s" % sub_dir)
    os.mkdir(sub_path)
    
    results = h5.H5()
    results_name = sub_path/("%s.h5" % sub_dir)
    results.filename = results_name.as_posix() #the file where all your training related data is getting stored
    results.save()
    print(results.filename)
    
    # load trained regression model
    reg = tf.keras.models.load_model(reg_path, custom_objects={'log_mse':log_mse})

    # load upper bound, lower bound and min max of different experiments
    ub, lb, n_var, y_max, y_min = load_training_data(train_path)

    ndim = len(ub)
    # load scaler
    scaler = joblib.load(scaler_path)

    # check NN training
    fname = 'nn_check_plot'
    check_nn(reg, train_path, sub_path, fname)


    # load experimental data 
    '''
    Before this step is done make sure you experimental data is already interpolated to the correct size.
    '''
    y_exp, x_exp = load_exp_data(exp_path)
    plt.plot(y_exp)
    plt.close()
    results.root.y_exp = y_exp
    results.root.x_exp = x_exp
    results.save()

    # transform the experimental data 
    y_exp_norm = y_exp_transform(y_exp, y_max, y_min)
    plt.plot(y_exp_norm)
    results.root.y_exp_norm = y_exp_norm
    print("shape of Y_exp_norm:", y_exp_norm.shape)
    results.save()

    # define the mcmc sampler 
    sampler = mcmcsampler(nwalkers, ndim, log_probability_vec, y_exp_norm, sigma, reg, batchsize, scaler, lb, ub)
      
    # find random initialization points for the walkers
    start, prob = unique_start_points(log_probability_vec, y_exp_norm, sigma, reg, batchsize, scaler, lb, ub, nwalkers, ndim)
    results.root.start_initial = start
    results.root.prob_initial = prob
    results.save()

    error = numpy.full(512, fill_value=1, dtype=float) * 0.00041
    sigma = 1e-4
    min_ln_pdf = -0.5*numpy.sum((((error)**2/(sigma**2)) + numpy.log(2*3.14*(sigma**2))))
    min_prob = min_ln_pdf
    print("min_prob", min_prob)

    auto_state, auto_chain, auto_probability = auto_high_probability(sampler, start)
    results.root.walkers_start = auto_state.coords
    results.root.walker_start_prob = auto_state.log_prob
    results.root.optimized.walkers_start_chain = auto_chain
    results.root.optimized.walkers_start_prob = auto_probability
    results.save()

    plot_start(auto_state, y_exp_norm, reg, batchsize, scaler,lb, ub, sub_path, results)

    
    # run final bayes step
    motherchainT, motherprobT = run_walkers_2(auto_state,sampler, results, sub_path)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time required to run the code: {total_time} seconds")
    results.root.total_time = total_time
    results.save()
    
    ub_ln = numpy.log(ub)
    lb_ln = numpy.log(lb)
    
    plot_corner(sub_path, motherchainT, ub_ln,lb_ln)
    
    eff_p, eff_p, Voc_p, Voc_e, Jsc_p, Jsc_e, V_mpp_p, V_mpp_e, J_mpp_p, J_mpp_e, Power_p, Power_e, Pmax_p, Pmax_e, FF_p,FF_e, best_prob_mc, max_index, best_params, jv_predicted, best_params_ln, best_parameters_actual, y_exp_bayes = best_params_post_processing(motherchainT, 
                                                                                motherprobT, reg, batchsize, scaler, ub_ln, lb_ln, y_exp, y_max, y_min, x_exp, name,sub_path)
    perfomance_metrics = numpy.hstack((eff_p, eff_p, Voc_p, Voc_e, Jsc_p, Jsc_e, V_mpp_p, V_mpp_e, J_mpp_p, J_mpp_e, Power_p, Power_e, Pmax_p, Pmax_e, FF_p,FF_e,))
    results.root.from_MC.performance_metrics = perfomance_metrics
    results.root.from_MC.best_prob_mc = best_prob_mc
    results.root.from_MC.max_index = numpy.array(max_index) 
    results.root.from_MC.best_params = best_params
    results.root.from_MC.jv_predicted = jv_predicted
    results.root.from_MC.best_params_ln = best_params_ln
    results.root.from_MC.best_parameters_actual = best_parameters_actual
    results.root.from_MC.y_exp_bayes = y_exp_bayes
    results.save()

    
    
  
    

if __name__ == "__main__":
    main(dir_path, name, reg_path, train_path, scaler_path,exp_path, exp)







    


