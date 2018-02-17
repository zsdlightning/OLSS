#!/bin/python


import os
import sys
import numpy as np
import scipy as sp
from scipy.stats import norm as normal
from scipy.special import *
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
import scipy.linalg as linalg
from sklearn import metrics
import random
import time

'''
This version deals with sparse features, VW format
'''


feature_off = 3


def vec(a):
    return a.reshape([a.size,1])


def normalize(val, mean_std):
    return (val - mean_std[0])/mean_std[1]

class EPSS:
    #d: dimension, rho: selection prior
    def __init__(self, d, rho0 = 0.5, n_epoch = 1, mini_batch = 1, tol = 1e-5, damping = 0.9, tau0 = 1.0):
        #Bernoli prior for selection variables
        self.rho = logit(rho0)
        self.tol = tol
        self.damping = damping
        self.tau0 = tau0
        self.INF = 1e6
        self.mini_batch = mini_batch
        self.n_epoch = n_epoch



    # normal_PDF / normal_CDF
    def pdf_over_cdf(self, input):
        #return normal.pdf(input)/normal.cdf(input)
        return np.exp( normal.logpdf(input) - normal.logcdf(input) )


    #batch training
    def train(self, X, y, intercept = False):
        if intercept:
            X = np.hstack([X, np.ones(X.shape[0]).reshape([X.shape[0],1])])
        self.init(X,y)
        n,d = X.shape
        X2 = X*X
        Y = np.tile(vec(y), [1,d])
        for it in xrange(self.max_iter):
            old_v = self.v.copy()
            old_mu = self.mu.copy()
            old_r = self.r.copy()
            #likelihood terms (future version should have go through one by one rather than parallel)
            v = np.tile(self.v, [n,1])
            mu = np.tile(self.mu, [n, 1])
            v_inv_not  = 1.0/v - 1.0/self.v_l
            v_not = 1.0/v_inv_not
            mu_not = v_not * (mu/v - self.mu_l/self.v_l)
            t1 = y * np.sum(mu_not * X, 1)
            t2 = np.sqrt(np.sum(v_not*X2 + 1.0, 1))
            t3 = self.pdf_over_cdf(t1/t2)/t2
            dmu_not = np.tile(vec(t3*y), [1, d]) * X
            dv_not = np.tile(vec(-0.5*t3*t1/(t2*t2)), [1,d]) * X2
            mu = mu_not + v_not * dmu_not
            v = v_not - v_not**2*(dmu_not**2 - 2*dv_not)
            #updated likelihood terms
            v_l_inv = 1/v - 1/v_not
            v_l_inv[ v_l_inv <= 0] = 1/self.INF
            v_l = 1.0/v_l_inv
            #damping
            v_l_inv = self.damping * 1.0/v_l + (1 - self.damping) * 1.0/self.v_l
            v_l_inv_mu = self.damping * ( mu/v - mu_not/v_not ) + (1 - self.damping) * self.mu_l/self.v_l
            self.v_l = 1/v_l_inv
            self.mu_l = self.v_l * v_l_inv_mu
            #update global terms
            v_inv_all =  np.sum(1/self.v_l, 0) + 1/self.v_p
            v_inv_mu = np.sum(self.mu_l/self.v_l, 0) + self.mu_p/self.v_p
            self.v = 1/v_inv_all
            self.mu = self.v * v_inv_mu
            #update prior terms
            v_inv_not = 1/self.v - 1/self.v_p
            v_not = 1/v_inv_not
            mu_not = v_not * (self.mu/self.v - self.mu_p/self.v_p)
            v_tilt = 1/(1/v_not + 1/self.tau0)
            mu_tilt = v_tilt * (mu_not/v_not)
            #log N(0 | mu_not, v_not + tau0)
            log_h = normal.logpdf(mu_not, scale = np.sqrt(v_not + self.tau0))
            #log N(0 | mu_not, v_not)
            log_g = normal.logpdf(mu_not, scale = np.sqrt(v_not))
            rho_p =  log_h - log_g
            sel_prob = expit(self.rho + rho_p)
            mu = sel_prob * mu_tilt
            v = sel_prob * (v_tilt + (1.0 - sel_prob)*mu_tilt**2)
            #damping
            self.rho_p = self.damping * rho_p + (1 - self.damping) * self.rho_p
            v_p_inv = 1/v - v_inv_not
            v_p_inv[ v_p_inv <= 0] = 1/self.INF
            v_p_inv_mu = mu/v - mu_not/v_not
            v_p_inv = self.damping * v_p_inv + (1 - self.damping) * 1/self.v_p
            v_p_inv_mu = self.damping * v_p_inv_mu + (1 - self.damping) * self.mu_p/self.v_p
            self.v_p = 1/v_p_inv
            self.mu_p = self.v_p * v_p_inv_mu
            #update global approx. dist.
            self.r = self.rho_p + self.rho
            v_inv_all =  np.sum(1/self.v_l, 0) + 1/self.v_p
            v_inv_mu = np.sum(self.mu_l/self.v_l, 0) + self.mu_p/self.v_p
            self.v = 1/v_inv_all
            self.mu = self.v * v_inv_mu
            #difference only on global approxiations
            diff = np.sqrt(np.sum((1/old_v - v_inv_all)**2) + np.sum((old_mu - self.mu)**2) + np.sum((old_r - self.r)**2))/(old_v.size + old_mu.size + old_r.size)
            print 'iter %d, diff = %g'%(it, diff)
            if diff < self.tol:
                break

    #note, n is an array
    def init_sep(self, n, d, damping_strategy = None , non_informative = True):
        if non_informative:
            #prior factors
            self.rho_p = np.zeros(d)
            self.mu_p = np.zeros(d)
            self.v_p = self.INF*np.ones(d)
            #average likelihood factors -- only for w
            self.mu_l = np.zeros(d)
            self.v_l = self.INF * np.ones(d)
            #global posterior parameters
            self.r = self.rho_p + self.rho
            self.mu = np.zeros(d)
            self.v = 1/(1.0/self.v_p + 1.0/self.v_l * n)


    #calculate the appearche of each features in the training data, used for the step-size of each approx. factor
    def calc_feature_appearence(self, d, fea2id, training_file):
        #including intercept
        res = np.zeros(d+1)
        with open(training_file, 'r') as f:
            ct = 0
            for line in f:
                ct = ct + 1
                items = line.strip().split(' ')
                res[ d ] = res[ d ] + 1
                for item in items[feature_off:]:
                    name = item.split(':')[0]
                    id = fea2id[name]
                    res[ id ]  = res[ id ] + 1
                if ct%10000 == 0:
                    print ct
        np.save('feature_appearence.npy',res)
        return res


    #this version is the same as train_stochastic_multi_rate, except that at the beining, I will update all the prior factors
    def train_stochastic_multi_rate(self, d, n_pos, n_neg, training_file, fea2id, fea2stats, Xtest, ytest, logger, n_batch_update_prior = 1, intercept = False, damping_both = True):
        #initialization
        #separate average likelihood for pos. & neg. samples
        d = d + 1
        self.INF = 1e6
        self.rho_p = np.zeros(d)
        self.mu_p = np.zeros(d)
        self.v_p = self.INF*np.ones(d)
        self.mu_l_pos = np.zeros(d)
        self.v_l_pos = self.INF * np.ones(d)
        self.mu_l_neg = np.zeros(d)
        self.v_l_neg = self.INF * np.ones(d)
        #global posterior parameters
        self.r = self.rho_p + self.rho
        self.mu = np.zeros(d)
        self.v = 1.0/(1.0/self.v_p + n_pos*1.0/self.v_l_pos + n_neg*1.0/self.v_l_neg)


        old_v = self.v.copy()
        old_mu = self.mu.copy()
        old_r = self.r.copy()
        it = 0
        curr = 0
        count = 0
        n_batch_pos = np.zeros(d)
        n_batch_neg = np.zeros(d)
        v_l_inv_batch_pos = np.zeros(d)
        v_l_inv_batch_neg = np.zeros(d)
        v_l_inv_mu_batch_pos = np.zeros(d)
        v_l_inv_mu_batch_neg = np.zeros(d)

        #first, update prior factors
        v_inv_not = 1/self.v - 1/self.v_p
        v_not = 1/v_inv_not
        mu_not = v_not * (self.mu/self.v - self.mu_p/self.v_p)
        v_tilt = 1/(1/v_not + 1/self.tau0)
        mu_tilt = v_tilt * (mu_not/v_not)
        #log N(0 | mu_not, v_not + tau0)
        log_h = normal.logpdf(mu_not, scale = np.sqrt(v_not + self.tau0))
        #log N(0 | mu_not, v_not)
        log_g = normal.logpdf(mu_not, scale = np.sqrt(v_not))
        rho_p =  log_h - log_g
        sel_prob = expit(self.rho + rho_p)
        mu = sel_prob * mu_tilt
        v = sel_prob * (v_tilt + (1.0 - sel_prob)*mu_tilt**2)
        #damping
        self.rho_p = self.damping * rho_p + (1 - self.damping) * self.rho_p
        v_p_inv = 1/v - v_inv_not
        v_p_inv[ v_p_inv <= 0] = 1/self.INF
        v_p_inv_mu = mu/v - mu_not/v_not
        v_p_inv = self.damping * v_p_inv + (1 - self.damping) * 1/self.v_p
        v_p_inv_mu = self.damping * v_p_inv_mu + (1 - self.damping) * self.mu_p/self.v_p
        self.v_p = 1/v_p_inv
        self.mu_p = self.v_p * v_p_inv_mu
        #update global approx. dist.
        self.r = self.rho_p + self.rho
        v_inv_all =  v_inv_not  + 1.0/self.v_p
        v_inv_mu = mu_not/v_not + self.mu_p/self.v_p
        self.v = 1.0/v_inv_all
        self.mu = self.v * v_inv_mu
        #for updating prior factors
        accumulate_ind = []


        start_time = time.clock()
        while it < self.n_epoch:
            with open(training_file, 'r') as f:
                for line in f:
                    count = count + 1
                    #extract feature values
                    items = line.strip().split(' ')
                    id = []
                    val = []
                    for item in items[feature_off:]:
                        key_val = item.split(':')
                        id.append(fea2id[key_val[0]])
                        if len(key_val) == 1:
                            val.append(1.0)
                        else:
                            val.append( float(key_val[1]) )
                            #val.append( normalize(float(key_val[1]), fea2stats[ key_val[0] ]) )
                    #intercept
                    id.append(d-1)
                    val.append(1.0)
                    #moment matching
                    xbatch = np.array(val)
                    xbatch2 = xbatch**2
                    ybatch = int(items[0])
                    if ybatch == 1:
                        #cavity dist. q^{-1}, the same for each batch-sample
                        v_inv_not  = 1.0/self.v[id] - 1.0/self.v_l_pos[id]
                        v_not = 1.0/v_inv_not
                        mu_not = v_not * (self.mu[id]/self.v[id] - self.mu_l_pos[id]/self.v_l_pos[id])
                        t1 = ybatch * np.sum(mu_not * xbatch)
                        t2 = np.sqrt(np.sum(v_not*xbatch2 + 1.0))
                        t3 = self.pdf_over_cdf(t1/t2)/t2
                        dmu_not = (t3*ybatch) * xbatch
                        dv_not = (-0.5*t3*t1/(t2*t2)) * xbatch2
                        mu = mu_not + v_not * dmu_not
                        v = v_not - v_not**2*(dmu_not**2 - 2*dv_not)
                        #obtain new batch likelihood approx.
                        v_l_inv = 1/v - 1/v_not
                        v_l_inv[ v_l_inv <= 0] = 1/self.INF
                        v_l_inv_mu = mu/v - mu_not/v_not
                        if damping_both:
                            v_l_inv = self.damping * v_l_inv + (1.0 - self.damping) * 1.0/self.v_l_pos[id]
                            v_l_inv_mu = self.damping * v_l_inv_mu + (1.0 - self.damping) * self.mu_l_pos[id]/self.v_l_pos[id]
                        n_batch_pos[id] += 1.0
                        v_l_inv_batch_pos[id] += v_l_inv
                        v_l_inv_mu_batch_pos[id] += v_l_inv_mu
                    else:
                        #cavity dist. q^{-1}, the same for each batch-sample
                        v_inv_not  = 1.0/self.v[id] - 1.0/self.v_l_neg[id]
                        v_not = 1.0/v_inv_not
                        mu_not = v_not * (self.mu[id]/self.v[id] - self.mu_l_neg[id]/self.v_l_neg[id])
                        t1 = ybatch * np.sum(mu_not * xbatch)
                        t2 = np.sqrt(np.sum(v_not*xbatch2 + 1.0))
                        t3 = self.pdf_over_cdf(t1/t2)/t2
                        dmu_not = (t3*ybatch) * xbatch
                        dv_not = (-0.5*t3*t1/(t2*t2)) * xbatch2
                        mu = mu_not + v_not * dmu_not
                        v = v_not - v_not**2*(dmu_not**2 - 2*dv_not)
                        #obtain new batch likelihood approx.
                        v_l_inv = 1/v - 1/v_not
                        v_l_inv[ v_l_inv <= 0] = 1/self.INF
                        v_l_inv_mu = mu/v - mu_not/v_not
                        if damping_both:
                            v_l_inv = self.damping * v_l_inv + (1.0 - self.damping) * 1.0/self.v_l_neg[id]
                            v_l_inv_mu = self.damping * v_l_inv_mu + (1.0 - self.damping) * self.mu_l_neg[id]/self.v_l_neg[id]
                        n_batch_neg[id] += 1.0
                        v_l_inv_batch_neg[id] += v_l_inv
                        v_l_inv_mu_batch_neg[id] += v_l_inv_mu
                    curr = curr + 1
                    #print 'batch %d'%curr

                    if count == self.mini_batch:
                        #stochastic update
                        ind = np.nonzero(n_batch_pos)
                        if ind[0].size>0:
                            v_l_inv_pos = ((n_pos[ind] - n_batch_pos[ind]) * (1.0/self.v_l_pos[ind]) + v_l_inv_batch_pos[ind])/n_pos[ind]
                            v_l_inv_mu_pos = ((n_pos[ind] - n_batch_pos[ind]) * (self.mu_l_pos[ind]/self.v_l_pos[ind]) + v_l_inv_mu_batch_pos[ind])/n_pos[ind]
                            self.v_l_pos[ind] = 1.0/v_l_inv_pos
                            self.mu_l_pos[ind] = self.v_l_pos[ind]*v_l_inv_mu_pos
                        accumulate_ind = list(set().union(accumulate_ind, list(ind[0])))
                        ind = np.nonzero(n_batch_neg)
                        if ind[0].size>0:
                            v_l_inv_neg = ((n_neg[ind] - n_batch_neg[ind]) * (1.0/self.v_l_neg[ind]) + v_l_inv_batch_neg[ind])/n_neg[ind]
                            v_l_inv_mu_neg= ((n_neg[ind] - n_batch_neg[ind]) * (self.mu_l_neg[ind]/self.v_l_neg[ind]) + v_l_inv_mu_batch_neg[ind])/n_neg[ind]
                            self.v_l_neg[ind] = 1.0/v_l_inv_neg
                            self.mu_l_neg[ind] = self.v_l_neg[ind]*v_l_inv_mu_neg

                        accumulate_ind = list(set().union(accumulate_ind, list(ind[0])))
                        v_inv_all  = 1.0/self.v_p + n_pos*(1.0/self.v_l_pos) + n_neg*(1.0/self.v_l_neg)
                        v_inv_mu = self.mu_p/self.v_p + n_pos*(self.mu_l_pos/self.v_l_pos) + n_neg*(self.mu_l_neg/self.v_l_neg)
                        self.v = 1.0/v_inv_all
                        self.mu = self.v*v_inv_mu
                        #clear
                        count = 0
                        n_batch_pos = np.zeros(d)
                        n_batch_neg = np.zeros(d)
                        v_l_inv_batch_pos = np.zeros(d)
                        v_l_inv_batch_neg = np.zeros(d)
                        v_l_inv_mu_batch_pos = np.zeros(d)
                        v_l_inv_mu_batch_neg = np.zeros(d)

                        #we control how often we update the prior factors
                        if (curr/self.mini_batch) % n_batch_update_prior == 0:
                            #update prior factors
                            v_inv_not = 1/self.v[accumulate_ind] - 1/self.v_p[accumulate_ind]
                            v_not = 1/v_inv_not
                            mu_not = v_not * (self.mu[accumulate_ind]/self.v[accumulate_ind] - self.mu_p[accumulate_ind]/self.v_p[accumulate_ind])
                            v_tilt = 1/(1/v_not + 1/self.tau0)
                            mu_tilt = v_tilt * (mu_not/v_not)
                            #log N(0 | mu_not, v_not + tau0)
                            log_h = normal.logpdf(mu_not, scale = np.sqrt(v_not + self.tau0))
                            #log N(0 | mu_not, v_not)
                            log_g = normal.logpdf(mu_not, scale = np.sqrt(v_not))
                            rho_p =  log_h - log_g
                            sel_prob = expit(self.rho + rho_p)
                            mu = sel_prob * mu_tilt
                            v = sel_prob * (v_tilt + (1.0 - sel_prob)*mu_tilt**2)
                            #damping
                            self.rho_p[accumulate_ind] = self.damping * rho_p + (1 - self.damping) * self.rho_p[accumulate_ind]
                            v_p_inv = 1/v - v_inv_not
                            v_p_inv[ v_p_inv <= 0] = 1/self.INF
                            v_p_inv_mu = mu/v - mu_not/v_not
                            v_p_inv = self.damping * v_p_inv + (1 - self.damping) * 1/self.v_p[accumulate_ind]
                            v_p_inv_mu = self.damping * v_p_inv_mu + (1 - self.damping) * self.mu_p[accumulate_ind]/self.v_p[accumulate_ind]
                            self.v_p[accumulate_ind] = 1/v_p_inv
                            self.mu_p[accumulate_ind] = self.v_p[accumulate_ind] * v_p_inv_mu
                            #update global approx. dist.
                            self.r[accumulate_ind] = self.rho_p[accumulate_ind] + self.rho
                            v_inv_all =  v_inv_not  + 1.0/self.v_p[accumulate_ind]
                            v_inv_mu = mu_not/v_not + self.mu_p[accumulate_ind]/self.v_p[accumulate_ind]
                            self.v[accumulate_ind] = 1.0/v_inv_all
                            self.mu[accumulate_ind] = self.v[accumulate_ind] * v_inv_mu
                            accumulate_ind = []

                        if (curr/self.mini_batch)%10 == 0:
                            diff = np.sum(np.abs((1/old_v - 1/self.v)) + np.sum(np.abs(old_mu - self.mu)) + np.sum(np.abs(old_r - self.r)))/(old_v.size + old_mu.size + old_r.size)
                            print >>logger, 'epoch %d, %d batches, diff = %g'%(it, curr/self.mini_batch, diff)
                            logger.flush()
                            print 'epoch %d, %d batches, diff = %g'%(it, curr/self.mini_batch, diff)
                            if diff < self.tol:
                                break
                            old_v = self.v.copy()
                            old_mu = self.mu.copy()
                            old_r = self.r.copy()


                            if (curr/self.mini_batch)%1000==0:
                                pred = self.predict(Xtest)
                                fpr,tpr,th = metrics.roc_curve(ytest, pred, pos_label=1)
                                val = metrics.auc(fpr,tpr)
                                print >>logger, 'auc = %g, feature # = %d'%(val, np.sum(self.r>0))
                                print 'auc = %g, feature # = %d'%(val, np.sum(self.r>0))
                                elapse = time.clock() - start_time
                                start_time = time.clock()
                                print '1000 batches, take %g seconds'%elapse

            #evaluation
            pred = self.predict(Xtest)
            fpr,tpr,th = metrics.roc_curve(ytest, pred, pos_label=1)
            val = metrics.auc(fpr,tpr)
            print >>logger, 'epoch %d, tau0 = %g, auc = %g, feature # = %d'%(it, self.tau0, val, np.sum(self.r>0))
            print 'epoch %d, tau0 = %g, auc = %g, feature # = %d'%(it, self.tau0, val, np.sum(self.r>0))
            it = it + 1
            curr = 0






    def predict(self, Xtest):
        d = self.mu.size
        if d == Xtest.shape[1] + 1:
            Xtest = np.hstack([Xtest, np.ones(Xtest.shape[0]).reshape([Xtest.shape[0],1])])
        elif d != Xtest.shape[1]:
            print 'inconsistent feature number'
            return

        #pred_prob = normal.cdf( np.dot(Xtest,self.mu) / np.sqrt( np.dot(Xtest**2, self.v) + 1 ) )
        #pred_prob = np.dot(Xtest,self.mu)
        mu = self.mu * (self.r>0)
        #mu = self.mu * (expit(self.r)>0.5)
        #v = self.v * (expit(self.r)>0.5)
        pred_prob = Xtest.dot(mu)
        #pred_prob = np.dot(Xtest, mu)
        #pred_prob = normal.cdf( np.dot(Xtest, mu) / np.sqrt( np.dot(Xtest**2, v) + 1 ) )
        return pred_prob



def test_ctr_large_sep_weighted():
    training_file = '/tmp/large-ctr-pxu-train'
    testing_file = '/tmp/large-ctr-pxu-test'
    fea2id = load_feature_id('feature_to_id.txt')
    fea2stats = load_feature_stats('mean_std_continuous_features.txt')
    '''
    Xtest,ytest = load_test_data('/tmp/large-ctr-pxu-test', fea2id, fea2stats)
    save_sparse_csr('/tmp/large-ctr-pxu-test-X', Xtest)
    np.save('/tmp/large-ctr-pxu-test-y', ytest)
    sys.exit(1)
    '''
    Xtest = load_sparse_csr('/tmp/large-ctr-pxu-test-X.npz')
    ytest = np.load('/tmp/large-ctr-pxu-test-y.npy')

    d = 204327
    #calc_feature_appearence_separately(d, fea2id, training_file)
    n_pos = np.load('feature_appearence_pos.npy')
    n_neg = np.load('feature_appearence_neg.npy')

    ep = EPSS(d, rho0 = 0.0000001, n_epoch = 1, mini_batch = 100, tol = 1e-5, damping = 0.9, tau0 = 1.0)
    with open('logger-2.txt','w') as f:
        ep.train_stochastic_multi_rate(d, n_pos, n_neg, training_file, fea2id, fea2stats, Xtest, ytest, f,  n_batch_update_prior = 1, damping_both = True)

    w = ep.mu * (expit(ep.r)>0.5)
    np.save('model-w-8.npy', w)
    r = ep.r
    np.save('sel-w-8.npy', r)





def test_ctr_large_sep():
    training_file = '/tmp/large-ctr-pxu-train'
    testing_file = '/tmp/large-ctr-pxu-test'
    fea2id = load_feature_id('feature_to_id.txt')
    fea2stats = load_feature_stats('mean_std_continuous_features.txt')
    '''
    Xtest,ytest = load_test_data('/tmp/large-ctr-pxu-test', fea2id, fea2stats)
    save_sparse_csr('/tmp/large-ctr-pxu-test-X', Xtest)
    np.save('/tmp/large-ctr-pxu-test-y', ytest)
    sys.exit(1)
    '''
    Xtest = load_sparse_csr('/tmp/large-ctr-pxu-test-X.npz')
    ytest = np.load('/tmp/large-ctr-pxu-test-y.npy')

    d = 204327
    ep = EPSS(d, rho0 = 0.5, n_epoch = 10, mini_batch = 100, tol = 1e-5, damping = 0.9, tau0 = 1.0)
    with open('logger.txt','w') as f:
        ep.train_stochastic_single_rate(d, training_file, fea2id, fea2stats, Xtest, ytest, f,  n_batch_update_prior = 1, damping_both = True)

    w = ep.mu * (expit(ep.r)>0.5)
    np.save('model-2.npy', w)
    r = ep.r
    np.save('sel-2.npy', r)


def load_feature_id(file_path):
    res = {}
    with open(file_path, 'r') as f:
        for line in f:
            name,id = line.strip().split('\t')
            res[ name ] = int(id)
    return res

def load_feature_stats(file_path):
    res = {}
    with open(file_path, 'r') as f:
        for line in f:
            name,mean,std= line.strip().split('\t')
            res[ name ] = [float(mean), float(std)]
    return res



def load_test_data(file_path, fea2id, fea2stats):
    y = []
    row_ind = []
    col_ind = []
    data = []
    row_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            y.append(int(items[0]))
            for item in items[feature_off:]:
                key_val = item.split(':')
                if key_val[0] not in fea2id:
                    continue
                id = fea2id[ key_val[0] ]
                if len(key_val) == 1:
                    data.append(1.0)
                else:
                    data.append(float(key_val[1]))
                    #data.append( normalize(float(key_val[1]), fea2stats[key_val[0]]) )
                row_ind.append(row_num)
                col_ind.append(id)
            #append intercept
            data.append(1.0)
            row_ind.append(row_num)
            col_ind.append(len(fea2id))
            row_num = row_num + 1
            if row_num%10000 == 0:
                print row_num
    Xtest = csr_matrix((data, (row_ind, col_ind)))
    y = np.array(y)
    return [Xtest,y]



def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices, indptr =array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])



def tune_news20(tau0):
    training_file = '../../../data/news20/news20.binary.shuf.vw.subtrain'
    testing_file = '../../../data/news20/news20.binary.shuf.vw.validation'
    fea2id = load_feature_id('feature_to_id.txt')
    #zsd, note i used a very hacking way to disable the normalization effect
    fea2stats = load_feature_stats('mean_std_continuous_features.txt')

    '''
    Xtest,ytest = load_test_data(testing_file, fea2id, fea2stats)
    save_sparse_csr('./news20.validate.X', Xtest)
    np.save('./news20.validate.y', ytest)
    sys.exit(1)
    '''

    Xtest = load_sparse_csr('./news20.validate.X.npz')
    ytest = np.load('./news20.validate.y.npy')

    d = len(fea2id)
    #calc_feature_appearence_separately(d, fea2id, training_file)
    n_pos = np.load('feature_appearence_pos.npy')
    n_neg = np.load('feature_appearence_neg.npy')

    ep = EPSS(d, rho0 = 0.5, n_epoch = 1, mini_batch = 100, tol = 1e-5, damping = 0.9, tau0 = tau0)
    with open('logger-news20-tune.txt','a+') as f:
        ep.train_stochastic_multi_rate(d, n_pos, n_neg, training_file, fea2id, fea2stats, Xtest, ytest, f,  n_batch_update_prior = 1, damping_both = True)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage %s <tau0>'%sys.argv[0]
        sys.exit(1)

    np.random.seed(0)
    tune_news20(float(sys.argv[1]))


