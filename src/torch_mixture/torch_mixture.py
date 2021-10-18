import os
import sklearn.cluster
import numpy.random
import numpy as np
import pandas as pd
import torch
import torch.distributions as tdist

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Mixture:
    def __init__(self,data,distributions,cluster_initialize=False):
        '''
           data: 1-D numpy array of data to fit
           distributions: list of torch.distributions objects containing distributions comprising mixture
           cluster_initialize: if True then use k-means to cluster data to initialize each mixture distribution
                               by mean and variance of cluster. If False then initialization is random
        '''
        ## initialize parameters of distribution
        self.distributions = distributions
        if cluster_initialize:
            self.cluster_initialize(data,distributions)
        else:
            self.random_initialize(data,distributions)
        # initialize mixing probabilities, equal prob for each distribution in the mixture
        self.weights = torch.ones(len(self.distributions),device=device)/len(self.distributions)
        self.data = torch.tensor(data,requires_grad=False,dtype=torch.float32,device=device)
        print(f'at initialization negative log-likelihood is {-self.log_lik().sum().item():,}')
    def calc_distributions(self):
        '''return list of torch.distributions set with current parameters'''
        distributions = [distr(**{k:v[0] if v[1] is None else v[1](v[0]) for k,v in param.items()})
                                   for param,distr in zip(self.param_dict,self.distributions)]
        return distributions
    def fit(self,learning_rate,n_iter,EM_extended=False):
        '''
           fit mixture distribution to data
           learning_rate: learning rate used by Adam optimizer
           n_iter: number of iterations to run the optimizer
           EM_extended: if True then conditional mixture probabilities are part of the M-step gradient, if False
                        then only the data likelihood incorporates parameters of distributions
           At each iteration Q function is constructed based on updated parameters. Gradient of log-likelihood
           is computed and optimizer takes one step before conditional mixture probabilities are recomputed,
           log-likelihood is recomputed and gradient is calculated again.
           Method assumes that all data can fit in GPU memory which should be the case with univariate data
        '''
        optimizer = torch.optim.Adam(self.params, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        for iter_i in range(n_iter):
            # initialize distribution objects with updated parameters
            distributions = self.calc_distributions()
            # compute conditional log-likelihoods and likelihoods
            logP = torch.column_stack([distr.log_prob(self.data) for distr in distributions])
            P = torch.exp(logP)
            # compute conditional mixing probabilities at each sample
            # If in EM_extended mode then conditional mixture weights will be treated as functions
            # of parameters and accounted for when computing the gradient; otherwise the classical EM
            # representation of the Q function is computed
            if EM_extended:
                W = (P * self.weights)/torch.unsqueeze(self.weights @ P.T,-1)
            else:
                with torch.no_grad():
                    W = (P * self.weights)/torch.unsqueeze(self.weights @ P.T,-1)
            # compute full log-likelihood
            log_lik = -(logP*W).sum()
            # update unconditional mixing probs to be means of conditional mixing probs at each sample
            with torch.no_grad():
                self.weights = W.mean(0)
            optimizer.zero_grad()
            log_lik.backward(retain_graph=False)
            # this is the modified M step computation
            optimizer.step()
            scheduler.step(log_lik)
        all_grads = numpy.array([v.grad.item() for v in self.params])
        print(f'negative log-likelihood is {-self.log_lik().sum().item():,}')
        print(f'relative gradients are {abs(all_grads/log_lik.item())}')
        with np.printoptions(precision=2, suppress=True, threshold=5):
            print(f'weights: {self.weights.cpu().numpy()}')
    def log_lik(self):
        ''' compute log likelihood of the data wrt to the distributions'''
        distributions = self.calc_distributions()
        logP = torch.column_stack([distr.log_prob(self.data) for distr in distributions])
        P = torch.exp(logP)
        log_lik = torch.log(self.weights @ P.T)
        return log_lik
    ## all code below is for setting parameters for initialization of distributions; it is not related
    ## to E-M algorithm
    def cluster_initialize(self,data,distributions,batch_size=256*os.cpu_count()):
        ''' partition data into clusters, one cluster for each distribution
            calculate mean and variance of each cluster and initialize parameters of each distribution
            to match corresponding mean and variance
            batch_size argument is needed for MiniBatchKMeans; sklearn recommends 256*number of cores
        '''
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=len(distributions), init='k-means++', max_iter=100, batch_size=batch_size, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
        labels = kmeans.fit_predict(np.atleast_2d(data).T)
        loc_scale = pd.Series(data).groupby(labels).agg([np.mean,np.std]).values
        loc_scale[:,1] *= 5
        self.initialize_all(loc_scale,distributions)
    def random_initialize(self,data,distributions):
        ''' compute mean and variance of data, sample from Normal with computed mean and variance
            to create random means for each mixing distribution, then initialize each distribution
            with that mean and original variance
        '''
        mu = np.mean(data)
        sigma = np.std(data)
        mus = numpy.random.randn(len(distributions))*sigma + mu
        ## make standard deviation high to make sure each mixture gets non-zero probability
        loc_scale = [(v,sigma*3) for v in mus]
        self.initialize_all(loc_scale,distributions)
    def initialize_all(self,loc_scale,distributions):
        import operator
        self.param_dict = [self.loc_scale_initialize(l_s,distr) for l_s,distr in zip(loc_scale,distributions)]
        self.params= sum([list(map(operator.itemgetter(0),v.values())) for v in self.param_dict],[])
    def loc_scale_initialize(self,l_s,distribution):
        mu,sigma = np.float32(l_s[0]),np.float32(l_s[1])
        if distribution in _dist_dict:
            return _dist_dict[distribution](mu,sigma)
        else:
            raise Exception(f'distribution {distribution} is not implemented')

        
######## Functions that convert mean and stddev to parameter values for each supported distribution ##########
######## In order to satisfy parameter constraints e.g. variance>0 each parameter has a function #############
######## that is applied to the parameter before passing it to distribution constructor ###########
def moments_Normal(mu,sigma):
    loc = torch.tensor(mu,device=device,requires_grad=True,dtype=torch.float32)
    scale = torch.tensor(np.sqrt(sigma),device=device,requires_grad=True,dtype=torch.float32)
    return {'loc':(loc,None),'scale':(scale,torch.square)}
def moments_LogNormal(mu,sigma):
    loc = torch.tensor(np.sqrt(np.log(mu**2/np.sqrt(mu**2+sigma**2))),device=device,requires_grad=True,dtype=torch.float32)
    scale = torch.tensor(np.sqrt(np.sqrt(np.log(1+sigma**2/mu**2))),device=device,requires_grad=True,dtype=torch.float32)
    return {'loc':(loc,torch.square),'scale':(scale,torch.square)}
def moments_Gamma(mu,sigma):
    alpha = torch.tensor(np.sqrt(mu**2/sigma**2),device=device,requires_grad=True,dtype=torch.float32)
    beta = torch.tensor(np.sqrt(mu/sigma**2),device=device,requires_grad=True,dtype=torch.float32)
    return {'concentration':(alpha,torch.square),'rate':(beta,torch.square)}
def moments_Gumbel(mu,sigma):
    scale_np = sigma*np.float32(np.sqrt(6)/pi)
    scale = torch.tensor(np.sqrt(scale_np),device=device,requires_grad=True,dtype=torch.float32)
    loc = torch.tensor(mu - scale_np*0.5772156649,device=device,requires_grad=True,dtype=torch.float32)
    return {'loc':(loc,None),'scale':(scale,torch.square)}
def moments_Cauchy(mu,sigma):
    loc = torch.tensor(mu,device=device,requires_grad=True,dtype=torch.float32)
    scale = torch.tensor(np.sqrt(sigma),device=device,requires_grad=True,dtype=torch.float32)
    return {'loc':(loc,None),'scale':(scale,torch.square)}
def moments_StudentT(mu,sigma):
    loc = torch.tensor(mu,device=device,requires_grad=True,dtype=torch.float32)
    scale = torch.tensor(np.sqrt(sigma/np.sqrt(3)),device=device,requires_grad=True,dtype=torch.float32)
    df = torch.tensor(3,device=device,requires_grad=True,dtype=torch.float32)
    return {'df':(df,None),'loc':(loc,None),'scale':(scale,torch.square)}
def moments_Laplace(mu,sigma):
    loc = torch.tensor(mu,device=device,requires_grad=True,dtype=torch.float32)
    scale = torch.tensor(np.sqrt(sigma/np.sqrt(2)),device=device,requires_grad=True,dtype=torch.float32)
    return {'loc':(loc,None),'scale':(scale,torch.square)}
def moments_Beta(mu,sigma):
    v = np.square(sigma)
    alpha = -(mu * (np.square(mu) - mu + v))/v 
    beta  = (mu - 1) * (np.square(mu) - mu + v)/v 
    alpha = torch.tensor(np.sqrt(alpha),device=device,requires_grad=True,dtype=torch.float32)
    beta = torch.tensor(np.sqrt(beta),device=device,requires_grad=True,dtype=torch.float32)
    return {'concentration1':(alpha,torch.square),'concentration0':(beta,torch.square)}
def moments_Exponential(mu,sigma):
    rate = torch.tensor(1/np.sqrt(mu),device=device,requires_grad=True,dtype=torch.float32)
    return {'rate':(rate,torch.square)}
def moments_FisherSnedecor(mu,sigma):
    v = np.square(sigma)
    d1 = -2*mu**2/((mu-1)*mu**2+(mu-2)*v)
    df1 = torch.tensor(np.sqrt(d1),device=device,requires_grad=True,dtype=torch.float32)
    df2 = torch.tensor(np.sqrt(2*mu/(mu-1)),device=device,requires_grad=True,dtype=torch.float32)
    return {'df1':(df1,torch.square),'df2':(df2,torch.square)}
def moments_HalfCauchy(mu,sigma):
    scale = torch.tensor(np.sqrt(sigma),device=device,requires_grad=True,dtype=torch.float32)
    return {'scale':(scale,torch.square)}
def moments_HalfNormal(mu,sigma):
    scale = torch.tensor(np.sqrt(sigma),device=device,requires_grad=True,dtype=torch.float32)
    return {'scale':(scale,torch.square)}
def moments_Pareto(mu,sigma):
    v = sigma**2
    mv = mu**2+v
    s1 = (mv-np.sqrt(v*mv))/mu
    s2 = mu**2/(np.sqrt(v*mv)-v)
    scale = torch.tensor(np.sqrt(s1),device=device,requires_grad=True,dtype=torch.float32)
    dshape = torch.tensor(np.sqrt(s2),device=device,requires_grad=True,dtype=torch.float32)
    return {'scale':(scale,torch.square),'alpha':(dshape,torch.square)}
def moments_Poisson(mu,sigma):
    rate = torch.tensor(np.sqrt(mu),device=device,requires_grad=True,dtype=torch.float32)
    return {'rate':(rate,torch.square)}
def moments_Bernoulli(mu,sigma):
    odds = np.sqrt((1-mu)/mu)
    prob = torch.tensor(odds,device=device,requires_grad=True,dtype=torch.float32)
    return {'probs':(prob,lambda v: 1/(1+torch.square(v)))}
def moments_Uniform(mu,sigma):
    v2 = sigma*np.sqrt(3)
    a = torch.tensor(mu-v2,device=device,requires_grad=True,dtype=torch.float32)
    b = torch.tensor(mu+v2,device=device,requires_grad=True,dtype=torch.float32)
    return {'low':(a,None),'high':(b,None)}

          
_dist_dict = {tdist.Normal:moments_Normal,tdist.LogNormal:moments_LogNormal,tdist.Gamma:moments_Gamma,tdist.Gumbel:moments_Gumbel,tdist.Cauchy:moments_Cauchy,tdist.StudentT:moments_StudentT,tdist.Laplace:moments_Laplace,tdist.Beta:moments_Beta,tdist.Exponential:moments_Exponential,tdist.FisherSnedecor:moments_FisherSnedecor,tdist.HalfCauchy:moments_HalfCauchy,tdist.HalfNormal:moments_HalfNormal,tdist.Pareto:moments_Pareto,tdist.Poisson:moments_Poisson,tdist.Bernoulli:moments_Bernoulli,tdist.Uniform:moments_Uniform}

def available_distributions():
    ''' return list of available distrbutions that can be used in mixtures '''
    return list(_dist_dict.keys())
