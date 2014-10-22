'''Check the efficiency correction for the method of moments.

The considered decays is like B -> K ll. Write PDF as sum over Legendre polynomials.

1. Draw samples from a uniform distribution of \cos \theta, and compute integrals, then reweight by P * eps
2. Draw from P * eps using MCMC from pypmc

Both methods give similar estimates of the angular
observables. Amazingly: 1. much simpler to implement, and yields the
true values up to machine precision every time. How?!

3. Plot the regions
4. Optimize using nlopt

'''
from __future__ import division, print_function

# import nlopt
import numpy as np
from numpy.polynomial.legendre import Legendre
import pypmc
from matplotlib import pyplot as plt
from scipy.integrate import quad

# np.random.seed(123465123)

# define a proposal
prop_dof   = 5.
prop_sigma = np.array([[3]])
prop = pypmc.density.student_t.LocalStudentT(prop_sigma, prop_dof)

class BtoKll:
    '''Differential decay rate of B -> K ll for fixed q^2.'''
    def __init__(self, eps=2/3*np.array([0.6, 0.05, -0.55])):
        '''eps: coefficients of Legendre polynomials of acceptance. Make sure it integrates to less than one.'''
        self._accept = Legendre(eps)
        # first 0.5 is fixed by normalization!
        self._S = np.zeros(3)
        self._S[0] = 0.5

    def differential_rate(self, S, x):
        # x = np.cos(theta)
        x = np.asarray(x)

        # first coefficient fixed
        self._S[1:] = S[:]
        P = Legendre(self._S)(x)

        # A_FB, F_H = S[0], S[1]
        # # Eq. (1) in arXiv:1403.8045v2
        # P = 0.75 * (1. - F_H) * (1 - x**2) + 0.5 * F_H + A_FB * x
        # ensure positive decay rate
        if (P <= 0.0).any():
            return -np.inf
        return np.log(P).sum()

    # def prior(self, S):
    #     A_FB, F_H = S[0], S[1]
    #     if abs(A_FB) > +1:
    #         return -np.inf
    #     if F_H < 0. or F_H > 3.:
    #         return -np.inf
    #     if abs(A_FB) > F_H / 2:
    #         return -np.inf
    #     return 0.

    def acceptance(self, x):
        eps = self._accept(x)
        return np.log(eps).sum()

decay = BtoKll()

# the "true" values of the parameters
S = np.array([0.37, 0.22])
log_target = lambda x: decay.differential_rate(S, x) + decay.acceptance(x)

# good initialization
start = np.array([0.6])

# sample inside fixed interval
indicator = pypmc.tools.indicator.hyperrectangle([-1], [1])

# define the markov chain object
mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, prop, start, indicator=indicator)

# run 100,000 steps adapting the proposal every 500 steps
# hereby save the accept count which is returned by mc.run
accept_count = 0
for i in range(100):
    accept_count += mc.run(500)
    mc.adapt()
    if i == 2:
        mc.history.clear()

# extract a reference to the history of all visited points
samples = mc.history[:]
accept_rate = float(accept_count) / len(samples)
print("The chain accepted %4.2f%% of the proposed points" % (accept_rate * 100) )

plt.hist(samples, bins=40)
# plt.show()
plt.savefig('cos_theta.pdf')

# range for A_FB, F_H
x_begin, x_end = -0.1, 0.1
y_begin, y_end = 0, 0.2

# reduce samples to a number similar to what LHCb sees
thin = 10 #len(samples) // 3000
thinned_samples = samples[::thin]
print("N samples", len(thinned_samples))

likelihood = lambda S: decay.differential_rate(S, thinned_samples)
posterior = lambda S: likelihood(S)# + decay.prior(S)

###
# Estimate with method of moments
###

order = 3

def mixing_uniform(samples):
    M = np.zeros((order, order))
    for i in range(order):
        tilde = np.zeros(order)
        tilde[i] = (2 * i + 1)
        for j in range(order):
            normal = np.zeros_like(tilde)
            normal[j] = 1
            # volume from uniform proposal * normalization from Legendre product * sum over target
            target = Legendre(tilde) * Legendre(normal) * decay._accept
            M[i,j] = 2 / len(samples) / 2 * np.sum(target(samples))
    return M

def integral_uniform(samples):
    '''Uncorrected estimate from uniform samples'''
    coefficients = np.zeros(order)
    coefficients[0] = 0.5
    coefficients[1:3] = S[:]

    result = np.zeros_like(coefficients)
    for i in range(order):
        tilde = np.zeros_like(coefficients)
        tilde[i] =  (2 * i + 1) / 2
        target = Legendre(tilde) * Legendre(coefficients) * decay._accept
        result[i] = 2 / len(samples) * np.sum(target(samples))

    return result

def integralMC(samples):
    '''MC estimate of integral expectation'''

    # activate only one coefficient i at a time
    S = np.zeros(order)
    for i in range(order):
        coefficients = np.zeros(order)
        coefficients[i] = (2 * i + 1) / 2
        ftilde = lambda x: Legendre(coefficients)(x)
        S[i] = 1. / len(samples) * np.sum(ftilde(samples))

    return S

# dual basis
def mixing():
    '''Return the mixing matrix'''
    M = np.zeros((order, order))
    for j in range(order):

        # generate samples only from comp. j
        coefficients = np.zeros(order)
        coefficients[0] = 0.5
        if j > 0:
            coefficients[j] = 0.5
        start = np.array([0.1])
        def log_target(x):
            P = Legendre(coefficients)(x)
            return np.log(P) + decay.acceptance(x)

        mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, prop, start, indicator=indicator)
        for i in range(60):
            mc.run(200)
            mc.adapt()
            if i == 2:
                mc.history.clear()

        def target(x):
            return Legendre(coefficients)(x) * decay._accept(x)

        # integral of P * eps
        norm = quad(target, -1, +1)[0]

        # extract a reference to the history of all visited points
        thinned_samples = mc.history[:][::10]
        M[:, j] = norm * integralMC(thinned_samples) / coefficients[j]
        if j >= 1:
            M[:,j] -= M[:, 0]

    return M

def single_uniform_analysis(verbose=True):
    # draw uniform samples
    uniform_samples = np.random.uniform(-1, 1, 10000)
    M_unif = mixing_uniform(uniform_samples)
    S_uncorr_unif = integral_uniform(uniform_samples)
    S_corrected_unif = np.linalg.inv(M_unif).dot(S_uncorr_unif)

    if verbose:
        print("\nuniform samples:\n")
        print("mixing matrix")
        print(M_unif)
        print("uncorrected S", S_uncorr_unif)
        print("corrected S", S_corrected_unif)
        print("true S", S)

def single_analysis(verbose=False):
    coefficients = np.zeros(order)
    coefficients[0] = 0.5
    coefficients[1:3] = S[:]
    target = Legendre(coefficients) * decay._accept

    norm = quad(target, -1, 1)[0]

    S_uncorr = norm * integralMC(thinned_samples)

    # build up matrix
    M = mixing()
    Minv = np.linalg.inv(M)
    S_corrected = Minv.dot(S_uncorr)

    if verbose:
        print("\nMCMC samples:\n")
        print("mixing matrix")
        print(M)
        print("uncorrected S", S_uncorr)
        print("corrected S", S_corrected)
        print("true S", S)

    return S_corrected

def repeat_analysis(N=50):
    collect_S = np.empty((N, order))

    for i in range(N):
        collect_S[i] = single_analysis()

    print(collect_S)
    print("mean")
    print(np.mean(collect_S, axis=0))
    print("std. dev")
    print(np.sqrt(np.var(collect_S, ddof=1, axis=0)))
    print("true S", S)

single_uniform_analysis(verbose=True)
single_analysis(verbose=True)
#repeat_analysis(25)

exit(0)

###
# optimize with nlopt
###
import nlopt

def nlopt_target(S, grad):
    if grad:
        return NotImplementedError
    return posterior(S)

D = len(S)
opt = nlopt.opt(nlopt.LN_COBYLA, D)
opt.set_max_objective(nlopt_target)
opt.set_lower_bounds([x_begin, y_begin])
opt.set_upper_bounds([x_end, y_end])
tol = 1e-14
opt.set_ftol_abs(tol)
opt.set_xtol_rel(np.sqrt(tol))
opt.set_maxeval(2000)
S_opt = opt.optimize(S.copy())
print('COBYLA:', S_opt)
#exit(0)

###
# now plot the posterior
###

prop_sigma = 0.02 * np.eye(2)
prop = pypmc.density.student_t.LocalGauss(prop_sigma)
start = S.copy()

# sample inside fixed interval
indicator = pypmc.tools.indicator.hyperrectangle([x_begin, y_begin], [x_end, y_end])

# define the markov chain object
mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(posterior, prop, start, indicator=indicator)

# run 100,000 steps adapting the proposal every 500 steps
# hereby save the accept count which is returned by mc.run
accept_count = 0
for i in range(40):
    accept_count += mc.run(500)
    mc.adapt()
    if i == 2:
        mc.history.clear()

# extract a reference to the history of all visited points
samples = mc.history[:]
accept_rate = float(accept_count) / len(samples)
print("The chain accepted %4.2f%% of the proposed points" % (accept_rate * 100) )

plt.figure()
plt.hexbin(samples[:,0], samples[:,1], gridsize=40, cmap='gray_r')
plt.savefig('posterior.pdf')

# define the grid to plot the banana on
N_x_points = 100
N_y_points = 100
# N points, but N-1 intervals in between
x_delta = (x_end - x_begin) / (N_x_points - 1)
y_delta = (y_end - y_begin) / (N_y_points - 1)
# mgrid omits last point, so add extra delta s.t. *_end really is the last point
grid = np.mgrid[x_begin : x_end + x_delta:x_delta, y_begin:y_end + y_delta:y_delta]
# evaluate the target all over the grid
values = np.empty(list(grid.shape)[1:])
assert values.shape == (grid.shape[1], grid.shape[2])
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        values[i,j] = likelihood(grid[:,i,j])

# rescale relative to maximum
values -= values.max()
values = np.exp(values)
print(values)

plt.figure();
plt.contourf(grid[0], grid[1], values, 5, cmap='gray_r')

# add allowed region
plt.plot([x_begin, 0.0], [y_end, 0.], 'k')
plt.plot([0.0, x_end], [0., y_end], 'k')

# add input parameters
plt.plot(S[0], S[1], marker='*', color='blue', markersize=20)

plt.xlabel(r'$A_{\rm FB}$')
plt.ylabel(r'$F_{H}$')
plt.title(r'$B^+ \to K^+ \mu^+ \mu^-$ at $q^2 \in [1,6]$ GeV$^2$')

plt.savefig('llh.pdf')
