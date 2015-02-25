#!/usr/bin/python
import numpy as np
import numpy.matlib as matlib
from numpy.polynomial.legendre import Legendre

import pypmc

from matplotlib import pyplot as plt

import scipy as sp
import scipy.integrate as integrate
from scipy.misc import factorial

import sys

class Process:
    def dim(self):
        '''
        Return the number of kinematic variables
        '''
        raise NotImplementedError()

    def pdf(self, x):
        '''
        Evaluate the PDF
        S: array of angular observables
        x: array of angular variables
        '''
        raise NotImplementedError()

    def log_pdf(self, x):
        '''
        Evaluate the natural logarithm of the PDF
        '''
        return np.log(self.pdf(x))

    def number_of_components(self):
        return NotImplementedError()

    def component(self, i, x):
        '''
        i: index of the component of the PDF
        x: array of angular variables
        '''
        raise NotImplementedError()

    def dual_component(self, i, x):
        '''
        i: index of the component of the PDF
        x: array of angular variables
        '''
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def integrate(self, function):
        raise NotImplementedError()

    def test(self):
        for i in range(self.number_of_components()):
            for j in range(self.number_of_components()):
                integrand = lambda x: self.dual_component(i, x) * self.component(j, x)
                print("test: i=%d, j=%d, result=%f" % (i, j, self.integrate(integrand)))


class BToKDilepton(Process):
    def __init__(self, S):
        self.S = S

    @staticmethod
    def make(S):
        return BToKDilepton(S)

    def dim(self):
        return 1

    def pdf(self, x):
        result = 0
        for i in [0, 1, 2]:
            result += self.S[i] * self.component(i, x)
        return result

    def number_of_components(self):
        return 3

    def component(self, i, x):
        coeffs = [0] * i
        coeffs.append(1.0)
        f = Legendre(coeffs)
        return f(x[0])

    def dual_component(self, i, x):
        return (2.0 * i + 1.0) / 2.0 * self.component(i, x)

    def start(self):
        return [np.random.uniform(-1, +1)]

    def integrate(self, function):
        # pdf and acceptance expect x as an array of kinematic variables: adjust for that!
        integrand = lambda x1: function([x1])
        integral, error = sp.integrate.quad(integrand, -1, +1)
        return integral

    def plot(self, samples, name):
        plt.figure()
        n, bins, patches = plt.hist(samples, 25 + 1, normed=1, facecolor='green', alpha=0.75)
        plt.savefig('samples-btokll-%s.pdf' % name)


class LambdaBToLambdaDilepton(Process):
    angular_momenta = [
        (0, 0,  0),
        (1, 0,  0),
        (2, 0,  0),
        (0, 1,  0),
        (1, 1,  0),
        (2, 1,  0),
        (1, 1, -1),
        (2, 1, -1),
        (1, 1, +1),
        (2, 1, +1)
    ]

    def __init__(self, S):
        self.S = S

    @staticmethod
    def make(S):
        return LambdaBToLambdaDilepton(S)

    def dim(self):
        return 3

    def pdf(self, x):
        result = 0
        for i in range(self.number_of_components()):
            result += self.S[i] * self.component(i, x)
        return result

    def number_of_components(self):
        return 10

    def component(self, i, x):
        c_th_1 = x[0]
        c_th_2 = x[1]
        th_3 = x[2]
        (l1, l2, m) = self.angular_momenta[i]

        result = 0.0
        abs_m = abs(m)
        if m == 0:
            result = 1.0
        elif m > 0:
            result = np.cos(abs_m * th_3)
        elif m < 0:
            result = np.sin(abs_m * th_3)

        m = abs_m

        result *= np.sqrt(factorial(l1 - m) * factorial(l2 - m) / (factorial(l1 + m) * factorial(l2 + m)))
        result *= sp.special.lpmv(m, l1, c_th_1) * sp.special.lpmv(m, l2, c_th_2)

        return result

    def dual_component(self, i, x):
        c_th_1 = x[0]
        c_th_2 = x[1]
        th_3 = x[2]
        (l1, l2, m) = self.angular_momenta[i]

        result = 0.0
        abs_m = abs(m)
        if m == 0:
            result = 1.0
        elif m > 0:
            result = 2.0 * np.cos(abs_m * th_3)
        elif m < 0:
            result = 2.0 * np.sin(abs_m * th_3)

        m = abs_m

        result *= np.sqrt(factorial(l1 - m) * factorial(l2 - m) / (factorial(l1 + m) * factorial(l2 + m)))
        result *= sp.special.lpmv(m, l1, c_th_1) * sp.special.lpmv(m, l2, c_th_2)
        result *= (2.0 * l1 + 1.0) * (2.0 * l2 + 1.0) / (8.0 * np.pi)
        return result

    def start(self):
        return [np.random.uniform(-1.0, +1.0), np.random.uniform(-1.0, +1.0), np.random.uniform(0.0, 2.0 * np.pi)]

    def integrate(self, function):
        # pdf and acceptance expect x as an array of kinematic variables: adjust for that!
        opts = [
            {"epsabs": 1e-5, "limit": 10},
            {"epsabs": 1e-5, "limit": 10},
            {"epsabs": 1e-5, "limit": 10}
        ]
        integrand = lambda x1,x2,x3: function([x1,x2,x3])
        integral, error = sp.integrate.nquad(integrand, [(-1.0, +1.0), (-1.0, +1.0), (0.0, 2.0 * np.pi)],
                    opts=opts
                )
        return integral

    def plot(self, samples):
        plt.figure()
        n, bins, patches = plt.hist(samples, 25 + 1, normed=1, facecolor='green', alpha=0.75)
        plt.savefig('lambdabtolambdall.pdf')


class Sampler:
    def __init__(self, dim):
        '''
        dim: dimensions of the phase space
        '''
        prop_dof   = 5
        prop_sigma = np.diag([3.] * dim)
        self.prop = pypmc.density.student_t.LocalStudentT(prop_sigma, prop_dof)

    def draw(self, log_target, start, indicator, chunks, chunk_size):
        self.mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, self.prop, start, indicator=indicator)
        accept_count = 0
        for i in range(chunks):
            accept_count += self.mc.run(chunk_size)
            self.mc.adapt()
            if i == 2:
                self.mc.history.clear()

        # extract a reference to the history of all visited points
        samples = self.mc.history[:]
        accept_rate = float(accept_count) / len(samples)
        #print("The chain accepted %4.2f%% of the proposed points" % (accept_rate * 100) )
        return samples

def unfolding_matrix(_process, acceptance, components, chunks, chunk_size, ovalue):
        if "B->Kll" == _process:
            process = BToKDilepton([0.5, 0.0, 0.0])
            factory = BToKDilepton.make
            indicator = pypmc.tools.indicator.hyperrectangle([-1], [1])
        elif "Lambda_b->Lambdall" == _process:
            process = LambdaBToLambdaDilepton([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            factory = LambdaBToLambdaDilepton.make
            indicator = pypmc.tools.indicator.hyperrectangle([-1.0, -1.0, 0.0], [+1.0, +1.0, 2.0 * np.pi])
        else:
            raise NotImplementedError("Unknown process %s", process)

        skip_index = int(chunks * chunk_size * 0.05)

        sampler = Sampler(process.dim())
        log_acceptance = lambda x: np.log(acceptance(x))
        raw_moments = np.matlib.zeros((components, components))
        for m in range(components): # index named as in draft
            S = [0.0] * components
            S[0] = ovalue
            S[m] = ovalue
            process = factory(S)
            Rm = process.integrate(lambda x: process.pdf(x) * acceptance(x))
            log_target = lambda x: process.log_pdf(x) + log_acceptance(x)
            samples = sampler.draw(log_target, start=process.start(), indicator=indicator, chunks=chunks, chunk_size=chunk_size)[skip_index:]
            for i in range(components):
                estimator = 0
                # discard burn-in of 2000 samples
                for x in samples:
                    estimator += process.dual_component(i, x)
                estimator *= Rm / len(samples)

                raw_moments[i, m] = estimator

        M = np.array(raw_moments) / ovalue
        for j in range(1, components):
            M[:, j] -= M[:, 0]

        return M
        #np.linalg.inv(M)

def raw_moments(process, acceptance, R, components, chunks, chunk_size):
        skip_index = int(chunks * chunk_size * 0.05)

        sampler = Sampler(process.dim())
        log_acceptance = lambda x: np.log(acceptance(x))
        raw_moments = [0.0, 0.0, 0.0, 0.0, 0.0]
        log_target = lambda x: process.log_pdf(x) + log_acceptance(x)
        indicator = pypmc.tools.indicator.hyperrectangle([-1], [1])
        samples = sampler.draw(log_target, start=process.start(), indicator=indicator, chunks=chunks, chunk_size=chunk_size)[skip_index:]
        samples = samples[::20]
        for i in range(components):
            estimator_mean = 0
            estimator_cov = 0
            for x in samples:
                estimator_mean += process.dual_component(i, x)

            estimator_mean *= R / len(samples)

            for x in samples:
                estimator_cov += (process.dual_component(i, x) - estimator_mean) * (process.dual_component(i, x) - estimator_mean)

            estimator_cov *= R * R / (len(samples) - 1)

            raw_moments[i] = [estimator_mean, np.sqrt(estimator_cov)]

        return raw_moments

def raw_samples(process, acceptance, R, components, chunks, chunk_size):
        skip_index = int(chunks * chunk_size * 0.05)

        sampler = Sampler(process.dim())
        log_acceptance = lambda x: np.log(acceptance(x))
        log_target = lambda x: process.log_pdf(x) + log_acceptance(x)
        indicator = pypmc.tools.indicator.hyperrectangle([-1], [1])
        samples = sampler.draw(log_target, start=process.start(), indicator=indicator, chunks=chunks, chunk_size=chunk_size)[skip_index:]
        samples = samples[::120]

        return samples


# Use cases
#   I. determine unfolding matrix:
#      a. flat acceptance "B->Kll"
def flat_acceptance_btokll_matrix():
    print("Flat acceptance in B->Kll")
    acceptance = lambda x: 1.
    samples = []
    for i in range(20):
        chunks=100
        chunk_size=750
        M = unfolding_matrix("B->Kll", acceptance, components=3, chunks=chunks, chunk_size=chunk_size, ovalue=0.5)
        output = []
        for n in M.flatten():
            output.append("%4.4f" % n)
        print "{ %s }, (* samples=%d *)" % (", ".join(output), chunks * chunk_size)

#      b. generic acceptance "B->Kll"
def generic_acceptance_btokll_matrix():
    acceptance_coeffs = np.array([0.47, 0.00, -0.27])
    components = 3 + len(acceptance_coeffs) - 1
    print("Generic acceptance Legendre([%s]) in B->Kll, with unfolding matrix %d x %d" % (','.join(acceptance_coeffs.astype(str)), components, components))
    acceptance = lambda x: Legendre(acceptance_coeffs)(x[0])
    samples = []
    for i in range(1):
        chunks=2000
        chunk_size=500
        M = unfolding_matrix("B->Kll", acceptance, components=components, chunks=chunks, chunk_size=chunk_size, ovalue=0.5)
        output = []
        for n in M.flatten():
            output.append("%4.4f" % n)
        print "{ %s }, (* samples=%d *)" % (", ".join(output), chunks * chunk_size)

#      c. flat acceptance "Lambda_b->Lambdall"
def flat_acceptance_lambdabtolambdall_matrix():
    print("Flat acceptance in Lambda_b->Lambdall")
    acceptance = lambda x: 1.
    samples = []
    for i in range(1):
        M = unfolding_matrix("Lambda_b->Lambdall", acceptance, components=10, chunks=30, chunk_size=500, ovalue=1/(8.0 * np.pi))
        output = []
        for n in M.flatten():
            output.append("%4.4f" % n)
        print "{ " + ", ".join(output) + " },"

#   II. determine corrected moments from MC estimators
#      a. flat acceptance "B->Kll"
def flat_acceptance_btokll_moments():
    signal_obs = np.array([0.50, 0.01, 0.02])
    signal_process = BToKDilepton(signal_obs)
    signal_raw = []
    for i in range(signal_process.number_of_components):
        sp.integrate.quad()

    print("Flat corrected moments in B->Kll[%s]" % ','.join(signal_obs.astype(str)))
    acceptance = lambda x: 1.
    samples = []
    for i in range(10):
        M = unfolding_matrix("B->Kll", acceptance, components=3, chunks=100, chunk_size=500, ovalue=0.5)

        output = []
        for n in M.flatten():
            output.append("%4.4f" % n)
        print "{ " + ", ".join(output) + " },"

#      b. generic acceptance "B->Kll"
def generic_acceptance_btokll_moments():
    signal_obs = np.array([0.50, 0.10, 0.20])
    signal_process = BToKDilepton(signal_obs)
    signal_raw = []
    acceptance_coeffs = np.array([0.47, 0.00, -0.27])
    acceptance = lambda x: Legendre(acceptance_coeffs)(x[0])
    R = 0.46666667
    moments = raw_moments(signal_process, acceptance, R, components=5, chunks=100, chunk_size=500)
    output = []
    for n in moments:
        output.append("{ %4.4f, %4.4f }" % (n[0], n[1]))
    print "{ " + ", ".join(output) + " },"

#   III. generate samples
#      a. flat acceptance "B->Kll"
def flat_acceptance_btokll_samples():
    signal_obs = np.array([0.50, 0.10, 0.20])
    signal_process = BToKDilepton(signal_obs)
    signal_raw = []
    acceptance = lambda x: 1.
    R = 0.5
    print "samplesBToKDileptonFlat = {"
    N = 4000
    for i in range(N):
        samples = raw_samples(signal_process, acceptance, R, components=5, chunks=50, chunk_size=500)
        output = []
        for n in samples:
            output.append("%4.4f" % n)
        komma = ","
        if i == N - 1:
            komma = ""
        print "{ " + ", ".join(output) + " }%s" % komma
    print "};"

#      b. generic acceptance "B->Kll"
def generic_acceptance_btokll_samples():
    signal_obs = np.array([0.50, 0.10, 0.20])
    signal_process = BToKDilepton(signal_obs)
    signal_raw = []
    acceptance_coeffs = np.array([0.47, 0.00, -0.27])
    acceptance = lambda x: Legendre(acceptance_coeffs)(x[0])
    R = 0.445333
    print "samplesBToKDileptonGeneric = {"
    N = 4000
    for i in range(N):
        samples = raw_samples(signal_process, acceptance, R, components=5, chunks=50, chunk_size=500)
        output = []
        for n in samples:
            output.append("%4.4f" % n)
        komma = ","
        if i == N - 1:
            komma = ""
        print "{ " + ", ".join(output) + " }%s" % komma
    print "};"


commands = {
        "unfolding-flat-btokll":              flat_acceptance_btokll_matrix,
        "unfolding-generic-btokll":           generic_acceptance_btokll_matrix,
        "unfolding-flat-lambdabtolambdall":   flat_acceptance_lambdabtolambdall_matrix,
        "moments-flat-btokll":                flat_acceptance_btokll_moments,
        "moments-generic-btokll":             generic_acceptance_btokll_moments,
        "samples-generic-btokll":             generic_acceptance_btokll_samples
}
if __name__ == '__main__':
    if not len(sys.argv) > 1:
        print("Need to specify a command!")
        sys.exit(-1)
    try:
        f = commands[sys.argv[1]]
        f()
    except KeyError:
        print("Unknown command '%s'" % sys.argv[1])
