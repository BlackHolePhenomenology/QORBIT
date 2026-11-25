import numpy as np
from numba import njit

from values.constants import M_sun,c_val,G_val
from values.binary_data import name_binary,M_binary,M_err_binary,f_U_val_binary,f_U_err_binary,f_L_val_binary,f_L_err_binary

from analysis.qpo_analysis import max_L_horizon
from analysis.qpo_analysis import f_U_err_direct
from analysis.qpo_analysis import f_L_err_direct
from analysis.qpo_analysis import ISCO_finder
from analysis.qpo_analysis import get_distance
from analysis.qpo_analysis import ISCO_refinder
from analysis.qpo_analysis import optimize_L



#---------------------------------------
#determin values used for MCMC analysis
#---------------------------------------

def get_L_BH(f):
    r"""
    Determine the global upper bound for the metric parameter :math:`L`
    for the all black hole condition.

    For each binary system, the maximum value of :math:`L` for which the metric
    still possesses an event horizon is determined.

    :param f:
        Numerical metric function ``f(r, L, M)``.
    :type f: callable
    :return:
        Global upper bound of the regulator parameter :math:`L`.
    :rtype: array
    :raises ValueError:
        If no valid upper bound for :math:`L` can be determined.
    """
    L_BH = []
    for i in range(len(M_binary)):
        M_i = M_binary[i]*M_sun
        L_BH.append(max_L_horizon(M_i,f))
            
    if L_BH == []:
        raise ValueError("No valid upper limits for L found for the binary systems")
        
    return L_BH

def get_L_best(f_U_func,f_L_func,V_2diff,f,f_diff):
    r"""
    Determine a preliminary best-fit value for the global metric parameter :math:`\ell`.

    The function performs a grid-based fit of the global regulator parameter
    :math:`\ell`. The physically allowed range is limited by the smallest value of
    :math:`\ell_{BH}` for which all considered binary systems still possess a horizon.
    The combined distance between the modeled and observed QPO frequencies is
    evaluated on a grid, and the value with the smallest distance is returned.

    :param f_U_func:
        Numerical function for the upper QPO frequency.
    :type f_U_func: callable
    :param f_L_func:
        Numerical function for the lower QPO frequency.
    :type f_L_func: callable
    :param V_2diff:
        Numerical function representing the second derivative of the
        effective potential.
    :type V_2diff: callable
    :param f:
        Numerical metric function ``f(r, L, M)``.
    :type f: callable
    :param f_diff:
        Numerical function representing the first derivative
        ``f'(r, L, M)`` of the metric function.
    :type f_diff: callable
    :return:
        Preliminary best-fit value of the global regulator parameter :math:`\ell`.
    :rtype: float
    """
    L_best = []
    L_max = get_L_BH(f)
    for i in range(len(M_binary)):
        L_temp = optimize_L(i,f_U_func,f_U_err_direct,f_L_func,f_L_err_direct,V_2diff,f,f_diff)
        if L_temp < 1:
            L_temp = 0.01*L_max[i]
        L_best.append(min(L_temp,0.99*L_max[i]))
    return L_best

def get_radii(f_U_func,f_L_func,V_2diff,f,f_diff):
    r"""
    Determine approximate best-fit orbital radii for all binary systems.

    The function determines source-dependent orbital radii for a preliminary
    best-fit value of the global regulator parameter :math:`\ell`. For each binary
    system, the event horizon and ISCO radius are calculated. The orbital
    radius is then varied on a radial grid outside the ISCO, and the radius
    minimizing the error-weighted distance between the modeled and observed
    QPO frequencies is selected.

    :param f_U_func:
        Numerical function for the upper QPO frequency.
    :type f_U_func: callable
    :param f_L_func:
        Numerical function for the lower QPO frequency.
    :type f_L_func: callable
    :param V_2diff:
        Numerical function representing the second derivative of the
        effective potential.
    :type V_2diff: callable
    :param f:
        Numerical metric function ``f(r, L, M)``.
    :type f: callable
    :param f_diff:
        Numerical function representing the first derivative
        ``f'(r, L, M)`` of the metric function.
    :type f_diff: callable
    :return:
        Best-fit orbital radii and corresponding ISCO radii for all
        binary systems.
    :rtype: tuple
    """
    r_best = []
    ISCOs = []

    L_best = get_L_best(f_U_func,f_L_func,V_2diff,f,f_diff)

    for i in range(len(M_binary)):
        M_i = M_binary[i]*M_sun
        M_i_err = M_err_binary[i]*M_sun
        
        ISCOs.append(ISCO_finder(M_i,1e-4,V_2diff,f,f_diff)[-1])

        r_0 = np.linspace(ISCOs[-1]*(1+1e-4), 10*ISCOs[-1], 1000)

        f_U_err_dir = f_U_err_direct(M_i,M_i_err,L_best[i],r_0,f_U_func)
        f_L_err_dir = f_L_err_direct(M_i,M_i_err,L_best[i],r_0,f_L_func)

        dist,index_pos = get_distance(f_U_val_binary[i], f_U_err_binary[i], f_L_val_binary[i], f_L_err_binary[i], f_U_func(r_0, L_best[i],M_binary[i]*M_sun), f_U_err_dir, f_L_func(r_0, L_best[i],M_binary[i]*M_sun), f_L_err_dir)
        pos = r_0[index_pos]
        r_best.append(pos)

    return r_best, ISCOs


#---------------------------------------
#define the distributions for MCMC
#---------------------------------------

@njit
def log_prior(theta, V_2diff, L_max, ISCOs):
    r"""
    Evaluate the log-prior probability for the model parameters.

    A hard uniform prior is imposed on the regulator parameter and the orbital
    radii. The regulator parameter must satisfy :math:`0<\ell<\ell_{BH}` and each
    orbital radius must lie outside the corresponding ISCO calculated for the
    current value of :math:`\ell`. Parameter combinations for which no finite ISCO
    exists are rejected.

    :param theta:
        Parameter vector ``(L, r_1, ..., r_N)``, containing the global
        regulator parameter and one orbital radius for each binary system.
    :type theta: numpy.ndarray
    :param V_2diff:
        Numerical function representing the second derivative of the
        effective potential.
    :type V_2diff: callable
    :param L_max:
        Upper bound of the regulator parameter.
    :type L_max: float
    :param ISCOs:
        Reference ISCO radii used as starting values for ``ISCO_refinder``.
    :type ISCOs: numpy.ndarray
    :return:
        Log-prior probability. Returns ``0.0`` inside the allowed parameter
        region and ``-numpy.inf`` outside it.
    :rtype: float
    """
    L_vals = theta[0:len(M_binary)]
    r_vals = theta[len(M_binary):]

    for i in range(len(M_binary)):
        L_i = L_vals[i]
        r_i = r_vals[i]

        if not (0 < L_i < L_max[i]):
            return -np.inf

        r_ISCO = ISCO_refinder(M_binary[i]*M_sun, L_i, V_2diff, ISCOs[i])

        if not np.isfinite(r_ISCO):
            return -np.inf

        if not (r_ISCO < r_i < 3*ISCOs[i]):
            return -np.inf

    return 0.0

@njit
def log_likelihood(theta, func_U, func_U_err, func_L, func_L_err):
    r"""
    Evaluate the log-likelihood for the QPO frequency fit.

    The function calculates the upper and lower model QPO frequencies for
    each binary system and compares them with the corresponding observed
    frequencies. Independent Gaussian uncertainties are assumed for the
    upper and lower QPO frequencies.

    The total variance of each frequency is obtained by adding the model
    uncertainty due to the mass uncertainty and the observational uncertainty
    in quadrature. Parameter combinations producing non-finite model
    frequencies are rejected.

    :param theta:
        Parameter vector ``(L, r_1, ..., r_N)``, containing the global
        regulator parameter and one orbital radius for each binary system.
    :type theta: numpy.ndarray
    :param func_U:
        Numerical function for the upper QPO frequency.
    :type func_U: callable
    :param func_U_err:
        Function for the uncertainty of the upper QPO frequency due to the
        mass uncertainty.
    :type func_U_err: callable
    :param func_L:
        Numerical function for the lower QPO frequency.
    :type func_L: callable
    :param func_L_err:
        Function for the uncertainty of the lower QPO frequency due to the
        mass uncertainty.
    :type func_L_err: callable
    :return:
        Total log-likelihood of the model parameters. Returns ``-numpy.inf``
        if a non-finite model frequency is encountered.
    :rtype: float
    """
    L_vals = theta[0:len(M_binary)]
    r_vals = theta[len(M_binary):]

    loglike = 0.0

    for i in range(len(M_binary)):
        r_i = r_vals[i]
        L_i = L_vals[i]
        M_i = M_binary[i]*M_sun
        M_err_i = M_err_binary[i]*M_sun

        f_U_model = func_U(r_i, L_i, M_i)
        f_L_model = func_L(r_i, L_i, M_i)

        if not np.isfinite(f_U_model) or not np.isfinite(f_L_model):
            return -np.inf

        delta_f_U = f_U_val_binary[i] - f_U_model
        delta_f_L = f_L_val_binary[i] - f_L_model

        sigma_U2 = func_U_err(M_i,M_err_i,L_i,r_i,func_U)**2+f_U_err_binary[i]**2
        sigma_L2 = func_L_err(M_i,M_err_i,L_i,r_i,func_L)**2+f_L_err_binary[i]**2

        loglike += -0.5 * (delta_f_U**2 / sigma_U2 + delta_f_L**2 / sigma_L2 + np.log(sigma_U2) + np.log(sigma_L2))

    return loglike

@njit
def log_posterior_probability(theta, func_U, func_U_err, func_L, func_L_err, V_2diff, L_max, ISCOs):
    r"""
    Evaluate the logarithmic posterior probability for the QPO frequency fit.

    The function combines the log-prior probability and the log-likelihood
    and is passed to the MCMC sampler. If either the prior or the likelihood
    is non-finite, ``-numpy.inf`` is returned and the proposed parameter
    vector is rejected.

    :param theta:
        Parameter vector ``(L, r_1, ..., r_N)``, containing the global
        regulator parameter and one orbital radius for each binary system.
    :type theta: numpy.ndarray
    :param func_U:
        Numerical function for the upper QPO frequency.
    :type func_U: callable
    :param func_U_err:
        Function for the uncertainty of the upper QPO frequency due to the
        mass uncertainty.
    :type func_U_err: callable
    :param func_L:
        Numerical function for the lower QPO frequency.
    :type func_L: callable
    :param func_L_err:
        Function for the uncertainty of the lower QPO frequency due to the
        mass uncertainty.
    :type func_L_err: callable
    :param V_2diff:
        Numerical function representing the second derivative of the
        effective potential.
    :type V_2diff: callable
    :param L_max:
        Upper bound of the regulator parameter.
    :type L_max: float
    :param ISCOs:
        Reference ISCO radii used as starting values for ``ISCO_refinder``.
    :type ISCOs: numpy.ndarray
    :return:
        Logarithmic posterior probability. Returns ``-numpy.inf`` if the
        prior or likelihood is non-finite.
    :rtype: float
    """
    lp = log_prior(theta, V_2diff, L_max, ISCOs)

    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, func_U, func_U_err, func_L, func_L_err)

    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

def prepare_mcmc_samples(sampler,burnin_factor=5.0,thin_factor=0.5):
    r"""
    Analyse an emcee chain and extract cleaned flat posterior samples.

    The integrated autocorrelation time is estimated for all sampled
    parameters. The burn-in is determined from the largest autocorrelation
    time, while the thinning interval is based on the smallest
    autocorrelation time. The remaining chain is flattened and returned.

    :param sampler:
        ``emcee`` sampler containing the MCMC chain.
    :type sampler: emcee.EnsembleSampler
    :param burnin_factor:
        Multiplicative factor applied to the maximum autocorrelation time
        to determine the number of discarded burn-in steps.
    :type burnin_factor: float
    :param thin_factor:
        Multiplicative factor applied to the minimum autocorrelation time
        to determine the thinning interval.
    :type thin_factor: float
    :return:
        Flattened posterior samples after burn-in removal and thinning.
    :rtype: numpy.ndarray
    """
    tau = sampler.get_autocorr_time()

    tau_max = float(np.max(tau))
    tau_min = float(np.min(tau))

    discard = int(np.ceil(burnin_factor * tau_max))

    thin = int(np.ceil(thin_factor * tau_min))

    flat_samples = sampler.get_chain(discard=discard,thin=thin,flat=True)
 
    return flat_samples


#-------------------------------------------
#determin values used for dynesty analysis
#-------------------------------------------
def ISCOs_dynesty(V_2diff,f,f_diff):
    r"""
    Calculate reference ISCO radii for all binary systems.

    The ISCO radius is evaluated at :math:`\ell=0` for each binary system and
    stored as a reference value for the Dynesty analysis.

    :param V_2diff:
        Numerical function representing the second derivative of the
        effective potential.
    :type V_2diff: callable
    :param f:
        Numerical metric function ``f(r, L, M)``.
    :type f: callable
    :param f_diff:
        Numerical function representing the first derivative
        ``f'(r, L, M)`` of the metric function.
    :type f_diff: callable
    :return:
        Reference ISCO radii at :math:`\ell=0` for all binary systems.
    :rtype: list
    """
    ISCOs_dynesty = []
    for i in range(len(M_binary)):
        ISCOs_dynesty.append(ISCO_finder(M_binary[i]*M_sun,0,V_2diff,f,f_diff)[-1])
    return ISCOs_dynesty



#---------------------------------------
#define the distributions for dynesty
#---------------------------------------

@njit
def ptform(u, V_2diff, L_max, ISCOs):
    r"""
    Transform Dynesty unit-cube samples into the physical parameter space.

    The regulator parameter :math:`\ell` is mapped uniformly from :math:`[0, 1]` to
    :math:`[0, L_{BH}]`. For each binary system, the corresponding ISCO is
    recalculated for the sampled value of :math:`\ell` and the orbital radius is
    mapped uniformly to the interval ``[r_ISCO, 3 * ISCOs]``.

    :param u:
        Sample from the Dynesty unit cube with components in ``[0, 1]``.
    :type u: numpy.ndarray
    :param V_2diff:
        Numerical function representing the second derivative of the
        effective potential.
    :type V_2diff: callable
    :param L_max:
        Upper bound of the regulator parameter :math:`\ell`.
    :type L_max: float
    :param ISCOs:
        Reference ISCO radii used as starting values for ``ISCO_refinder``.
    :type ISCOs: array-like
    :return:
        Transformed parameter vector ``(L, r_1, ..., r_N)`` in the physical
        parameter space.
    :rtype: numpy.ndarray
    """
    theta = np.empty(2*len(M_binary),dtype=np.float64)

    for i in range(len(M_binary)):
        M_i = M_binary[i]*M_sun

        L = 0 + u[i] * (L_max[i] - 0)

        theta[i] = L

        r_ISCO = ISCO_refinder(M_i, L, V_2diff, ISCOs[i])

        theta[len(M_binary) + i] = (r_ISCO + u[len(M_binary) + i] * (3*ISCOs[i]-r_ISCO))

    return theta

@njit
def log_likelihood_dynesty(theta, func_U, func_U_err, func_L, func_L_err):
    r"""
    Evaluate the log-likelihood for the QPO frequency fit.

    For each binary system, the modeled upper and lower QPO frequencies are
    compared with the corresponding observed frequencies. Independent
    Gaussian uncertainties are assumed for both frequencies. The total
    variance is obtained by adding the model uncertainty due to the mass
    uncertainty and the observational uncertainty in quadrature.

    The full Gaussian normalization is included in the log-likelihood.
    Parameter combinations producing non-finite model frequencies are
    rejected.

    :param theta:
        Parameter vector :math:`(\ell, r_1, ..., r_N)`, containing the global
        regulator parameter and one orbital radius for each binary system.
    :type theta: numpy.ndarray
    :param func_U:
        Numerical function for the upper QPO frequency.
    :type func_U: callable
    :param func_U_err:
        Function for the uncertainty of the upper QPO frequency due to the
        mass uncertainty.
    :type func_U_err: callable
    :param func_L:
        Numerical function for the lower QPO frequency.
    :type func_L: callable
    :param func_L_err:
        Function for the uncertainty of the lower QPO frequency due to the
        mass uncertainty.
    :type func_L_err: callable
    :return:
        Total log-likelihood of the model parameters. Returns ``-numpy.inf``
        if a non-finite model frequency is encountered.
    :rtype: float
    """
    L_vals = theta[0:len(M_binary)]
    r_vals = theta[len(M_binary):]

    loglike = 0.0

    log_2pi = np.log(2.0 * np.pi)

    for i in range(len(M_binary)):
        r_i = r_vals[i]
        L_i = L_vals[i]
        M_i = M_binary[i]*M_sun
        M_i_err = M_err_binary[i]*M_sun

        f_U_model = func_U(r_i, L_i, M_i)
        f_L_model = func_L(r_i, L_i, M_i)

        if not np.isfinite(f_U_model) or not np.isfinite(f_L_model):
            return -np.inf

        delta_f_U = f_U_val_binary[i] - f_U_model
        delta_f_L = f_L_val_binary[i] - f_L_model

        sigma_U2 = f_U_err_binary[i]**2+func_U_err(M_i,M_i_err,L_i,r_i,func_U)**2
        sigma_L2 = f_L_err_binary[i]**2+func_L_err(M_i,M_i_err,L_i,r_i,func_L)**2

        loglike += -0.5 * (delta_f_U**2 / sigma_U2 + delta_f_L**2 / sigma_L2 + log_2pi + np.log(sigma_U2) + log_2pi + np.log(sigma_L2))

    return loglike