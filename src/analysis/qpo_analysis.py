import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from scipy.optimize import brentq
from numba import njit
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from values.constants import M_sun,c_val,G_val
from values.binary_data import name_binary,M_binary,M_err_binary,f_U_val_binary,f_U_err_binary,f_L_val_binary,f_L_err_binary

def lambdify_numba(args, expression):
    r"""
    Convert a SymPy expression into a NumPy-compatible numerical function
    and compile it with Numba.

    The symbolic expression is first converted into a numerical function
    using :func:`sympy.lambdify` with NumPy as the numerical backend.
    Common subexpression elimination (CSE) is enabled to reduce redundant
    computations. The resulting function is subsequently compiled with
    :func:`numba.njit`.

    :param args:
        Variables or sequence of variables that are used as arguments of
        the generated numerical function.
    :type args: sympy.Symbol or sequence of sympy.Symbol

    :param expression:
        SymPy expression to be converted into a numerical function.
    :type expression: sympy.Expr

    :return:
        Numba-compiled numerical function corresponding to ``expression``.
    :rtype: numba.core.registry.CPUDispatcher
    """

    func = sp.lambdify(args, expression, modules="numpy", cse=True, docstring_limit=0)

    return njit(func, cache=False, fastmath=False)

def calc_freq(metric):
    r"""
    Calculate QPO frequency functions for a given spacetime metric.

    The required metric functions and their derivatives are evaluated
    symbolically with SymPy and converted into Numba-compiled numerical
    functions.

    :param metric:
        Metric definition containing the metric function ``f(r)`` and the
        areal-radius function ``R(r)`` as its first two elements.
    :type metric: sequence
    :return:
        Numerical functions for the upper and lower QPO frequencies,
        their mass uncertainties, the second derivative of the effective
        potential, the metric function, and its first derivative.
    :rtype: tuple
    """

    #use the variables as symbols
    M = sp.symbols("M")
    M_err = sp.symbols("M_err")
    r = sp.symbols("r")
    L = sp.symbols("L")
    G = sp.symbols("G")
    c = sp.symbols("c")

    #symbolic calculation of the differentials
    f = metric[0]  #get f(r) for the chosen metric
    f_diff = sp.diff(metric[0],r,1).simplify()  #first derivative of f(r)
    f_2diff = sp.diff(metric[0],r,2).simplify()   #second derivative of f(r)

    R = metric[1]  #get R(r) for the chosen metric
    R_diff = sp.diff(metric[1],r,1).simplify()    #first derivative of R(r)
    R_2diff = sp.diff(metric[1],r,2).simplify()   #second derivative of R(r)

    J_sq = sp.simplify(R**3*f_diff/(2*R_diff*f-R*f_diff))    #define the square of the angular momentum

    E_sq = 2*f**2*R_diff/(2*R_diff*f - R*f_diff)    #define the square of energy

    J2 = sp.symbols("J2")   #treat J as a symbol for the derivatives of the potential energy
    V_J = (1+J2/R**2)*f     #define the potential enrgy
    V_diff_J = sp.diff(V_J,r,1).simplify()     #first derivative of V(r)
    V_2diff_J = sp.diff(V_J,r,2).simplify()    #second derivative of V(r)


    #resubstitute J in the potential enregy and the derivatives
    V = V_J.subs(J2,J_sq).simplify()
    V_diff = V_diff_J.subs(J2,J_sq).simplify()
    V_2diff = V_2diff_J.subs(J2,J_sq).simplify()

    #symbolic calculation of the frequencies
    f_U = sp.simplify(c/(2*sp.pi)*sp.sqrt(f_diff/(2*R*R_diff)))
    f_L = sp.simplify(c/(2*sp.pi)*(sp.sqrt(f_diff/(2*R*R_diff)) - sp.sqrt(1/2 * V_2diff)))

    f_U_err = sp.diff(f_U,M,1)*M_err
    f_L_err = sp.diff(f_L,M,1)*M_err

    #substitute the values and redefine the functions as numerical functions
    f_U_func = lambdify_numba((r,L,M), f_U.subs([(G,G_val),(c,c_val)]))
    f_U_err_func = lambdify_numba((r,L,M,M_err), f_U_err.subs([(G,G_val),(c,c_val)]))
    f_L_func = lambdify_numba((r,L,M), f_L.subs([(G,G_val),(c,c_val)]))
    f_L_err_func = lambdify_numba((r,L,M,M_err), f_L_err.subs([(G,G_val),(c,c_val)]))
    V_2diff = lambdify_numba((r,L,M), V_2diff.subs([(G,G_val),(c,c_val)]))
    f = lambdify_numba((r,L,M), f.subs([(G,G_val),(c,c_val)]))
    f_diff = lambdify_numba((r,L,M), f_diff.subs([(G,G_val),(c,c_val)]))

    return f_U_func,f_U_err_func,f_L_func,f_L_err_func,V_2diff,f,f_diff

def calc_J(metric):
    r"""
    Calculate angular momentum and the potential for a given spacetime metric.

    The required metric functions and their derivatives are evaluated
    symbolically with SymPy and converted into Numba-compiled numerical
    functions.

    :param metric:
        Metric definition containing the metric function ``f(r)`` and the
        areal-radius function ``R(r)`` as its first two elements.
    :type metric: sequence
    :return:
        Angular momentum squared and, the second derivative of the effective
        potential dependent on the squared angular momentum.
    :rtype: tuple
    """

    #use the variables as symbols
    M = sp.symbols("M")
    r = sp.symbols("r")
    L = sp.symbols("L")
    G = sp.symbols("G")
    c = sp.symbols("c")

    #symbolic calculation of the differentials
    f = metric[0]  #get f(r) for the chosen metric
    f_diff = sp.diff(metric[0],r,1).simplify()  #first derivative of f(r)

    R = metric[1]  #get R(r) for the chosen metric
    R_diff = sp.diff(metric[1],r,1).simplify()    #first derivative of R(r)

    J_sq = sp.simplify(R**3*f_diff/(2*R_diff*f-R*f_diff))    #define the square of the angular momentum


    J2 = sp.symbols("J2")   #treat J as a symbol for the derivatives of the potential energy
    V_J = (1+J2/R**2)*f     #define the potential enrgy
    V_diff_J = sp.diff(V_J,r,1).simplify()    #second derivative of V(r)

    #substitute the values and redefine the functions as numerical functions
    J_sq = lambdify_numba((r,L,M), J_sq.subs([(G,G_val),(c,c_val)]))
    V_J = lambdify_numba((r,L,M,J2), V_J.subs([(G,G_val),(c,c_val)]))
    V_diff_J = lambdify_numba((r,L,M,J2), V_diff_J.subs([(G,G_val),(c,c_val)]))

    return J_sq,V_J,V_diff_J


def calc_r_s(M):
    r"""
    Calculate the Schwarzschild radius.

    :param M: Mass of the object.
    :type M: float
    :return: Schwarzschild radius.
    :rtype: float
    """
    return (2*G_val*M)/c_val**2

def find_horizon(M,L,f,grid=None):
    r"""
    Find the horizon radii of a given metric function.

    The horizons are determined by searching for roots of the metric function
    ``f`` on a radial grid. If multiple horizons exist, the largest radius,
    corresponding to the last element of the returned list, is the outer
    event horizon.

    :param M: Mass of the object.
    :type M: float
    :param L: Regulator parameter of the metric.
    :type L: float
    :param f: Numerical metric function ``f(r, L, M)``.
    :type f: callable
    :param grid: Optional radial search grid.
    :type grid: numpy.ndarray or None
    :return:
        List of horizon radii. If no horizon exists, ``["none", 0]``
        is returned.
    :rtype: list
    """
    r_horizon = []
    if grid is None:
        r_s = calc_r_s(M)
        grid = np.linspace(1e-4, (1+1e-4)*r_s, 1000)
    for i in range(len(grid)-1):
        f_1 = f(grid[i], L,M)
        f_2 = f(grid[i+1], L,M)

        if f_1 == 0:
            r_horizon.append(grid[i])
        if f_1*f_2<0:
            try:
                r_horizon.append(brentq(lambda x: f(x, L,M), grid[i], grid[i+1]))
            except(ZeroDivisionError):
                print("a")
                continue
            

    #for safety
    if r_horizon == []:
        r_horizon.append("none")
        r_horizon.append(1e-4)

    return r_horizon

def max_L_horizon(M,f):
    r"""
    Determine the maximum regulator parameter for which a horizon exists.

    The function iteratively searches for the transition between parameter
    values for which the metric possesses at least one horizon and values
    for which no horizon exists. The search step is reduced once both sides
    of this transition have been identified.

    :param M: Mass of the object.
    :type M: float
    :param f: Numerical metric function ``f(r, L, M)``.
    :type f: callable
    :return:
        Approximate maximum value of ``L`` for which the metric still
        possesses a horizon.
    :rtype: float
    """
    r_s = calc_r_s(M)
    L=0.5*r_s
    delta_L=0.1*r_s
    higher,lower = False,False
    r_min,r_max = 1e-4, (1+1e-4)*r_s
    for i in range(50):
        r_h = find_horizon(M,L,f,np.linspace(r_min, r_max, 1000))
        if r_h[0] == "none":
            higher = True
            delta_L = -abs(delta_L)
        else:
            lower = True
            if len(r_h)==2:
                r_min,r_max = r_h[0],r_h[1]
            delta_L = abs(delta_L)
        if higher == True & lower == True:
            delta_L = 0.5*delta_L
        L += delta_L
    return L

@njit
def defined_area(M,L,f_diff):
    r"""
    Find radial positions where the first derivative of the metric function vanishes.

    The function searches for roots of ``f'(r, L, M)`` on a radial grid.
    These radii determine boundaries of the region in which the orbital
    frequency is real and therefore physically defined.

    :param M: Mass of the object.
    :type M: float
    :param L: Regulator parameter of the metric.
    :type L: float
    :param f_diff: Numerical function representing the first derivative
        ``f'(r, L, M)`` of the metric function.
    :type f_diff: callable
    :return:
        List of radii where ``f'(r, L, M) = 0``. If no root is found,
        ``[0]`` is returned.
    :rtype: list
    """
    n=1
    while n < 100:

        r_a = (n-1)*5000+1
        r_b = n*5000

        f_diff_1 = f_diff(r_a, L,M)
        f_diff_2 = f_diff(r_b, L,M)
            
        if f_diff_1*f_diff_2<0:
            for _ in range(100):
                r_c = r_a + 0.5 * (r_b - r_a)

                f_diff_3 = f_diff(r_c, L,M)

                if f_diff_3 == 0 or 0.5 * abs(r_b - r_a) <= 1e-3:
                    return r_c

                if f_diff_1*f_diff_3<0:
                    r_b = r_c
                    f_diff_2 = f_diff_3
                else:
                    r_a = r_c
                    f_diff_1 = f_diff_3
            return r_a + 0.5*(r_b-r_a)
        n += 1
    
    return 0

#find r-ISCO
def ISCO_finder(M,L,V_2diff,f,f_diff):
    r"""
    Find radii of marginally stable circular orbits, including the ISCO.

    The function searches for roots of the second derivative of the effective
    potential, ``V''(r, L, M) = 0``, on a radial grid outside ``r_h``.
    Only roots satisfying ``f'(r, L, M) > 0`` are retained. If no such root
    is found, the boundary returned by ``defined_area`` is used instead.

    :param M: Mass of the object.
    :type M: float
    :param L: Regulator parameter of the metric.
    :type L: float
    :param V_2diff: Numerical function representing the second derivative
        of the effective potential.
    :type V_2diff: callable
    :param f: Numerical metric function ``f(r, L, M)``.
        :type f: callable
    :param f_diff: Numerical function representing the first derivative
        ``f'(r, L, M)`` of the metric function.
    :type f_diff: callable
    :return:
        List of radii satisfying the marginal stability condition. If no
        such radius is found, the result of ``defined_area`` is returned.
    :rtype: list
    """
    r_ISCO = []
    r_s = calc_r_s(M)
    r_h = find_horizon(M,L,f)[-1]
    r_grid = np.linspace(r_h, 4*r_s, 1000)
    for i in range(len(r_grid)-1):
        V_2diff_1 = V_2diff(r_grid[i], L,M)
        V_2diff_2 = V_2diff(r_grid[i+1], L,M)

        if V_2diff_1 == 0:
            r_ISCO.append(r_grid[i])
        if V_2diff_1*V_2diff_2<0:
            try:
                r_ISCO.append(brentq(lambda x: V_2diff(x, L,M), r_grid[i], r_grid[i+1]))
            except(ZeroDivisionError,ValueError):
                continue

    r_ISCO = [x for x in r_ISCO if f_diff(x,L,M)>0]

    #for safety
    if r_ISCO == []:
        r_ISCO = ["no ISCO", defined_area(M,L,f_diff)]
    return r_ISCO

#find r-ISCO
@njit
def ISCO_refinder(M,L,V_2diff,old_ISCO):
    r"""
    Recalculate the ISCO radius starting from a previously known ISCO.

    The function searches outward and inward from ``old_ISCO`` for a sign
    change of the second derivative of the effective potential. Once a root
    is bracketed, its position is refined using a bisection method.

    :param M: Mass of the object.
    :type M: float
    :param L: Regulator parameter of the metric.
    :type L: float
    :param V_2diff: Numerical function representing the second derivative
        of the effective potential.
    :type V_2diff: callable
    :param old_ISCO: Previously known ISCO radius used as the starting point
        of the search.
    :type old_ISCO: float
    :return:
        Updated ISCO radius. Returns ``numpy.nan`` if no root is found
        within the search range.
    :rtype: float
    """
    n=1
    r_s = (2*G_val*M)/c_val**2
    while n < 500:
        for sign in [-1,1]:
            r_step = sign*n*r_s/100
            
            r_a = old_ISCO
            r_b = old_ISCO+r_step

            V_2diff_1 = V_2diff(r_a, L,M)
            V_2diff_2 = V_2diff(r_b, L,M)
            
            if V_2diff_1 == 0:
                return old_ISCO
            if V_2diff_1*V_2diff_2<0:
                for _ in range(100):
                    r_c = r_a + 0.5 * (r_b - r_a)

                    V_2diff_3 = V_2diff(r_c, L,M)

                    if V_2diff_3 == 0 or 0.5 * abs(r_b - r_a) <= 1e-3:
                        return r_c

                    if V_2diff_1*V_2diff_3<0:
                        r_b = r_c
                        V_2diff_2 = V_2diff_3
                    else:
                        r_a = r_c
                        V_2diff_1 = V_2diff_3
                return r_a + 0.5*(r_b-r_a)
        n += 1
    
    return np.nan

def f_U_err_direct(M,M_err,L,r_0,f_U_func):
    r"""
    Estimate the uncertainty of the upper QPO frequency due to the mass uncertainty.

    :param M: Mass of the object.
    :type M: float
    :param M_err: Uncertainty of the mass.
    :type M_err: float
    :param L: Regulator parameter of the metric.
    :type L: float
    :param r_0: Orbital radius.
    :type r_0: float
    :param f_U_func: Numerical function for the upper QPO frequency.
    :type f_U_func: callable
    :return: Absolute change in the upper QPO frequency for a mass shift of ``M_err``.
    :rtype: float
    """
    f_err_1 = np.abs(f_U_func(r_0, L,M)-f_U_func(r_0, L,M-M_err))
    f_err_2 = np.abs(f_U_func(r_0, L,M)-f_U_func(r_0, L,M+M_err))
    return np.fmax(f_err_1,f_err_2)

def f_L_err_direct(M,M_err,L,r_0,f_L_func):
    r"""
    Estimate the uncertainty of the lower QPO frequency due to the mass uncertainty.

    :param M: Mass of the object.
    :type M: float
    :param M_err: Uncertainty of the mass.
    :type M_err: float
    :param L: Regulator parameter of the metric.
    :type L: float
    :param r_0: Orbital radius.
    :type r_0: float
    :param f_L_func: Numerical function for the lower QPO frequency.
    :type f_L_func: callable
    :return: Absolute change in the lower QPO frequency for a mass shift of ``M_err``.
    :rtype: float
    """
    f_err_1 = np.abs(f_L_func(r_0, L,M)-f_L_func(r_0, L,M-M_err))
    f_err_2 = np.abs(f_L_func(r_0, L,M)-f_L_func(r_0, L,M+M_err))
    return np.fmax(f_err_1,f_err_2)

def get_distance(f_U, err_U, f_L, err_L, func_U, err_func_U, func_L, err_func_L):
    r"""
    Calculate the minimum normalized squared distance between model and observed QPO frequencies.

    The distance is calculated from the squared residuals of the upper and
    lower QPO frequencies, normalized by the combined model and observational
    uncertainties. The minimum value and the corresponding model index are
    returned.

    :param f_U: Observed upper QPO frequency.
    :type f_U: float
    :param err_U: Uncertainty of the observed upper QPO frequency.
    :type err_U: float
    :param f_L: Observed lower QPO frequency.
    :type f_L: float
    :param err_L: Uncertainty of the observed lower QPO frequency.
    :type err_L: float
    :param func_U: Model values of the upper QPO frequency.
    :type func_U: numpy.ndarray
    :param err_func_U: Uncertainties of the modeled upper QPO frequency.
    :type err_func_U: numpy.ndarray
    :param func_L: Model values of the lower QPO frequency.
    :type func_L: numpy.ndarray
    :param err_func_L: Uncertainties of the modeled lower QPO frequency.
    :type err_func_L: numpy.ndarray
    :return:
        Minimum normalized squared distance and index of the corresponding
        model point.
    :rtype: tuple
    """
    s = min(((func_U-f_U)/np.sqrt(err_func_U**2+err_U**2))**2+((func_L-f_L)/np.sqrt(err_func_L**2+err_L**2))**2)    
    index_min = np.argmin(((func_U-f_U)/np.sqrt(err_func_U**2+err_U**2))**2+((func_L-f_L)/np.sqrt(err_func_L**2+err_L**2))**2)  
    return s,index_min

def optimize_L(binary,func_U,err_func_U,func_L,err_func_L,V_2diff,f,f_diff,L_max = None):
    r"""
    Determine the regulator parameter that best reproduces the observed QPO frequencies.

    Starting from the maximum value of ``L`` for which a horizon exists,
    the regulator parameter is varied iteratively to minimize the normalized
    squared distance between the modeled and observed QPO frequencies. The
    search direction is determined from the upper QPO frequency and the step
    size is reduced when the fit becomes worse.

    :param binary: Index of the considered binary system.
    :type binary: int
    :param func_U: Numerical function for the upper QPO frequency.
    :type func_U: callable
    :param err_func_U: Function for the uncertainty of the upper QPO frequency.
    :type err_func_U: callable
    :param func_L: Numerical function for the lower QPO frequency.
    :type func_L: callable
    :param err_func_L: Function for the uncertainty of the lower QPO frequency.
    :type err_func_L: callable
    :param V_2diff: Numerical function representing the second derivative
        of the effective potential.
    :type V_2diff: callable
    :param f: Numerical metric function ``f(r, L, M)``.
    :type f: callable
    :param f_diff: Numerical function representing the first derivative
        ``f'(r, L, M)`` of the metric function.
    :type f_diff: callable
    :param L_max: 
        Optional maximum value of ``L``. If provided used as upper bound of ``L``
        otherwise the upper bound is the maximum ``L`` for which the binary 
        system has an event horizon.
    :type L_max: float or None
    :return: Approximate best-fit value of the regulator parameter ``L``.
    :rtype: float
    """
    M = M_binary[binary]*M_sun
    M_err = M_err_binary[binary]*M_sun
    f_U_val = f_U_val_binary[binary]
    f_U_err = f_U_err_binary[binary]
    f_L_val = f_L_val_binary[binary]
    f_L_err = f_L_err_binary[binary]

    if L_max is None:
        L_max=max_L_horizon(M,f)
    L = 0.5 *L_max
    delta_L = L-1e-4
    r_ISCO = ISCO_finder(M,L,V_2diff,f,f_diff)[-1]
    r_0 = np.linspace(r_ISCO*(1+1e-4), 10*r_ISCO, 10000)
    s,r_index = get_distance(f_U_val, f_U_err, f_L_val, f_L_err, func_U(r_0, L,M), err_func_U(M,M_err,L,r_0,func_U), func_L(r_0, L,M), err_func_L(M,M_err,L,r_0,func_L))
    L_best = L
    s_best = s
    for i in range(100):
        for sign in [-1,1]:
            L_temp = L + sign * delta_L
            if L_temp < 0 or L_temp > L_max:
                continue
            r_ISCO = ISCO_finder(M,L_temp,V_2diff,f,f_diff)[-1]
            r_0 = np.linspace(r_ISCO+1e-4, 10*r_ISCO, 10000)
            s_new,r_index_new = get_distance(f_U_val, f_U_err, f_L_val, f_L_err, func_U(r_0, L_temp,M), err_func_U(M,M_err,L_temp,r_0,func_U), func_L(r_0, L_temp,M), err_func_L(M,M_err,L_temp,r_0,func_L))
            if s_new < s_best:
                L_best = L_temp
                s_best = s_new
        delta_L = delta_L*(0.5)
        L = L_best
        s = s_best
        if L <0:
            L = 0
        if L > L_max:
            L = L_max
    return L

def optimized_dist(func_U,err_func_U,func_L,err_func_L,V_2diff,f,f_diff):
    r"""
    Calculate the combined optimized distance for multiple binary systems.

    For each binary system, the regulator parameter ``L`` is optimized
    individually. The corresponding ISCO and radial frequency curve are then
    calculated, and the minimum distance between the modeled and observed QPO
    frequencies is added to the total distance.

    :param func_U: Numerical function for the upper QPO frequency.
    :type func_U: callable
    :param err_func_U: Function for the uncertainty of the upper QPO frequency.
    :type err_func_U: callable
    :param func_L: Numerical function for the lower QPO frequency.
    :type func_L: callable
    :param err_func_L: Function for the uncertainty of the lower QPO frequency.
    :type err_func_L: callable
    :param V_2diff: Numerical function representing the second derivative
        of the effective potential.
    :type V_2diff: callable
    :param f: Numerical metric function ``f(r, L, M)``.
    :type f: callable
    :param f_diff: Numerical function representing the first derivative
        ``f'(r, L, M)`` of the metric function.
    :type f_diff: callable
    :return: Combined optimized distance for all binary systems.
    :rtype: float
    """
    dist = 0
    for binary in range(len(M_binary)):
        M = M_binary[binary]*M_sun
        M_err = M_err_binary[binary]*M_sun
        L = optimize_L(binary,func_U,err_func_U,func_L,err_func_L,V_2diff,f,f_diff)
        r_ISCO = ISCO_finder(M,L,V_2diff,f,f_diff)
        r_0 = np.linspace(r_ISCO[-1]*(1+1e-4), 10*r_ISCO[-1], 1000)
        s,r_index = get_distance(f_U_val_binary[binary], f_U_err_binary[binary], f_L_val_binary[binary], f_L_err_binary[binary], 
                                 func_U(r_0, L,M), err_func_U(M,M_err,L,r_0,func_U), 
                                 func_L(r_0, L,M), err_func_L(M,M_err,L,r_0,func_L))
        dist += s**2
    return dist

def global_dist(L_array,func_U,err_func_U,func_L,err_func_L,V_2diff,f,f_diff):
    r"""
    Calculate the combined distance for a global regulator parameter.

    For each value of ``L``, the minimum distance between the modeled and
    observed QPO frequencies is calculated for every binary system and
    combined into a total distance.

    :param L_array: Regulator parameter values to be evaluated.
    :type L_array: array-like
    :param func_U: Numerical function for the upper QPO frequency.
    :type func_U: callable
    :param err_func_U: Function for the uncertainty of the upper QPO frequency.
    :type err_func_U: callable
    :param func_L: Numerical function for the lower QPO frequency.
    :type func_L: callable
    :param err_func_L: Function for the uncertainty of the lower QPO frequency.
    :type err_func_L: callable
    :param f: Numerical metric function ``f(r, L, M)``.
    :type f: callable
    :param V_2diff: Numerical function representing the second derivative
        of the effective potential.
    :type V_2diff: callable
    :param f_diff: Numerical function representing the first derivative
        ``f'(r, L, M)`` of the metric function.
    :type f_diff: callable
    :return: Combined distance for each evaluated value of ``L``.
    :rtype: list
    """
    dist = []
    for i in range(len(L_array)):
        dist_new = 0
        for binary in range(len(M_binary)):
            M = M_binary[binary]*M_sun
            M_err = M_err_binary[binary]*M_sun
            r_ISCO = ISCO_finder(M,L_array[i],V_2diff,f,f_diff)
            r_0 = np.linspace(r_ISCO[-1]*(1+1e-4), 10*r_ISCO[-1], 1000)
            s,r_index = get_distance(f_U_val_binary[binary], f_U_err_binary[binary], f_L_val_binary[binary], f_L_err_binary[binary], 
                                     func_U(r_0, L_array[i],M), err_func_U(M,M_err,L_array[i],r_0,func_U), 
                                     func_L(r_0, L_array[i],M), err_func_L(M,M_err,L_array[i],r_0,func_L))
            dist_new +=s**2
        dist.append(dist_new)

    return dist

def create_plot(metric,binary,L,pltName = None):
    r"""
    Create a QPO frequency plot for a selected metric and binary system.

    The function calculates the upper and lower QPO frequencies as a function
    of the orbital radius and compares the resulting model curve with the
    observed QPO frequencies.

    The plot contains the following elements:

    * The central QPO frequency curve calculated using the measured mass.
    * Two additional frequency curves calculated for ``M - M_err`` and
      ``M + M_err``.
    * A shaded region between these curves illustrating the effect of the
      mass uncertainty on the QPO frequency curve.
    * Error bars along the central frequency curve obtained from first-order
      propagation of the mass uncertainty.
    * The model point with the minimum normalized squared distance to the
      observed QPO frequencies, marked as ``r_best``.
    * The QPO frequencies at the ISCO, marked as ``r_ISCO``.
    * The observed upper and lower QPO frequencies including their
      measurement uncertainties.

    :param metric:
        Metric definition containing the metric function ``f(r)``, the
        areal-radius function ``R(r)``, and the metric label.
    :type metric: sequence
    :param binary: Index of the considered binary system.
        :type binary: int
    :param L:
        Regulator parameter of the metric.
    :type L: float
    :param pltName:
        Optional filename for saving the plot. If provided, the figure is
        saved as a PNG file in the ``Pictures`` directory.
    :type pltName: str or None
    :return: None
    :rtype: None
    """

    plt.rcParams.update({'font.size': 22,'xtick.labelsize': 15,'ytick.labelsize': 15})

    f_U_func,f_U_err_func,f_L_func,f_L_err_func,V_2diff,f,f_diff = calc_freq(metric)

    M = M_binary[binary]*M_sun
    M_err = M_err_binary[binary]*M_sun
    f_U_val = f_U_val_binary[binary]
    f_U_err = f_U_err_binary[binary]
    f_L_val = f_L_val_binary[binary]
    f_L_err = f_L_err_binary[binary]

    r_s = calc_r_s(M)

    r_ISCO = ISCO_finder(M,L,V_2diff,f,f_diff)

    r_ISCO_max = ISCO_finder(M+M_err,L,V_2diff,f,f_diff)

    r_ISCO_min = ISCO_finder(M-M_err,L,V_2diff,f,f_diff)

    r_0 = np.linspace(r_ISCO[-1]*(1+1e-4), 10*r_ISCO[-1], 1000)
    r_0_min = np.linspace(r_ISCO_min[-1]*(1+1e-4), 10*r_ISCO[-1], 1000)
    r_0_max = np.linspace(r_ISCO_max[-1]*(1+1e-4), 10*r_ISCO[-1], 1000)

    f_U_err_dir = f_U_err_direct(M,M_err,L,r_0,f_U_func)
    f_L_err_dir = f_L_err_direct(M,M_err,L,r_0,f_U_func)

    dist,index_pos = get_distance(f_U_val, f_U_err, f_L_val, f_L_err, f_U_func(r_0, L,M), f_U_err_dir, f_L_func(r_0, L,M), f_L_err_dir)
    pos = r_0[index_pos]

    plt.plot(f_U_func(r_0_min, L,M-M_err), f_L_func(r_0_min, L,M-M_err),color="green",alpha=0.5)
    plt.plot(f_U_func(r_0, L,M), f_L_func(r_0, L,M), label=f"$(f_U,f_L)_\mathrm{{{metric[2]}}}$",color="green")
    plt.plot(f_U_func(r_0_max, L,M+M_err), f_L_func(r_0_max, L,M+M_err),color="green",alpha=0.5)

    plt.fill(
        np.concatenate([f_U_func(r_0_min, L,M-M_err), f_U_func(r_0_max, L,M+M_err)[::-1]]),
        np.concatenate([f_L_func(r_0_min, L,M-M_err), f_L_func(r_0_max, L,M+M_err)[::-1]]),
        color="green",
        alpha=0.3
    )

    r_0_err = np.linspace(r_ISCO[-1]*(1+1e-4), 10*r_ISCO[-1], 300)
    r_0_err = r_0_err[1:]

    if metric[2] == "Schwarzschild":
        plt.errorbar(f_U_func(r_0_err, L,M), f_L_func(r_0_err, L,M),xerr=abs(f_U_err_func(r_0_err, L,M, M_err)),yerr=abs(f_L_err_func(r_0_err, L,M, M_err)),ecolor="#7A8FA6",fmt="none")

    plt.plot(f_U_func(pos, L,M), f_L_func(pos, L,M),'rX',label=f"$r_{{best}}$/$r_s$ = {round(pos/r_s,3)}")

    plt.plot(f_U_func(r_ISCO[-1]*(1+1e-4), L,M), f_L_func(r_ISCO[-1]*(1+1e-4), L,M),'ro',label=f"$r_{{ISCO}}$/$r_s$ = {round(r_ISCO[-1]/r_s,3)}")
    plt.errorbar(f_U_val, f_L_val, xerr=f_U_err, yerr=f_L_err, fmt="none",ecolor="black",zorder=20, label=name_binary[binary])

    #plt.gca().set_aspect('equal', adjustable='box')
    plt.gcf().set_size_inches(10, 10)

    plt.xlabel("$f_U$/Hz")
    plt.ylabel("$f_L$/Hz")
    plt.legend(numpoints=1, loc="best")
    plt.grid(linestyle="dashed")

    if pltName != None:
        plt.savefig("../results/"+pltName+".png",dpi=300)

    plt.show()

def find_L_crit(M,V_2diff,f,f_diff):
    r"""
    Determine the critical regulator parameter :math:`\ell_{\mathrm{crit}}`
    for which an ISCO still exists.

    Starting from the maximum regulator parameter compatible with an event
    horizon, the function iteratively searches for the transition between
    values of :math:`\ell` for which an ISCO exists and values for which no
    ISCO is found. The search step is reduced whenever the search direction
    changes.

    :param M:
        Mass of the object.
    :type M: float
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
        Approximate critical value :math:`\ell_{\mathrm{crit}}` below which
        an ISCO exists.
    :rtype: float
    """
    L_max = max_L_horizon(M,f)
    L=L_max
    L_step = L_max/4
    dir = 1
    for _ in range(1000):
        if ISCO_finder(M,L,V_2diff,f,f_diff)[0] == "no ISCO":
            L -= L_step
            if dir == 1:
                L_step = L_step/2
            dir = -1
        else:
            L_sol = L
            L += L_step 
            if dir == -1:
                L_step = L_step/2
            dir = 1
    return L_sol

def V_eff_extrema(M,L,J2,V_diff_J):
    """
    Find radial positions of extrema of the effective potential.

    The function searches for roots of the first derivative of the effective
    potential on a radial grid. These roots correspond to circular-orbit
    candidates for the specified angular momentum.

    :param M:
        Mass of the object.
    :type M: float
    :param L:
        Regulator parameter of the metric.
    :type L: float
    :param J2:
        Squared angular momentum of the particle.
    :type J2: float
    :param V_diff_J:
        Numerical function representing the first derivative of the effective
        potential ``V'(r, L, M, J2)`` at fixed angular momentum.
    :type V_diff_J: callable
    :return:
        List of radii at which the effective potential has an extremum.
    :rtype: list
    """
    r_extrema = []
    r_grid = np.linspace(0.1, 60*M*G_val/c_val**2, 100)
    for i in range(len(r_grid)-1):
        V_diff_J_1 = V_diff_J(r_grid[i], L,M,J2)
        V_diff_J_2 = V_diff_J(r_grid[i+1], L,M,J2)
    
        if V_diff_J_1 == 0:
            r_extrema.append(r_grid[i])
        if V_diff_J_1*V_diff_J_2<0:
            try:
                r_extrema.append(brentq(lambda x: V_diff_J(x, L,M,J2), r_grid[i], r_grid[i+1]))
            except(ZeroDivisionError,ValueError):
                continue
    
    return r_extrema

def stable_orbits_plot(metric,M,max_L=1,example_functions=False,ISCO_line=False,MSCO_line=False):

    plt.rcParams.update({'font.size': 10,'xtick.labelsize': 10,'ytick.labelsize': 10})

    f_U_func,f_U_err_func,f_L_func,f_L_err_func,V_2diff,f,f_diff = calc_freq(metric)
    J_sq,V_J,V_diff_J = calc_J(metric)

    r_s = 2*G_val*M/c_val**2
    L_max = max_L_horizon(M,f)

    l = np.linspace(0, max_L*r_s, 200)
    j2 = np.linspace(r_s**2, 5*r_s**2, 200)
    L, J2 = np.meshgrid(l, j2)

    Z = np.zeros_like(L)
    L_crit = find_L_crit(M,V_2diff,f,f_diff)

    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            r_extrema = V_eff_extrema(M, L[i, j], J2[i, j], V_diff_J)
            if L[i,j]<L_crit:
                if len(r_extrema) == 1:
                    if r_extrema[0] > ISCO_finder(M,L[i,j],V_2diff,f,f_diff)[-1]:
                        Z[i, j] = 2
                    else:
                        Z[i, j] = 1
                else:
                    Z[i, j] = 3
            else:
                Z[i,j] = 4
                
    plt.pcolormesh(J2/r_s**2, L/r_s, Z, shading="auto", cmap="viridis")
    plt.axhline(L_max/r_s, color="red", linewidth=2)
    plt.axhline(L_crit/r_s, color="blue", linewidth=2)

    cmap = plt.get_cmap("viridis")
    values = np.unique(Z)
    norm = plt.Normalize(vmin=Z.min(), vmax=Z.max())

    legend_elements = []
    text = ["no orbits", "1 inner orbit", "1 outer orbit", "1 inner and 2 outer orbits","1 orbit"]
    for i in values:
        legend_elements.append(Patch(facecolor=cmap(norm(i)),label=text[int(i)]))
    legend_elements.append(Line2D([0], [0], color="red", linewidth=2,label=fr"$L_{{\mathrm{{BH}}}}/(2GM)={L_max/r_s:.3f}$"))
    legend_elements.append(Line2D([0], [0], color="blue", linewidth=2,label=fr"$L_{{\mathrm{{crit}}}}/(2GM)={L_crit/r_s:.3f}$"))

    
    J_ISCO = []
    L_values_ISCO = np.linspace(0,L_crit,100)
    for i in range(len(L_values_ISCO)):
        J_ISCO.append(J_sq(ISCO_finder(M,L_values_ISCO[i],V_2diff,f,f_diff)[-1],L_values_ISCO[i],M))

    J_ISCO = np.asarray(J_ISCO)
    if ISCO_line == True:
        plt.plot(J_ISCO/r_s**2,L_values_ISCO/r_s,color="black")

    
    J_MSCO = []
    L_values_MSCO = np.linspace(0.85*L_crit,L_crit,100)
    for i in range(len(L_values_MSCO)):
        J_MSCO.append(J_sq(ISCO_finder(M,L_values_MSCO[i],V_2diff,f,f_diff)[0],L_values_MSCO[i],M))
    
    J_MSCO = np.asarray(J_MSCO)
    if MSCO_line == True:
        plt.plot(J_MSCO/r_s**2,L_values_MSCO/r_s,color="black")
    
        
    plt.legend(handles=legend_elements)
    plt.title(rf"Number of extrema of $V_{{\mathrm{{eff}}}}$ in the metric {metric[-1]}")
    plt.xlabel("$J^2$/$(4G^2M^2)$")
    plt.ylabel("$L$/$(2GM)$")
    plt.xlim(1, 5)
    plt.savefig("../results/circular_orbits.png",dpi=300)
    plt.show()

    if example_functions == True:
        r_val = np.linspace(0.0,30000.0,1000)
        L = 0.9*L_crit
        J_ISCO = J_sq(ISCO_finder(M,L,V_2diff,f,f_diff)[-1],L,M)
        J_MSCO = J_sq(ISCO_finder(M,L,V_2diff,f,f_diff)[0],L,M)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.set_title(rf"$V_{{\mathrm{{eff}}}}$ in the horizonless case")
        ax1.set_ylim(0.6, 1.3)
        ax1.set_xlabel("$r$/m")
        ax1.set_ylabel("$V_{{\mathrm{{eff}}}}$/J")
        ax1.plot(r_val,V_J(r_val,L,M_sun,(J_ISCO+J_MSCO)/2),color=cmap(norm(3)),label=rf"$L/(2GM)={L/r_s:.3f}$, $J/(4G^2M^2)={(J_ISCO+J_MSCO)/2/r_s**2:.3f}$")
        ax1.plot(r_val,V_J(r_val,L,M_sun,0.7*J_ISCO),color=cmap(norm(0)),label=rf"$L/(2GM)={L/r_s:.3f}$, $J/(4G^2M^2)={0.7*J_ISCO/r_s**2:.3f}$")
        ax1.plot(r_val,V_J(r_val,L,M_sun,1*J_MSCO),color=cmap(norm(2)),label=rf"$L/(2GM)={L/r_s:.3f}$, $J/(4G^2M^2)={1*J_MSCO/r_s**2:.3f}$")
        ax1.plot(r_val,V_J(r_val,L_crit*1.3,M_sun,3*r_s**2),color=cmap(norm(4)),label=rf"$L/(2GM)={L_crit*1.3/r_s:.3f}$, $J/(4G^2M^2)={3}$")
        ax1.legend()
        ax2.set_title(rf"$V_{{\mathrm{{eff}}}}$ in the horizon case")
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_xlim(0, 15000)
        ax2.set_xlabel("$r$/m")
        ax2.set_ylabel("$V_{{\\mathrm{{eff}}}}$/J")
        ax2.plot(r_val,V_J(r_val,0.9*L_max,M_sun,1.5*J_ISCO),color=cmap(norm(3)),label=rf"$L/(2GM)={0.9*L_max/r_s:.3f}$, $J/(4G^2M^2)={1.5*J_ISCO/r_s**2:.3f}$")
        ax2.plot(r_val,V_J(r_val,0.9*L_max,M_sun,0.5*J_ISCO),color=cmap(norm(0)),label=rf"$L/(2GM)={0.9*L_max/r_s:.3f}$, $J/(4G^2M^2)={0.5*J_ISCO/r_s**2:.3f}$")
        ax2.legend()
        plt.savefig("../results/example_graphs.png",dpi=300)
        plt.show()