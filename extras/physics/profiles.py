
import numpy as np
import scipy.special as sc
from astropy.cosmology import Planck18


def rho_gNFW(r, rho_s, r_s, gamma):
    """  The density profile rho(r) of a generalized NFW profile

    Parameters
    ----------
    r : float array like shape (N,)
        Radius
    rho_s : float array like shape (M,)
        The value of rho at r = 0
    r_s : float array like shape (M,)
        The scale radius
    gamma : float array like shape (M,)
        The power law index

    Returns
    -------
    rho: float array like shape (N, M)
    """
    # broadcast rho_s, gammma, r_s to the same shape as r
    rho_s = np.broadcast_to(rho_s, r.shape)
    gamma = np.broadcast_to(gamma, r.shape)
    r_s = np.broadcast_to(r_s, r.shape)

    x = r / r_s
    rho =  rho_s * x**(-gamma) * (1 + x)**(-3 + gamma)
    return rho

def beta_OM(r, beta_0, r_a):
    """ The velocity anisotropy profile Beta(r) of the Osipkov-Merritt model

    Parameters
    ----------
    r : float array like shape (N,)
        Radius
    beta_0 : float array like shape (M,)
        The value of Beta at r = 0
    r_a : float array like shape (M,)
        The scale radius of the velocity anisotropy

    Returns
    -------
    beta: float array like shape (N, M)
    """
    # broadcast beta_0, r_a to the same shape as r
    beta_0 = np.broadcast_to(beta_0, r.shape)
    r_a = np.broadcast_to(r_a, r.shape)

    x = r / r_a
    return (beta_0 + x**2) / (1 + x**2)


def menc_gNFW(r, rho_s, r_s, gamma):
    """ Enclosed mass profile for a generalized NFW profile. Formula:
    ```
        M_enclosed (r) = 4 pi rho_s r_s^3 \int_0^{r/r_s} x^{2-gamma} (1 + x)^{-3 + gamma} dx
            = 4 pi rho_s r_s^3 (x^{3 - gamma} / (3 - gamma)) 2F1(3-gamma, 3-gamma, 4-gamma, -x)
    ```
        where 2F1 is the hypergeometric function
    Some references:
    - https://dlmf.nist.gov/8.17#E7

    Parameters
    ----------
    r : float array like shape (N,)
        Radius
    rho_s : float array like shape (M,)
        The value of rho at r = 0
    r_s : float array like shape (M,)
        The scale radius
    gamma : float array like shape (M,)
        The power law index

    Returns
    -------
    M_enc: float array like shape (N, M)

    """
    # broadcast rho_s, gammma, r_s to the same shape as r
    rho_s = np.broadcast_to(rho_s, r.shape)
    gamma = np.broadcast_to(gamma, r.shape)
    r_s = np.broadcast_to(r_s, r.shape)

    x = r / r_s
    return 4 * np.pi * rho_s * r_s**3 * (
        x**(3-gamma) * sc.hyp2f1(3-gamma, 3-gamma, 4-gamma, -x) / (3-gamma))


def rhobar_gNFW(r, rho_s, r_s, gamma):
    """ Mean density profile of the generalized NFW profile. Formula:
    ```
        rho_bar(r) = M_enc(r) / (4 pi r^3 / 3)
    ```
    Parameters
    ----------
    r : float array like shape (N,)
        Radius
    rho_s : float array like shape (M,)
        The value of rho at r = 0
    r_s : float array like shape (M,)
        The scale radius
    gamma : float array like shape (M,)
        The power law index

    Returns
    -------
    rho_bar: float array like shape (N, M)
    """
    return menc_gNFW(r, rho_s, r_s, gamma) / (4 * np.pi * r**3 / 3)


# def R_vir(rho_s, r_s, gamma, vir=200, n_steps=10000):
#     """ Solve for the virial radius. Default to c200, the radius within which the average
#     densityis 200 times the critical density of the Universe at redshift z = 0
#     """
#     # critical density
#     rho_c = Planck18.critical_density(0).to_value(u.Msun / u.kpc**3)

#     # calculate enclosed mass and average density
#     rho_avg = rho_bar(r, rho_s, r_s, gamma)

#     try:
#         return interp.interp1d(rho_avg/rho_c, r)(vir)
#     except:
#         return np.nan


# def M_vir(rho_s, r_s, gamma, vir=200, n_steps=10000):
#     """ Calculate the virial mass. Default to M200 """
#     return M_enc(R_vir(rho_s, r_s, gamma, vir=vir, n_steps=n_steps), rho_s, r_s, gamma)

# def c_vir(rho_s, r_s, gamma, vir=200, n_steps=10000):
#     """ Calculate the virial mass. Default to c200 """
#     return R_vir(rho_s, r_s, gamma, vir=vir, n_steps=n_steps) / r_s


# def J_factor(rho_s, r_s, gamma, dc=1, vir=200, n_steps=10000):
#     """ Calculate the J factor for a generalized NFW profile. Integrate up to c_vir """
#     c = c_vir(rho_s, r_s, gamma, vir, n_steps=n_steps)
#     a = 2 * gamma

#     J = - np.power(c, 3-a) * np.power(c+1, a-5) * (
#         np.power(a, 2) - a*(2*c + 9) + 2*c*(c+5) + 20) / (a-5)*(a-4)*(a-3)
#     J = 4 * np.pi * np.power(rho_s, 2) * np.power(r_s, 3) * J / dc**2

#     # Convert J from Msun^2 / kpc^5 to GeV^2 / cm^5
#     J = J * u.Msun**2 / u.kpc**5
#     J = J.to_value(u.GeV**2 / const.c**4 / u.cm**5)
#     return J


# def log10_plummer2d(R, L, r_star):
#     """ Log 10 of the Plummer 2D profile
#     Args:
#         R: projected radius
#         params: L, a
#     Returns:
#         log I(R) = log {L(1 + R^2 / a^2)^{-2} / (pi * r_star^2)}
#     """
#     logL = np.log10(L)
#     logr_star = np.log10(r_star)
#     return (
#         logL - 2 * logr_star - 2 * np.log10(1 + R**2 / r_star**2)
#         - np.log10(np.pi))


# def log10_plummer3d(r, L, r_star):
#     """ Log 10 of the Plummer 2D profile
#     Args:
#         R: projected radius
#         params: L, r_star
#     Returns:
#         log I(R) = log {L(1 + R^2 / a^2)^{-2} / (pi * r_star^2)}
#     """
#     logL = np.log10(L)
#     logr_star = np.log10(r_star)
#     return (
#         logL - 3 * logr_star - (5/2) * np.log10(1 + r**2 / r_star**2)
#         - np.log10(4 * np.pi / 3)
#     )
