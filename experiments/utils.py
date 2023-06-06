
import astropy.constants as const
import astropy.units as u
import numpy as np
import scipy
import jax.numpy as jnp
from astropy import constants as const

kpc_to_km = (u.kpc).to(u.km)

def jax_cumtrapz(y, x):
    """
    Cumulative trapezoidal integration of y with respect to x.
    """
    dx = jnp.diff(x)
    res = jnp.cumsum((y[1:] + y[:-1]) / 2 * dx)
    return res

def jax_cumtrapz_init(y, x, initial=0):
    """
    Cumulative trapezoidal integration of y with respect to x.
    """
    dx = jnp.diff(x)
    res = jnp.cumsum((y[1:] + y[:-1]) / 2 * dx)
    res = jnp.insert(res, 0, initial)
    return res

def poiss_err(n, alpha=0.32):
    """
    Poisson error (variance) for n counts.
    An excellent review of the relevant statistics can be found in
    the PDF statistics review: http://pdg.lbl.gov/2018/reviews/rpp2018-rev-statistics.pdf,
    specifically section 39.4.2.3

    Parameters
    ----------
    n : array_like
        number of counts
    alpha : float
        central confidence level 1-alpha, i.e. alpha = 0.32 corresponds to 68% confidence

    Returns
    -------
    n_lo : array_like
        lower error on the number of counts
    n_hi : array_like
        upper error on the number of counts

    """
    n_lo = scipy.stats.chi2.ppf(alpha/2,2*n)/2
    n_hi = scipy.stats.chi2.ppf(1-alpha/2,2*(n+1))/2
    return n_lo, n_hi


def calc_projected_nstar(R, alpha=0.32):
    """ Calculate the projected number of stars as a function of 
    projected radius R

    Parameters
    ----------
    R : array_like
        projected radius of stars
    alpha : float
        central confidence level 1-alpha, i.e. alpha = 0.32 corresponds to 
        68% confidence

    Returns
    -------
    n_data : array_like
        number of stars in each bin
    n_data_lo : array_like
        lower bound on the number of stars in each bin
    n_data_hi : array_like
        upper bound on the number of stars in each bin
    logR_bins_lo : array_like
        lower edge of the bins
    logR_bins_hi : array_like
        upper edge of the bins

    """
    logR = jnp.log10(R)
    n_total = len(logR)

    # bin the projected radius using log scale
    n_bins = int(jnp.ceil(jnp.sqrt(n_total)))
    logR_min = jnp.floor(jnp.min(logR)*10) / 10
    logR_max = jnp.ceil(jnp.max(logR)*10) / 10
    n_data, logR_bins = jnp.histogram(logR, n_bins, range=(logR_min, logR_max))

    # remove bins with zero counts
    # ignore bin with zero count
    select = n_data > 0
    n_data = n_data[select]
    logR_bins_lo = logR_bins[:-1][select]
    logR_bins_hi = logR_bins[1:][select]

    # compute the lower and upper bound for each bin
    n_data_lo, n_data_hi = poiss_err(n_data, alpha=0.32)

    return n_data, n_data_lo, n_data_hi, logR_bins_lo, logR_bins_hi


def calc_Sigma(R, alpha=0.32):
    """ Calculate the projected 2d light profile Sigma(R) 
    where R is the projected radius

    Parameters
    ----------
    R : array_like
        projected radius of stars
    alpha : float
        central confidence level 1-alpha, i.e. alpha = 0.32 corresponds to 
        68% confidence

    Returns
    -------
    Sigma_data : array_like
        projected 2d light profile in each bin
    Sigma_data_lo : array_like
        lower bound on the projected 2d light profile in each bin
    Sigma_data_hi : array_like
        upper bound on the projected 2d light profile in each bin
    logR_bins_lo : array_like
        lower edge of the bins
    logR_bins_hi : array_like
        upper edge of the bins
    """
    # projected number of stars
    n_data, n_data_lo, n_data_hi, logR_bins_lo, logR_bins_hi = calc_projected_nstar(
        R, alpha=alpha)

    # calculate light profile from projected number of stars
    R_bins_lo = jnp.power(10, logR_bins_lo)
    R_bins_hi = jnp.power(10, logR_bins_hi)
    delta_R2 = (R_bins_hi**2 - R_bins_lo**2)
    Sigma_data = n_data / (jnp.pi * delta_R2)
    Sigma_data_lo = n_data_lo / (jnp.pi * delta_R2)
    Sigma_data_hi = n_data_hi / (jnp.pi * delta_R2)

    return Sigma_data, Sigma_data_lo, Sigma_data_hi, logR_bins_lo, logR_bins_hi


def calc_sigma2_nu(r, cmass, nu, g):
    """ Calculate the 3D Jeans integration:
    .. math::
    sigma2(r0) nu(r0) g(r_0) =  int_r0^\infty G M(r) nu(r) g(r) / r^2 dr

    where:
    - G is the gravitational constant
    - M(r) is the enclosed radius at radius r in Msun
    - nu(r) is the 3D light profile
    - g(r) is the anistropy integral

    Parameters
    ----------
    r: array of N float
        The 3D radius in kpc
    cmass: array of N float
        The cumulative mass in Msun
    nu: array of N float
        The 3D light profile
    g: array of N float
        The anistropy integral
    """
    inte = cmass * nu * g / r**2
    inte_res = jax_cumtrapz_init(y=inte[::-1], x=r, initial=0)
    return inte_res[::-1] / g

def calc_gint(r, beta):
    """ Calculate the Jeans velocity anisotropy integral g(r) defined as:
    .. math::
        g(r) = exp( 2 \int beta(r) / r dr )

    where:
        beta(r) is the velocity anisotropy
        r is the 3d radius
    """
    g = jnp.exp(jax_cumtrapz(y=2 * beta / r, x=r))
    g = jnp.append(g, g[-1])
    return g


def calc_sigma2p_Sigma(R, r, sigma2_nu, beta):
    """ Calculate the projected Jeans integration:
    .. math::
    sigma2_p(R) Sigma(R) = 2 * int_R^\intfy (1 - beta * R^2 /r^2) (nu(r) sigma2(r) r) / sqrt(r^2 - R^2) dr

    where:
    - R is the projected radius
    - Sigma(R) is the 3D light profile

    Parameters
    ----------
    R: array of M float
        The 2D radius in kpc
    r: array of N float
        The 3D radius in kpc
    sigma2_nu: array of N float
        The 3D velocity dispersion
    beta: array of N float
        The velocity anisotropy

    Returns
    -------
    sigma2p_Sigma: array of M float
        The projected velocity dispersion
    """
    # dr = r[1] - r[0]
    # R = R[:, None]
    # r = r[None, :]
    # sigma2_nu = sigma2_nu[None, :]
    # rminR2 = r**2 - R**2
    # beta = beta[None, :]
    # inte = (1 - beta * R**2 / r**2) * sigma2_nu * r
    # inte = jnp.where(rminR2 > 0,  inte / jnp.sqrt(rminR2), 0)
    # sigma2p_Sigma = 2 * jnp.trapz(inte, x=r, axis=1)
    return sigma2_nu  #sigma2p_Sigma
