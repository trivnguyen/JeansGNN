
# import numpyro
# import numpyro.distributions as dist
# from numpyro.infer import SVI, Predictive, Trace_ELBO, TraceGraph_ELBO, RenyiELBO, autoguide
# from numpyro.infer.initialization import init_to_median, init_to_uniform
# from numpyro.infer.reparam import NeuTraReparam
# from numpyro.infer import MCMC, NUTS
# from numpyro import optim
# from jax.example_libraries import stax
# import optax

# class SVIModule():

#     def __init__(self, parameters):
#         super().__init__(parameters=parameters)
#         """

#         Parameters
#         ----------
#         parameters : dict
#             bilby.Likelihood requires a dict of parameters
#         """
#         self.result = None
#         self.priors = {}

#     def log_likelihood(self):
#         raise NotImplementedError

#     def run_sampler(self, *args, **kargs):
#         self.result = bilby.run_sampler(
#             likelihood=self, priors=self.priors, *args, **kargs)

#     def get_credible_intervals(self, key, p=0.95):
#         """ Return the credible intervals of the posterior of a given key """
#         lo = (1 - p) / 2 * 100
#         hi = (1 + p) / 2 * 100
#         values = self.result.posterior[key].values
#         return np.percentile(values, q=[lo, hi])

#     def get_median(self, key):
#         """ Return the median of the posterior of a given key """
#         values = self.result.posterior[key].values
#         return np.percentile(values, q=50)

#     def get_mean_and_std(self, key):
#         """ Return the mean and standard deviation of the posterior of a given key """
#         values = self.result.posterior[key].values
#         return np.mean(values), np.std(values)

#     def load_result(self, *args, **kargs):
#         self.result = bilby.core.result.read_in_result(*args, **kargs)




# def model():
#     logL = numpyro.sample("logL", dist.Uniform(-5, 5))
#     logr_star = numpyro.sample("logr_star", dist.Uniform(-2, 2))
#     loglike = log_likelihood(logL, logr_star)
#     return numpyro.factor('log_likelihood', loglike)

# rng_key=jax.random.PRNGKey(1)

# guide = autoguide.AutoIAFNormal(model, num_flows=4, hidden_dims=[64,64], nonlinearity=stax.Elu)
# optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.), optax.adam(3e-4)))

# svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=16))
# svi_results = svi.run(rng_key, 50_000)
