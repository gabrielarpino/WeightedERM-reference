import jax
jax.config.update("jax_enable_x64", True) # float64
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jaxopt
from tqdm.auto import tqdm
import numpy as np
import jax.scipy.optimize as jso
import scipy.optimize
from WERM.config import η_to_ψ

def prox_obj_no_idx(t, args): 
    z, b, π_i = args
    return (b * π_i * M_fixed(t) + (1/2) * jnp.linalg.norm(t - z)**2)

def prox(z, b, π_i):
    """ Proximal operator with respect to the loss b π_i M """

    # def prox_obj(t): 
    #     """ Proximal objective of g. ℓ_fixed must be globally defined through the AMP function for this to be well defined. """
    #     return (b * π_i * M_fixed(t) + (1/2) * jnp.linalg.norm(t - z)**2)[0] # Have to index at zero because jaxopt.minimize expects a scalar but passes in an array initializer

    # res = jso.minimize(prox_obj, jnp.array([1.0]), method="BFGS", options={"maxiter": 10000}) # Currently, this optimization is not always successful, even after increasing maxiter.
    # return res.x[0] # was giving nans

    # Solve using jaxopt, so that we can differentiate through the solution

    # gd = jaxopt.GradientDescent(fun=prox_obj_no_idx, maxiter=1e10, tol = 1e-5, stepsize = 1e-1, implicit_diff=True) # Extremely fast, and seems to give just as good results
    gd = jaxopt.GradientDescent(fun=prox_obj_no_idx, maxiter=1e10, tol = 1e-7, stepsize = 1e-1, implicit_diff=True) # Extremely fast, and seems to give just as good results
    args = (z, b, π_i)
    res = gd.run(0.85, args).params
    return res

def S(z, b, π_i):
    return z - prox(z, b, π_i)
S_mapped = jax.vmap(S, in_axes=(0, None, 0))

def prox_prime(z, b, π_i):
    """ Tested, and yields the correct result in linear regression with change points. """
    ℓ_double_prime = jax.grad(jax.grad(M_fixed))
    prox_plug = prox(z, b, π_i)
    return jnp.power(b * π_i * ℓ_double_prime(prox_plug) + 1, -1)
prox_prime_mapped = jax.vmap(prox_prime, in_axes=(0, None, 0))

def S_prime(z, b, π_i):
    return 1 - prox_prime(z, b, π_i)
S_prime_mapped = jax.vmap(S_prime, in_axes=(0, None, 0))

def fp_linear_model_old(M, ρ, L, η, π, δ, n, ε): 
    """ For brevity, μ and σ in the following code refer to μ̃, σ̃ in the paper. """

    global M_fixed
    M_fixed = M

    μ = np.zeros(L)
    τ = np.ones(L) * 0.5
    b = 1.0
    σ_squared = 0.5

    num_iters = 10
    for iter in tqdm(range(num_iters)):

        # Compute μ
        G = np.random.randn(n)
        def μ_obj(l, b): 
            return np.sum(δ/n * S_prime_mapped((ε + τ[l-1] * G)[η[l - 1]: η[l]], b, π[η[l - 1]: η[l]]))
        for l in range(1, L+1): 
            μ[l-1] = μ_obj(l, b)
        μ = np.array(μ)
        print("μ: ", μ)
        
        # Compute σ
        G = np.random.randn(n)
        def σ_obj(l, b):
            return jnp.sum((δ**2 / n) * jnp.power(S_mapped((ε + τ[l-1] * G)[η[l - 1]: η[l]], b, π[η[l - 1]: η[l]]), 2))
        σ_squared = np.sum([σ_obj(l, b) for l in range(1, L+1)])
        print("σ_squared: ", σ_squared)

        # Compute τ
        τ_squared = np.zeros(L)
        for l in range(1, L+1): 
            ind_l = np.zeros(L)
            ind_l[l-1] = 1
            τ_squared[l-1] = σ_squared / δ + (ind_l - μ).T @ ρ @ (ind_l - μ)
        τ = np.sqrt(τ_squared)
        print("τ: ", τ)
        
        # Solve for b
        G = np.random.randn(n)
        def μ_obj(l, b): 
            return np.sum(δ/n * S_prime_mapped((ε + τ[l-1] * G)[η[l - 1]: η[l]], b, π[η[l - 1]: η[l]]))
        def sum_μ_obj(b): 
            return np.sum([μ_obj(l, b) for l in range(1, L+1)])

        print("Running fsolve...")
        # b = scipy.optimize.fsolve(lambda b: sum_μ_obj(b) - 1, 0.5) # Could probably be made faster if we pass in fprime. Yields non convergence, likely due to the fact that the function is defined using samples. 
        b = scipy.optimize.minimize(lambda b: np.power(sum_μ_obj(b) - 1, 2), jnp.array([1.0]), method="BFGS", options={"maxiter": 100}).x[0] # switch to jso instead of scipy for speed
        print("Done fsolve, b = ", b)

    return μ, σ_squared, τ_squared, b

def fp_linear_model_unif_prior(M, ρ, L, η_, δ, n, noise_std):
    """ Has to be generalized for non-gaussian noises, etc. """

    global M_fixed
    M_fixed = M

    num_samples = 2000
    # num_samples = n
    π = (num_samples - jnp.arange(num_samples)) / num_samples
    ε = jnp.array(np.random.randn(num_samples) * noise_std)

    # η = jnp.array([0, *((η_ * num_samples).astype(int)), num_samples])
    if η_.size == 0:
        ψ = jnp.zeros(num_samples).astype(int) 
    else:
        ψ = η_to_ψ((η_ * num_samples).astype(int), num_samples) # Seems to be correct
    G_1 = jnp.array(np.random.randn(num_samples))
    G_2 = jnp.array(np.random.randn(num_samples))

    def μ̃(x, l, args):
        b_ = x[-1]
        τ_arr = x[:-1]
        δ, ρ, ε, π = args
        ε_and_τ = ε + τ_arr[l] * G_1
        return ( (δ / num_samples) * jnp.where(ψ == l, S_prime_mapped(ε_and_τ, b_, π), 0)).sum() # Written this way for differentiability
    μ̃_jit = jax.jit(μ̃)

    def σ̃_l_squared(x, l, args): 
        b_ = x[-1]
        τ_arr = x[:-1]
        δ, ρ, ε, π = args
        ε_and_τ = ε + τ_arr[l] * G_2
        return ((δ**2 / num_samples) * jnp.where(ψ == l, jnp.power(S_mapped(ε_and_τ, b_, π), 2), 0)).sum() # Written this way for differentiability

    def σ̃_squared(x, args):
        return jnp.sum(jnp.array([σ̃_l_squared(x, l, args) for l in range(L)]))
    σ̃_squared_jit = jax.jit(σ̃_squared)

    def τ_squared_l(x, l, args):
        δ, ρ, ε, π = args
        ind_l = np.zeros(L)
        ind_l[l] = 1
        μ̃_arr = jnp.array([μ̃(x, l, args) for l in range(L)])
        return (σ̃_squared(x, args) / δ) + (ind_l - μ̃_arr).T @ ρ @ (ind_l - μ̃_arr)

    def τ_squared_obj(x, args):
        b_ = x[-1]
        τ_arr = x[:-1]
        return jnp.array([jnp.power(jnp.power(τ_arr[l], 2) - τ_squared_l(x, l, args), 2) for l in range(L)]).sum()

    def joint_obj_jaxopt(x, args):
        δ, ρ, ε, π = args
        return τ_squared_obj(x, args) + jnp.power( 1 - jnp.sum(jnp.array([μ̃(x, l, args) for l in range(L)])) , 2)
        # return τ_squared_obj(x, args) + np.power( 1 - np.sum([μ̃(x, l, args) for l in range(L)]) , 2)
    joint_obj_jaxopt_jit = jax.jit(joint_obj_jaxopt)

    def fsolve_obj(x, args): 
        """ TODO: Implement this."""
        b_ = x[-1]
        τ_arr = x[:-1]
        τ_obj_list = [jnp.sqrt(τ_squared_l(x, l, args)) - τ_arr[l] for l in range(L)]
        μ̃_obj = (1 - jnp.sum(jnp.array([μ̃(x, l, args) for l in range(L)])))
        return jnp.array([*τ_obj_list, μ̃_obj])
    fsolve_obj_jit = jax.jit(fsolve_obj)

    # gd = jaxopt.GradientDescent(fun=joint_obj_jaxopt, maxiter=1e10, tol = 1e-2, \
    #                             stepsize = 5e-4).run(jnp.array([*(0.85*np.ones(L)), 0.75]), \
    #                                                  (δ, ρ, ε, π))  # Make step size smaller if getting nans
    # gd = jaxopt.GradientDescent(fun=joint_obj_jaxopt, maxiter=1e10, tol = 1e-2, \
    #                             stepsize = 1e-1).run(jnp.array([*(0.85*np.ones(L)), 0.75]), \
    #                                                  (δ, ρ, ε, π))  # Make step size smaller if getting nans
    # res = gd.params # Indexing problems with jaxopt
    # τ_arr, b = res[0], res[1]

    # Solve using fsolve, faster than minimize with the jacobian
    J_fsolve = jax.jacrev(fsolve_obj_jit, argnums=0)
    res, infodict, ier, mesg  = scipy.optimize.fsolve(fsolve_obj_jit, jnp.array([*(0.85*np.ones(L)), 0.75]), args = ([δ, ρ, ε, π]), fprime = J_fsolve, full_output = True, maxfev = 1000)
    print("Optimization Results: x = ", res, ", ier = ", ier, ", mesg = ", mesg, ", infodict = ", infodict)
    τ_arr, b = res[:-1], res[-1]
    
    # Solve using Minimize
    # J = jax.jit(jax.jacrev(joint_obj_jaxopt, argnums=0))
    # J = jax.grad(joint_obj_jaxopt_jit, argnums=0)
    # opt_res = scipy.optimize.minimize(joint_obj_jaxopt_jit, x0 = np.array([*(0.85*np.ones(L)), 0.75]), args = ([δ, ρ, ε, π]), method="BFGS", options={"maxiter": 100}, jac = J) # switch to jso instead of scipy for speed
    # print("Optimization Results: x = ", opt_res.x, ", fun = ", opt_res.fun, ", success = ", opt_res.success, ", message = ", opt_res.message, ", nfev = ", opt_res.nfev, ", nit = ", opt_res.nit, ", njev = ", opt_res.njev, ", status = ", opt_res.status)
    # res = opt_res.x
    # print("res: ", res)
    # τ_arr, b = res[:-1], res[-1]

    # Compute σ̃
    σ̃ = np.sqrt(σ̃_squared_jit(res, (δ, ρ, ε, π)))
    # σ = σ̃ / (δ * b)

    μ̃_arr = np.array([μ̃_jit(res, l, (δ, ρ, ε, π)) for l in range(L)])
    print("μ̃_arr: ", μ̃_arr)
    
    μ_arr = μ̃_arr / (δ * b)
    σ = σ̃ / (δ * b)

    return μ̃_arr, σ̃, μ_arr, σ

def fp_linear_model_vanilla(M, δ, noise_std): 
    """ Tested, works as expected. """

    global M_fixed
    M_fixed = M

    τ = 0.5 
    b = 0.6

    num_samples = 4000
    ε = jnp.array(np.random.randn(num_samples) * noise_std)
    # π = jnp.ones(num_samples) # Tested, works fine
    π = jnp.arange(num_samples) / num_samples # Tested, works, but is numerically more delicate. Tolerance for GD needs to be high, step size small, same with that of the prox operator. 
    assert num_samples <= len(ε), "n must be at least as large as the number of samples."

    def joint_obj_jaxopt(x, args):
        τ_, b_ = x
        δ, ε, G, π  = args
        return jnp.power(δ * jnp.mean(jnp.power(S_mapped(ε + τ_ * G, b_, π), 2)) - τ_**2, 2) \
                + jnp.power(δ * jnp.mean(S_prime_mapped(ε + τ_ * G, b_, π)) - 1, 2)

    G = jnp.array(np.random.randn(num_samples))
    gd = jaxopt.GradientDescent(fun=joint_obj_jaxopt, maxiter=1e10, tol = 1e-3, stepsize = 1e-2).run(jnp.array([0.85, 0.75]), (δ, ε, G, π))  # Make step size smaller if getting nans
    res = gd.params
    τ, b = res[0], res[1]

    # Compute σ̃
    σ̃ = np.sqrt(δ) * τ
    # σ = σ̃ / (δ * b)

    return σ̃


