import jax
jax.config.update("jax_enable_x64", True) # float64
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jaxopt
from tqdm.auto import tqdm
import numpy as np
import jax.scipy.optimize as jso

def norm_sq_corr(x1, x2):
  return jnp.square(jnp.dot(x1, x2)) / (jnp.square(jnp.linalg.norm(x1)) * jnp.square(jnp.linalg.norm(x2)) + 1e-16)

def MSE(beta, beta_hat):
  return jnp.mean(jnp.square(beta - beta_hat))

""" Non-separable AMP to simulate Weighted ERM. Takes in a sequence of loss functions ℓᵢ, assumes no penalty on the signal for now """

def f(w, c):
    return -w/c
f_mapped = jax.vmap(f, in_axes=(0, None))

def ḟ(w, c):
    return -1/c
ḟ_mapped = jax.vmap(ḟ, in_axes=(0, None))

def obj_(z, args): 
    """ Proximal objective of g. ℓ_fixed must be globally defined through the AMP function for this to be well defined. """
    u, v, b, π_i = args
    return (π_i * ℓ_fixed(z, v) + (1/(2*b)) * jnp.linalg.norm(z - u)**2)[0] # Have to index at zero because jaxopt.minimize expects a scalar but passes in an array initializer

def obj_no_idx(z, args):
    """ Wrapper for obj_ that does not index into the result, works with jaxopt.GradientDescent. """
    u, v, b, π_i = args
    return (π_i * ℓ_fixed(z, v) + (1/(2*b)) * jnp.linalg.norm(z - u)**2)

def ḡ(u, v, b, π_i):
    """ Proximal Operator of f"""

    # Solve using jaxopt, so that we can differentiate through the solution

    # gd = jaxopt.GradientDescent(fun=obj_no_idx, maxiter=10000, implicit_diff=True)
    # init_params = 1.0
    # args = (u, v, b, π_i)
    
    # # test_grad = jax.grad(obj_no_idx, 0)
    # # test_res = test_grad(13., (1.5, 1.5, 1.5, 1.5))
    # # test_grad = jax.grad(lambda x: obj_(x, args), 0)(13.0)
    # # jax.debug.print("test_grad: {x}", x = test_res)
    # return gd.run(init_params, args).params # Works, runs slightly faster than scipy opt, and is differentiable. 

    # solve using scipy, is a bit slower but more reliable. 
    # args = (u, v, b, π_i)
    # res = jso.minimize(obj_, jnp.array([1.0]), (args, ), method="BFGS", options={"maxiter": 10000}) # Currently, this optimization is not always successful, even after increasing maxiter.
    # jax.debug.print("res status: {x}, success: {y}", x =res.status, y = res.success)
    # jax.debug.print("res success: {x}", x =res.success) # The optimization is not always successful, that might cause problems. 
    # return res.x[0] # was giving nans

    # Solve using grad descent like in the state evolution equations
    gd = jaxopt.GradientDescent(fun=obj_no_idx, maxiter=1e10, tol = 1e-7, stepsize = 1e-1, implicit_diff=True) # Extremely fast, and seems to give just as good results
    args = (u, v, b, π_i)
    res = gd.run(0.85, args).params
    return res
ḡ_mapped = jax.vmap(ḡ, in_axes=(0, 0, None, 0))

def ḡ̇(u, v, b, π_i):
    """ Tested, and yields the correct result in linear regression with change points. """
    ℓ_double_prime = jax.grad(jax.grad(ℓ_fixed))
    g_plug = ḡ(u, v, b, π_i)
    return (b * π_i * ℓ_double_prime(g_plug, v) + 1) ** -1
ḡ̇_mapped = jax.vmap(ḡ̇, in_axes=(0, 0, None, 0))

def g(u, v, b, π_i):
    return (ḡ(u, v, b, π_i) - u) / b
g_mapped = jax.vmap(g, in_axes=(0, 0, None, 0))

def ġ_jax(u, v, b, π_i):
    return jax.grad(g, 0)(u, v, b, π_i)
ġ_jax_mapped = jax.vmap(ġ_jax, in_axes=(0, 0, None, 0))

def ġ(u, v, b, π_i):
    return (ḡ̇(u, v, b, π_i) - 1) / b
ġ_mapped = jax.vmap(ġ, in_axes=(0, 0, None, 0))

def dgdz(u, v, b, ℓ):
    """ To be tested """
    ℓ_prime = jax.grad(ℓᵢ)
    return b * ℓ_prime(u, v)

# Can already try running the regular AMP. 

def amp(X, y, n, p, π, ℓ, max_iter=10):
    """ AMP Algorithm, π is the marginal probability of each signal being present, ℓ is the loss function (same for every i). 
    Works, but iterates blow up to nan eventually, possibly due to numerics. 
    """

    # Define the global loss ℓ_fixed once ℓ is passed in, as functions are not valid JAX types to vmap over
    global ℓ_fixed
    ℓ_fixed = ℓ

    ŝ = jnp.zeros(n)
    b = 0.5 # initial value of b must be greater than zero
    β̂ = np.random.normal(size=p) # initial value of β must be greater than zero
    θ_prev = jnp.zeros(n)
    for _ in tqdm(range(max_iter)):
        # θ update steps
        θ = X @ β̂ - b * ŝ
        # θ_test = ḡ(θ[0], y[0], b, π[0])
        # print("θ_test args: ", θ[0], y[0], b, π[0])
        # print("θ_test: ", θ_test)
        θ̂ = ḡ_mapped(θ, y, b, π).flatten()
        ŝ = g_mapped(θ, y, b, π).flatten()
        c = jnp.mean(ġ_mapped(θ, y, b, π), axis=0) # This is likely the bottleneck in computation
        print("c: ", c)

        # β update steps
        β = X.T @ ŝ - c * β̂
        β̂ = f_mapped(β, c)
        b = p/n * jnp.mean(ḟ_mapped(β, c), axis=0) # This can be made closed form

        # Check convergence
        print("Normalized MSE: ", MSE(θ_prev, θ) / (np.linalg.norm(θ_prev) + 1e-16)**2)
        θ_prev = θ

    return β̂, θ̂, β, θ