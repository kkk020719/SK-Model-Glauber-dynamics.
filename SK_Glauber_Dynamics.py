import jax.numpy as jnp
import jax


def glauber_dynamics(x0, beta, niter, G, key):
    """
    for performing one simulation of glauber dynamics for the given conditions
    :param key: random key
    :param x0: initial state
    :param beta: inverse temperature
    :param G: random matrix of attraction
    :param niter: number of iterations

    """

    dim = len(x0)

    def one_step(x, key):  # current state and the index i
        """one step of glauber dynamics"""
        key, key_ = jax.random.split(key)
        idx = jax.random.randint(key_, shape=(), minval=0, maxval=dim)

        y = x.at[idx].set(1)
        z = x.at[idx].set(-1)
        
        field = 0
        eHamil = 4 * jnp.dot(G[idx], x) + field * sum(x)

        p = jnp.exp(-beta * eHamil / dim ** (1 / 2)) / (jnp.exp(-beta * eHamil / dim ** (1 / 2)) + 1)
        key, key_ = jax.random.split(key)
        x = jax.random.bernoulli(key_, p) * y + (1 - jax.random.bernoulli(key_, p)) * z
        return x, x

    # run glauber
    key_ = jax.random.split(key, niter)
    _, trajectory = jax.lax.scan(one_step, x0, key_)
    return trajectory

def compute_energy(result, niter):
    """ Computes the quadratic Hamiltonian for a given state of spins"""
    def energy_fn(carry, i):
        x, G = carry
        energy = jnp.dot(result[i].T, jnp.dot(result[i], G)) / (n ** (3 / 2))
        return (x, G), energy

    _, energies = lax.scan(energy_fn, (result[0], G), jnp.arange(niter))
    return energies


# compile
run_glauber = jax.jit(glauber_dynamics, static_argnums=(2,))
