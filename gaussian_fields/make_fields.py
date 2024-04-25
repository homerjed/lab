import os
import jax
import jax.numpy as jnp
import jax.random as jr 
# import jax_cosmo as jc
import numpy as np
from tensorflow_probability.substrates.jax import distributions as tfd


def Pk_cosmology(k, A=0.4, B=0.6):
    cosmo_params = jc.Planck15(Omega_c=A, sigma8=B)
    return jc.power.linear_matter_power(cosmo_params, k)


def Pk(k, A, B):
    return A * (k ** -B)


def simulator(rng, θ, **simulator_kwargs):#foregrounds=None):
    def fn(rng, A, B):
        dim = len(simulator_kwargs["shape"])
        L = simulator_kwargs["L"]
        if jnp.isscalar(L):
            L = [L] * int(dim)
        Lk = ()
        shape = ()
        for i, _shape in enumerate(simulator_kwargs["shape"]):
            Lk += (_shape / L[i],)
            if _shape % 2 == 0:
                shape += (_shape + 1,)
            else:
                shape += (_shape,)
        
        k = simulator_kwargs["k"]
        k_shape = k.shape
        k = k.flatten()[1:]
        # tpl = ()
        # for _d in range(dim):
        #     tpl += (_d,)

        V = jnp.prod(jnp.array(L))
        scale = V ** (1. / dim)            
        fft_norm = jnp.prod(jnp.array(Lk))

        rng, key = jr.split(rng)
            
        # Gaussian magnitudes and uniform phases (chosen for GRF)
        mag = jr.normal(key, shape=shape)
        pha = 2. * jnp.pi * jr.uniform(key, shape=shape)

        # Make hermitian field (reality condition)
        revidx = (slice(None, None, -1),) * dim
        mag = (mag + mag[revidx]) / jnp.sqrt(2) 
        pha = (pha - pha[revidx]) / 2 + jnp.pi
        dk = mag * (jnp.cos(pha) + 1j * jnp.sin(pha))
        cutidx = (slice(None, -1),) * dim
        dk = dk[cutidx]
        
        Pk_fn = Pk_cosmology if simulator_kwargs["cosmological"] else Pk
            
        powers = jnp.concatenate(
            [jnp.zeros(1), jnp.sqrt(Pk_fn(k, A=A, B=B))]
        ).reshape(k_shape)
        
        if simulator_kwargs['vol_norm']:
            powers /= V
            
        if simulator_kwargs["log_normal"]:
            powers = jnp.real(
                jnp.fft.ifftshift(
                    jnp.fft.ifftn(powers) * fft_norm) * V)
    
            powers = jnp.log(1. + powers)
            powers = jnp.abs(jnp.fft.fftn(powers))  
        
        fourier_field = powers * dk
        # fourier_field = jax.ops.index_update(
        #     fourier_field, np.zeros(dim, dtype=int), np.zeros((1,)))
        fourier_field = fourier_field.at[dim].set(0.)
        
        if simulator_kwargs["log_normal"]:
            field = jnp.real(jnp.fft.ifftn(fourier_field)) * fft_norm * jnp.sqrt(V)
            field = jnp.exp(field - jnp.var(field) / 2.) - 1.
        else:
            field = jnp.real(jnp.fft.ifftn(fourier_field) * fft_norm * jnp.sqrt(V) ** 2.)
            
        if simulator_kwargs["N_scale"]:
            field *= scale    
            
        # if foregrounds is not None:
        #     rng, key = jr.split(key)
        #     foreground = foregrounds[
        #         jr.randint(
        #             key, 
        #             minval=0, 
        #             maxval=foregrounds.shape[0], 
        #             shape=())]    
        #     field = jnp.expand_dims(field + foreground, (0,))
            
        if not simulator_kwargs["squeeze"]:
            field = jnp.expand_dims(field, (0, 1))
            
        return jnp.array(field, dtype='float32')

    if isinstance(θ, tuple):
        A, B = θ
    else:
        A = jnp.take(θ, 0, axis=-1)
        B = jnp.take(θ, 1, axis=-1)
    if A.shape == B.shape:
        if len(A.shape) == 0:
            return fn(rng, A, B)
        else:
            keys = jr.split(rng, num=A.shape[0] + 1)
            rng = keys[0]
            keys = keys[1:]
            return jax.vmap(
                lambda key, A, B: simulator(
                    key, (A, B), simulator_kwargs=simulator_kwargs))(
                keys, A, B)

if __name__ == "__main__":
    from chainconsumer import ChainConsumer
    import matplotlib.pyplot as plt

    key = jr.PRNGKey(0)

    cosmological = False
    LN = True
    n_sims = 20_000
    n_pix = 64
    shape = (n_pix, n_pix)
    AB = jnp.array([0.5, 0.5]) # Fiducial cosmology
    n_fids = 400_000

    lower = jnp.array([0.1, 0.1])
    upper = jnp.array([1., 1.])

    parameter_prior = tfd.Blockwise([
        tfd.Uniform(lower[i], upper[i]) for i in range(2)]
    )
    
    print(f"Making {'log normal' if LN else 'Gaussian'} fields for n_pix={n_pix}...")

    parameters = parameter_prior.sample(n_sims, seed=key)

    k = jnp.sqrt(
        jnp.sum(
            jnp.array(
                jnp.meshgrid(
                    *(
                        (
                            jnp.hstack(
                                (
                                    jnp.arange(0, _shape // 2 + 1),  
                                    jnp.arange(-_shape // 2 + 1, 0)
                                )
                            ) * 2 * jnp.pi / _shape
                        ) ** 2. 
                    for _shape in shape
                )
            )
        ), axis=0)
    )
    print(k.shape)

    simulator_kwargs = dict(
        k=k,
        L=n_pix,
        shape=shape,
        vol_norm=True,
        N_scale=True,
        squeeze=True,
        log_normal=LN,
        cosmological=cosmological
    )

    keys = jr.split(key, n_sims)

    _simulator = lambda key, θ: simulator(key, θ, **simulator_kwargs)

    fields = jax.vmap(_simulator)(keys, parameters)

    data_dir = "/Users/Jed.Homer/phd/teaching/sbi_ml_lab/gaussian_fields/data/"

    field_type = "LN" if LN else "G"
    filename = field_type + "_fields.npy"

    jnp.save(os.path.join(data_dir, filename), fields)
    jnp.save(os.path.join(data_dir, "field_parameters.npy"), parameters)

    keys = jr.split(key, n_fids)

    fields = jax.vmap(_simulator, in_axes=(0, None))(keys, AB)

    mean = fields.mean(axis=0)
    covariance = jnp.cov(fields.reshape(n_fids, -1), rowvar=False)

    # Get derivatives (jacobian of simulator w.r.t. parameters at fixed parameters for same noise realisations as those for mean)
    _simulator = lambda θ, key: simulator(key, θ, **simulator_kwargs)
    derivatives = jax.vmap(jax.jacfwd(_simulator, argnums=0), in_axes=(None, 0))(AB, keys[:20_000])
    # _simulator = lambda key, θ: simulator(key, θ, **simulator_kwargs)
    # derivatives = jax.vmap(jax.jacfwd(_simulator, argnums=0), in_axes=(0, None))(keys[:20_000], AB)
    print(derivatives.shape)
    derivatives = derivatives.mean(axis=0)
    print(derivatives.shape)
    derivatives = derivatives.reshape(-1, 2).T
    # derivatives = derivatives.transpose(2, 0, 1).reshape(2, -1)

    jnp.save(os.path.join(data_dir, field_type + "_alpha.npy"), AB)
    jnp.save(os.path.join(data_dir, field_type + "_mean.npy"), mean)
    jnp.save(os.path.join(data_dir, field_type + "_covariance.npy"), covariance)
    jnp.save(os.path.join(data_dir, field_type + "_derivatives.npy"), derivatives)

    # Fisher forecast

    def fisher(θ, k, N):
        A, B = θ
        k = k[1:N // 2, 1:N // 2].flatten()
        _Pk = Pk(k, A, B)
        
        Cinv = jnp.diag(1. / _Pk)
        C_A = jnp.diag(k ** -B)
        C_B = jnp.diag(-_Pk * np.log(k))

        F_AA = 0.5 * jnp.trace((C_A @ Cinv @ C_A @ Cinv))
        F_AB = 0.5 * jnp.trace((C_A @ Cinv @ C_B @ Cinv))
        F_BA = 0.5 * jnp.trace((C_B @ Cinv @ C_A @ Cinv))
        F_BB = 0.5 * jnp.trace((C_B @ Cinv @ C_B @ Cinv))

        return jnp.array([[F_AA, F_AB], [F_BA, F_BB]])

    F = fisher(AB, k, n_pix)
    Finv = jnp.linalg.inv(F)

    jnp.save(os.path.join(data_dir, field_type + "_Finv.npy"), Finv)

    # H = (n_fids - (n_pix ** 2) - 2) / (n_fids - 1)
    # precision = jnp.linalg.inv(covariance)
    # F = jnp.linalg.multi_dot([derivatives, precision, derivatives.T])
    # Finv = jnp.linalg.inv(F)

    c = ChainConsumer()
    AB = np.asarray(AB)
    c.add_covariance(AB, Finv)
    fig = c.plotter.plot(truth=AB, figsize=(5., 5.))
    plt.savefig("Finv.png")
    plt.close()

    print("...done.")
