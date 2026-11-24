# Basic Concepts

## Fields

Maps live on a flat, periodic `nside x nside` box with `theta_pix` arcminutes per pixel. Three field types (`cmb_lensing/fields.py`) cover the polarization modes:

| Type | Content | Arrays |
|---|---|---|
| `FlatS0` | a scalar map (e.g. temperature or lensing potential) | `scalar_matrix` |
| `FlatS2` | polarization (E/B - Modes or Q/U - Modes) | `polar_matrix_1`, `polar_matrix_2` |
| `FlatS02` | temperature + polarization | `scalar_matrix`, `polar_matrix_1`, `polar_matrix_2` |

A field carries a **basis**: either MAP (real space, `(nside, nside)`, float64) or FOURIER (the `rfft2` half plane, `(nside, nside//2 + 1)` complex128). Polarization is implicitly in either the EB or the QU parametrization. The conversions are

```python
field = map(field)        #FOURIER -> MAP (irfft2)
field = fourier(field)    #MAP -> FOURIER (rfft2)
field = eb2qu(field)      #EB -> QU (in Fourier)
field = qu2eb(field)      #QU -> EB (in Fourier)
```

`lense_flow` takes fields in MAP / QU. Most other code works in FOURIER / EB.

## Operators

Diagonal-in-Fourier operators (covariances, beam, mask, the mixing matrices) are in `cmb_lensing/matrix_operators.py`:

| Type | Used for |
|---|---|
| `DiagonalScalar` | T-only, and every phi operator |
| `DiagonalEB` | E/B operators |
| `BlockTEB` | T/E/B operators with a 2x2 T/E block (TT, TE, ET, EE, BB) |

Arithmetic is element-wise per block, except `BlockTEB` multiplication, which is the 2x2 T/E matrix product. `pinv` and `log_det` dispatch on the operator type.

Fields and operators are `flax.struct.dataclass` pytrees, so they pass through `jax.jit`, `jax.grad` and `jax.vmap` directly.

## Polarization modes

`pol` selects the configuration everywhere (`load_sim`, `map_joint`, `sample_joint`):

| `pol` | Fields | Operators | Data set |
|---|---|---|---|
| `"I"` | `FlatS0` | `DiagonalScalar` | `DataSetT` |
| `"P"` | `FlatS2` | `DiagonalEB` | `DataSetEB` |
| `"IP"` | `FlatS02` | `BlockTEB` | `DataSetTEB` |

## Simulating data

`cmb_lensing.simulate.load_sim` is the main entry point. It

1. runs CAMB for the requested cosmology,
2. builds the field, phi and noise covariances in Fourier space,
3. draws the unlensed field, phi and the noise,
4. lenses the field with `primal_lense_flow`,
5. returns a `DataSetT` / `DataSetEB` / `DataSetTEB` holding the fields, the operators and the simulated data.

```python
from cmb_lensing.simulate import load_sim
data_set = load_sim(nside = 128,         #pixels per side
                    theta_pix = 2.5,     #arcmin per pixel
                    pol = "IP",          #"I", "P" or "IP"
                    master_seed = 67,
                    uk_arcmin_t = 5,     #white noise level in T (P is sqrt(2) higher)
                    l_knee = 0)          #0 for pure white noise
```

Useful attributes of the returned data set: `unlensed_field`, `lensed_field`, `phi`, `data`, `field_covariance`, `phi_covariance`, `noise_covariance`, `mask`, `beam`, `mixing_d`, `mixing_g`, `quadratic_estimate`.

