# FNO_2: A Convolutional–Fourier Hybrid Operator for Variable-Scale Super-Resolution

`src/model/models_fno_2.py` implements **FNO_2**, a neural operator that maps
a low-resolution fluid state to a high-resolution one. It combines a
convolutional feature extractor with Fourier neural operator (FNO) blocks,
and is designed to support **variable upscale factors** — the same trained
network can produce 2× or 4× super-resolution by passing `upsample_factor`
as a runtime argument to `forward`.

## Architecture

The forward pass has five stages:

**1. Lifting.** A `3×3×3` convolution projects the input from `in_channel`
(5 for 3-D primitives: density, $v_x$, $v_y$, $v_z$, pressure) to a wider
`n_channels`-dimensional feature space.

**2. Residual feature extraction.** A stack of `n_residual_blocks` residual
blocks (each two `3×3×3` convolutions with a ReLU and a skip connection,
`ResBlock` from `models_edsr.py`) refines local spatial features at the LR
resolution. A global skip connection adds the lifted input to the output of
this stack.

**3. Upsampling.** The features are interpolated to the target resolution
using trilinear interpolation (`F.interpolate(..., mode="trilinear"`), with
the scale factor determined by the `upsample_factor` argument. This is the
key point that makes the architecture variable-scale: the network performs
no fixed sub-pixel shuffle, so the same weights apply at any integer upscale
factor.

**4. Spectral refinement.** A stack of `n_operator_blocks` FNO blocks
operates on the upsampled features. Each `FNOBlock` applies a `SpectralConv3d`
layer (global, frequency-domain mixing) and a pointwise `1×1×1` convolution
(`nn.Conv3d(..., 1)`), sums the two, and passes the result through a ReLU.
This is the standard Fourier layer formulation from Li et al. (2021), with
the local path provided by the pointwise conv.

**5. Projection and constraints.** A final `3×3×3` convolution maps back to
`in_channel` output channels. Two optional post-processing steps follow:

- **Softmax constraint** (`SoftmaxConstraint`, from arXiv:2208.05424):
  enforces that the average-pooled super-resolution output matches the
  low-resolution input. It computes `avg_pool3d(sr, k=upsample_factor)` and
  rescales the SR so the pooled version equals the LR, preserving large-scale
  mass/energy consistency. Activated by `apply_constraint`.
- **Positivity ReLU**: applies `ReLU` to channel 0 (density) and channel 4
  (pressure), enforcing the physical non-negativity of these quantities.
  Activated by `apply_positivity_relu=True` (default).

## The Spectral Convolution Layer (`SpectralConv3d`)

The `SpectralConv3d` layer (`src/model/fno_layer.py`) performs the
frequency-domain operation:

1. Compute the 3-D real FFT of the input: $\hat{x} = \text{rFFT3d}(x)$.
2. Apply a learned complex linear transform $R$ to a **truncated set** of
   Fourier modes (the lowest `modes1 × modes2 × modes3` wavenumbers in each
   octant). All higher modes are set to zero — this truncation is what gives
   the FNO its spectral filtering and parameter efficiency.
3. Inverse FFT back to physical space.

Four separate complex weight tensors (`weights1`–`weights4`) handle the four
quadrants of the 3-D Fourier cube formed by the sign combinations of
$(k_x, k_y)$, exploiting the Hermitian symmetry of the real FFT. The weight
shape is `(in_channels, out_channels, modes, modes, modes)` with
`dtype=cfloat`, and the transform is a batched complex matrix multiply via
`einsum("bixyz,ioxyz->boxyz")`.

## Shifting Modes

The `shifting_modes` parameter controls **which portion of the Fourier
spectrum** the learned weights act on. Rather than always applying the
operator to the lowest `modes` wavenumbers ($k \in [0, \text{modes})$), the
layer targets the band
$k \in [\text{shifting\_modes}, \text{shifting\_modes} + \text{modes})$.

Concretely, in `SpectralConv3d.forward` (lines 124–135), the index slices are:

```python
kx_pos = slice(shifting_modes, modes + shifting_modes)    # [s, s+m)
kx_neg = slice(-modes - shifting_modes, -shifting_modes)  # [-s-m, -s)
```

and analogously for $k_y, k_z$. So the operator skips the first
`shifting_modes` wavenumbers and acts on the next `modes` ones. All modes
below `shifting_modes` and above `shifting_modes + modes` are zeroed in the
output.

**Purpose.** Training multiple CFNO variants with different `shifting_modes`
values (the experiment suite uses $\{0, 4, 8, 12, 16\}$) lets each model
specialize in a different spectral band. A model with `shifting_modes=0`
learns the low-frequency (large-scale) structure, while `shifting_modes=16`
targets higher-frequency (smaller-scale) detail. This forms a **spectral
ensemble**: each variant focuses its representational capacity on a different
octave of the turbulent cascade, and together they cover a broader range of
wavenumbers than a single operator with the same `modes` count could refine.

**Constraint.** The layer asserts `modes + shifting_modes ≤ N/2` (line 107),
so the target band must fit within the resolved positive-frequency half of
the grid.

## Configuration Used in This Work

From `train_best_models.py` and `evaluation/benchmark.py`:

| Parameter | Value | Role |
|-----------|-------|------|
| `in_channel` | 5 | primitive state channels |
| `n_channels` | 32 | feature width |
| `n_residual_blocks` | 3 | depth of convolutional feature extractor |
| `n_operator_blocks` | 2 | depth of Fourier refinement (capped at 2; ≥3 diverges, see `AGENTS.md`) |
| `modes` | 16 | Fourier modes per dimension per octant |
| `shifting_modes` | $\{0,4,8,12,16\}$ | spectral band offset (one model per value) |
| `apply_constraint` | `False` | softmax constraint disabled |
| `last_layer_kernel` | 3 | output convolution kernel size |
| `apply_positivity_relu` | `True` | density & pressure non-negativity enforced |

## References

- Li, Z. et al. (2021). *Fourier Neural Operator for Parametric Partial
  Differential Equations.* ICLR. https://openreview.net/pdf?id=c8P9NQVtmnO
- Rahman, S.M.A. et al. (2022). *U-shaped transformer for skip connections and
  Fourier spectral methods.* arXiv:2208.05424 (SoftmaxConstraint).
- “Commonly used (FNO) modification with shared weights” per the inline
  citation at `models_fno_2.py:42`, referencing arXiv:2111.13802.
