# EDSR: Enhanced Deep Residual Network for 3-D Fluid Super-Resolution

`src/model/models_edsr.py` implements **EDSR**, a fully convolutional
super-resolution network adapted from the 2-D image-SR architecture of Lim et
al. (2017) to 3-D volumetric fluid states. It is the convolutional baseline
against which the Fourier-based FNO_2 is compared in this work. Unlike FNO_2,
the upscale factor is fixed at construction time via `PixelShuffle3d`, so a
given EDSR instance operates at a single scale (×4 in this work).

## Architecture

The forward pass has three stages — **head → body → tail** — following the
canonical EDSR design.

**1. Head (lifting).** A single `3×3×3` convolution
(`nn.Conv3d(input_channels, n_feats, kernel_size, padding="same")`) projects
the input from `input_channels` (5 for 3-D primitives: density, $v_x$, $v_y$,
$v_z$, pressure) to the `n_feats`-dimensional feature space.

**2. Body (residual feature extraction).** A stack of `n_resblocks` residual
blocks (`ResBlock`), followed by a single `3×3×3` convolution. Each `ResBlock`
consists of two `3×3×3` convolutions with a ReLU between them and an identity
skip connection:

```
x → Conv3d → ReLU → Conv3d → (+x) → out
```

A global skip connection adds the head output to the body output
(`res += x` in `forward`). This double skip (per-block + global) is the
defining feature of the "enhanced" EDSR variant over the original ResNet-based
SR networks: it stabilizes training of deep stacks (32+ blocks in the original
paper) and lets the body learn a residual refinement rather than a full
mapping.

**3. Tail (upsampling + projection).** The tail consists of:

- **`Upsampler`**: a sequential module that implements learned upsampling via
  `PixelShuffle3d`. For a scale factor $s = 2^n$, it applies $n$ identical
  stages, each composed of:
  1. `nn.Conv3d(n_feats, 8·n_feats, kernel_size=3, padding=1)` — expands the
     channel dimension by $2^3 = 8$ (the 3-D analogue of the 2-D $r^2$ factor);
  2. `PixelShuffle3d(2)` — reshapes the 8×-expanded channels into a
     $2×2×2$ spatial block, doubling each spatial dimension.

  For `scale = 4`, two such stages are chained (`log₂4 = 2`), so the spatial
  volume grows by $4^3 = 64$-fold. The channel count returns to `n_feats`
  after each shuffle. An optional activation (`"relu"` or `"prelu"`) and
  batch norm can be inserted per stage; both are disabled in this work.
- A final `3×3×3` convolution projects from `n_feats` back to
  `input_channels` output channels.

**4. Positivity constraint (optional).** If
`apply_positivity_relu=True` (default), a `ReLU` is applied to channel 0
(density) and channel 4 (pressure) after the tail, enforcing the physical
non-negativity of these quantities — identical to the FNO_2 positivity
constraint.

## The `PixelShuffle3d` Upsampling Layer

`PixelShuffle3d` (`src/utils/pixel_shuffle3d.py`, adapted from
github.com/scalyvladimir/pixel_shuffle3d) is the 3-D generalization of
Shi et al.'s (2016) pixel shuffle. It is a parameter-free reshape operation:

1. **Channel expansion** (done by the preceding `Conv3d`): the convolution
   maps `n_feats → 8·n_feats` channels, packing the $2×2×2$ sub-voxel
   information into the channel axis.
2. **Reshape**: the 8-fold channel factor is rearranged into a
   $2×2×2$ spatial block, doubling each spatial dimension while reducing
   channels back to `n_feats`. Formally, an input
   `(B, 8·C, D, H, W)` becomes `(B, C, 2D, 2H, 2W)`.

This is **learned upsampling** — unlike trilinear interpolation, the
convolution weights determine how sub-voxel detail is distributed. It is the
standard SR upsampling mechanism and is more expressive than interpolation,
but it **bakes the scale factor into the network architecture**: an EDSR
trained at ×4 cannot be evaluated at ×2 without architectural changes, in
contrast to FNO_2 where `upsample_factor` is a runtime argument.

## `ResBlock`

The residual block (`models_edsr.py:12–30`) is the standard post-activation
ResNet v2 block, adapted to 3-D:

```
Conv3d(3×3×3, same) → ReLU → Conv3d(3×3×3, same) → (+identity)
```

No batch normalization is used — this is the key simplification of EDSR over
SRResNet: removing BN reduces memory and stabilizes training at depth, at the
cost of a small accuracy hit that is compensated by the ability to stack more
blocks. The block preserves spatial and channel dimensions, so it can be
repeated any number of times.

## Configuration Used in This Work

From `train_best_models.py` and `evaluation/benchmark.py`:

| Parameter | Value | Role |
|-----------|-------|------|
| `input_channels` | 5 | primitive state channels |
| `n_resblocks` | 16 | depth of residual body |
| `n_feats` | 64 | feature width |
| `kernel_size` | 3 | convolution kernel size |
| `scale` | 4 | fixed upscale factor (two `PixelShuffle3d(2)` stages) |
| `activation_f` | `False` | no activation in the upsampler stages |
| `apply_positivity_relu` | `True` | density & pressure non-negativity enforced |

For comparison, the original 2-D EDSR paper used `n_resblocks=32`,
`n_feats=256`, `scale=4` — the configuration here is substantially smaller
(64 feats vs 256, 16 blocks vs 32) to fit the 3-D volume and the cluster's
GPU memory budget.

## Comparison with FNO_2

| Aspect | EDSR | FNO_2 |
|--------|------|-------|
| Core operation | Local convolution (3×3×3) | Global spectral conv + local conv |
| Upsampling | Learned `PixelShuffle3d` (fixed scale) | Trilinear interpolation (runtime scale) |
| Variable scale | No — scale baked into architecture | Yes — `upsample_factor` is a forward arg |
| Spectral awareness | None — inductive bias is local | Explicit — Fourier modes are learned directly |
| Parameter count | ~5.4 M (this config) | ~33.7 M (this config) |
| Memory at training | Lower (no FFT, no complex weights) | Higher (3-D FFT, 4 complex weight tensors) |
| Physics constraints | Positivity ReLU on ρ, P | Positivity ReLU + optional `SoftmaxConstraint` |
| Role in this work | Convolutional baseline | Flagship operator |

EDSR is a pure local-convolution architecture: its receptive field grows
linearly with depth (each `ResBlock` adds 2 cells per direction, so 16 blocks
give ~32 cells of effective receptive field — about one quarter of the 128³
HR box). FNO_2, by contrast, has global receptive field from the first
spectral layer, which is the structural advantage it exploits for turbulent
fields with long-range correlations.

## References

- Lim, B., Son, S., Kim, H., Nah, S., & Lee, K.M. (2017). *Enhanced Deep
  Residual Networks for Single Image Super-Resolution.* CVPR Workshops.
  https://arxiv.org/abs/1707.02921
- Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P., Bishop, R.,
  Rueckert, D., & Wang, Z. (2016). *Real-Time Single Image and Video
  Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural
  Network.* CVPR. https://arxiv.org/abs/1609.05158
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for
  Image Recognition.* CVPR. https://arxiv.org/abs/1512.03385
- 3-D `PixelShuffle3d` implementation adapted from
  github.com/scalyvladimir/pixel_shuffle3d.
