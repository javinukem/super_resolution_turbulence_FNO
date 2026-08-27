# Reynolds Number — Turbulent Initial States

Physical analysis of the Reynolds number for the initial turbulent states in
the `training_best_models_experiment` dataset, derived from the parameters in
`data/dataset_specifications.md`. The dataset is generated with **jf1uids**, a
JAX-based compressible Euler solver, so the "Reynolds number" has two distinct
meanings: the **physical Re** of the gas the initial conditions represent, and
the **numerical Re** the simulation actually resolves.

---

## 1. Inputs (from `dataset_specifications.md`)

| Quantity | Symbol | Value |
|----------|--------|-------|
| Mass density (from IC) | $\rho_0 = 2 m_p$ | $3.35 \times 10^{-24}\ \text{g/cm}^3$ |
| Pressure (from IC) | $P_0 = k_B \cdot 3\times 10^4$ | $4.14 \times 10^{-12}\ \text{erg/cm}^3$ |
| Number density (from $\rho_0$) | $n_H = \rho_0 / m_p$ | $2\ \text{cm}^{-3}$ |
| Temperature (implicit in $P_0$) | $T$ | $3 \times 10^4\ \text{K}$ |
| Adiabatic index | $\gamma$ | $5/3$ |
| RMS velocity | $U_\text{rms}$ | $50\ \text{km/s}$ |
| Box size | $L$ | $3\ \text{pc} = 9.26 \times 10^{18}\ \text{cm}$ |
| Grid cells per dim | $N$ | $128$ |
| Cell size | $\Delta x = L/N$ | $7.23 \times 10^{16}\ \text{cm} = 0.0234\ \text{pc}$ |
| Forcing wavenumber range | $k \in$ | $[2, 64]$ |
| Forcing scale | $L_f = L/k_\text{min}$ | $1.5\ \text{pc} = 4.63 \times 10^{18}\ \text{cm}$ |
| Solver | — | Euler (inviscid, no physical viscosity) |

> **Provenance of $T = 3\times 10^4$ K and the $\mu$ inconsistency.**
> Both $\rho_0$ and $P_0$ are set in
> `src/dataset_generation/turbulence_data_generation_all_states.py:115-116`:
> ```python
> rho_0 = 2 * c.m_p / u.cm**3          # -> rho0 = 2 * m_p  [g/cm^3]   => n_H = 2 cm^-3
> p_0  = 3e4 * u.K / u.cm**3 * c.k_B   # -> p0   = 1 * k_B * 3e4  [erg/cm^3] => n = 1 cm^-3
> ```
> The density implies $n_H = 2\ \text{cm}^{-3}$, but the pressure expression
> `3e4 * K / cm**3 * k_B` corresponds to $P = n\,k_B T$ with $n = 1\ \text{cm}^{-3}$
> (the `1/u.cm**3` is the astropy unit denominator). For neutral atomic H
> ($\mu = 1$) the pressure should be $P = 2\, k_B T$; for fully ionized H
> ($\mu \approx 0.5$, $n_\text{tot} = n_H + n_e = 4\ \text{cm}^{-3}$) it should
> be $P = 4\, k_B T$. The stored $P_0/\rho_0$ ratio implies an **effective
> mean molecular weight**:
> $$\mu_\text{eff} = \frac{\rho_0}{m_p \cdot n_\text{from P}} = \frac{2 m_p}{m_p \cdot 1} = 2$$
> This corresponds to molecular H₂ — physically inconsistent with
> $T = 3\times 10^4$ K (where H₂ is fully dissociated) and almost certainly an
> unintended factor-of-2 (or 4) error in the `p_0` expression.
>
> **However, this does not affect the simulation dynamics.** jf1uids never
> uses $\mu$ or $T$ — the solver computes the sound speed directly from the
> stored primitives as $c_s = \sqrt{\gamma P/\rho}$
> (`jf1uids/fluid_equations/fluid.py:328`). Molecular weight only appears in
> a post-processing helper (`CodeUnits.get_temperature_from_internal_energy`)
> that is never called during integration. The ground-truth $c_s$ for the
> Reynolds analysis is therefore $\sqrt{\gamma P_0/\rho_0}$, computed below.

---

## 2. Derived flow parameters

The sound speed is computed from the **solver's own definition**, using the
stored $P_0$ and $\rho_0$ — no $\mu$ or $T$ needed:

$$c_s = \sqrt{\frac{\gamma P_0}{\rho_0}} = \sqrt{\frac{(5/3) \cdot 4.14\times 10^{-12}}{3.35\times 10^{-24}}} = 14.4\ \text{km/s}$$

For reference, the naive $c_s = \sqrt{\gamma k_B T / (\mu m_p)}$ gives:
- $\mu = 1$ (neutral atomic H): $c_s = 20.3\ \text{km/s}$ — **wrong**, does not
  match the stored $P_0/\rho_0$ ratio.
- $\mu = 2$ (molecular H₂, what the data implies): $c_s = 14.4\ \text{km/s}$ —
  matches, confirming $\mu_\text{eff} = 2$.

$$\mathcal{M} = \frac{U_\text{rms}}{c_s} = \frac{50}{14.4} \approx 3.48$$

The initial state is **strongly supersonic** (Mach ≈ 3.5).

---

## 3. Physical Reynolds number

Since jf1uids solves the **Euler equations** (no viscosity term), the physical
Re depends on the microphysical viscosity of the gas at $T = 3\times 10^4$ K —
which is not part of the simulation but characterizes what the ICs represent.
Two limiting cases bracket the answer:

### 3.1 Neutral hydrogen (mean-free-path estimate)

Using $n_H = 2\ \text{cm}^{-3}$ (from $\rho_0$) for the collision rate, and the
thermal speed from the stored $P_0/\rho_0$ (consistent with the solver's
$c_s$, avoiding the inconsistent $T$):

$$v_\text{th} = \sqrt{\frac{8 P_0}{\pi \rho_0}} = 17.8\ \text{km/s}$$

$$\lambda_\text{mfp} = \frac{1}{n_H \sigma_{HH}}, \quad \sigma_{HH} \approx 10^{-15}\ \text{cm}^2$$

$$\nu = 0.499\, v_\text{th}\, \lambda_\text{mfp}
     = 0.499 \sqrt{\frac{8 P_0}{\pi \rho_0}} \cdot \frac{1}{n_H \sigma_{HH}}$$

| Quantity | Value |
|----------|-------|
| Thermal speed $v_\text{th}$ (from $P_0/\rho_0$) | $17.8\ \text{km/s}$ |
| Mean free path $\lambda_\text{mfp}$ | $5.0 \times 10^{14}\ \text{cm} = 1.6 \times 10^{-4}\ \text{pc}$ |
| Kinematic viscosity $\nu$ | $4.43 \times 10^{20}\ \text{cm}^2/\text{s}$ |
| **Re (box, $L = 3$ pc)** | **$1.0 \times 10^5$** |
| **Re (forcing, $L_f = 1.5$ pc)** | **$5.2 \times 10^4$** |

### 3.2 Ionized hydrogen (Spitzer viscosity)

$$\eta_S = \frac{5}{6} \frac{\sqrt{m_p}\, (k_B T)^{5/2}}{\pi^{3/2} e^4 \ln\Lambda}, \quad \ln\Lambda = 20$$

| Quantity | Value |
|----------|-------|
| Dynamic viscosity $\eta_S$ | $6.35 \times 10^{-6}\ \text{g/(cm s)}$ |
| Kinematic viscosity $\nu_S$ | $1.90 \times 10^{18}\ \text{cm}^2/\text{s}$ |
| **Re (box, $L = 3$ pc)** | **$2.4 \times 10^7$** |
| **Re (forcing, $L_f = 1.5$ pc)** | **$1.2 \times 10^7$** |

At $T = 3\times 10^4$ K the gas is partially ionized, so the true physical Re
lies between these two estimates: roughly $\mathbf{10^5 \lesssim \text{Re}
\lesssim 10^7}$.

---

## 4. Numerical Reynolds number (what the simulation resolves)

The solver is inviscid, so the effective viscosity is set by **numerical
dissipation**, which scales as $\nu_\text{num} \sim c_s \Delta x$:

$$\nu_\text{num} \approx c_s \cdot \Delta x = 14.4\ \text{km/s} \times 7.23\times 10^{16}\ \text{cm} = 1.04 \times 10^{23}\ \text{cm}^2/\text{s}$$

$$\text{Re}_\text{num} = \frac{U_\text{rms} \cdot L}{c_s \cdot \Delta x}
                       = \mathcal{M} \cdot N
                       = 3.48 \times 128
                       \approx \mathbf{446}$$

This is the Reynolds number the simulation actually captures. It is many
orders of magnitude below the physical Re of the gas — the simulation is
**strongly under-resolved** in the viscous-dissipation sense, which is
expected and standard for driven-turbulence Euler simulations used as ML
training data.

### Resolved cascade range

The forcing injects energy at $k \in [2, 64]$, giving a resolved inertial
range of

$$\frac{k_\text{max}}{k_\text{min}} = \frac{64}{2} = 32\ \ \text{(≈ 5 octaves)}$$

Beyond $k = 64$ (scales smaller than $L/64 \approx 0.047$ pc, i.e. ~2 grid
cells), energy is removed by numerical (shock-capturing) dissipation rather
than a physical viscous cutoff.

---

## 5. Kolmogorov dissipation scale

For the physical Re values, the Kolmogorov microscale is

$$\eta = L\, \text{Re}^{-3/4}$$

| Case | Re (box) | $\eta$ | $\eta / \Delta x$ |
|------|----------|--------|---------------------|
| Neutral H | $1.0 \times 10^5$ | $1.6 \times 10^{15}\ \text{cm} = 5.2 \times 10^{-4}\ \text{pc}$ | $2.2 \times 10^{-2}$ |
| Ionized H | $2.4 \times 10^7$ | $2.7 \times 10^{13}\ \text{cm} = 8.6 \times 10^{-6}\ \text{pc}$ | $3.7 \times 10^{-4}$ |

In both cases $\eta \ll \Delta x$ — the true viscous dissipation scale is
**orders of magnitude below the grid resolution**. The simulation cannot
resolve the physical dissipation scale; this is the regime where an Euler
(implicit-LES) approach is the standard modeling choice.

---

## 6. Summary

| Regime | Reynolds number | Notes |
|--------|-----------------|-------|
| Physical (neutral H) | $\sim 10^5$ | Mean-free-path viscosity, $v_\text{th}$ from $P_0/\rho_0$ |
| Physical (ionized H) | $\sim 10^7$ | Spitzer viscosity, $\ln\Lambda = 20$ |
| **Numerical (simulation)** | **$\sim 4.5 \times 10^2$** | $\mathcal{M} \cdot N = 3.48 \times 128$ |
| Mach number | $3.48$ | Strongly supersonic |
| Sound speed | $14.4\ \text{km/s}$ | From $\sqrt{\gamma P_0/\rho_0}$ (solver definition) |
| Thermal speed | $17.8\ \text{km/s}$ | From $\sqrt{8 P_0/(\pi \rho_0)}$ (consistent with solver) |
| Effective $\mu$ | $2$ | IC inconsistency ($\rho_0$ implies $n=2$, $P_0$ implies $n=1$) |
| Resolved cascade | ~5 octaves | $k \in [2, 64]$ |
| Kolmogorov scale vs grid | $\eta / \Delta x \sim 10^{-2}$ to $10^{-4}$ | Strongly unresolved (expected for Euler ILES) |

The dataset represents **compressible, strongly supersonic (Mach ~3.5)
turbulence** with a numerical Reynolds number of ~450, far below the physical
Re of the ISM gas it models. The initial conditions use a $k^{-2}$ power
spectrum over $k \in [2, 64]$, and the Euler solver acts as an
implicit-LES: numerical dissipation replaces the unresolved physical viscous
cutoff. This is consistent with the intended use as super-resolution training
data — the ML model learns to recover small-scale structure that the
simulation resolves only down to the grid scale.

---

## Reproduction

The script that computed these values is at
`data/compute_reynolds_number.py` (run with `python3`). It uses only standard
physical constants and the parameters listed in §1. The sound speed is
computed from $\sqrt{\gamma P_0/\rho_0}$ (the solver's own definition), not
from $T$ and an assumed $\mu$.
