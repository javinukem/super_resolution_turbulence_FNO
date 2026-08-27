"""
Reynolds number computation for the turbulent initial states of the
training_best_models_experiment dataset.

Physical parameters come from data/dataset_specifications.md.
The sound speed is computed from the actual stored P0 and rho0 using
the solver's own definition  c_s = sqrt(gamma * P / rho)  — no mean
molecular weight is needed, and the solver itself never uses one.

Run:  python3 data/compute_reynolds_number.py
"""

import numpy as np

# ── Physical constants (cgs) ──
m_p = 1.6726e-24   # proton mass [g]
k_B = 1.3806e-16   # Boltzmann [erg/K]
e   = 4.8032e-10   # electron charge [esu]
pc  = 3.0857e18    # parsec [cm]
km  = 1e5          # kilometre [cm]

# ── Dataset parameters (from data/dataset_specifications.md) ──
# IC definitions in turbulence_data_generation_all_states.py:115-116:
#   rho_0 = 2 * c.m_p / u.cm**3          -> rho0 = 2 * m_p  [g/cm^3]
#   p_0  = 3e4 * u.K / u.cm**3 * c.k_B   -> p0   = 1 * k_B * 3e4  [erg/cm^3]
#
# NOTE: the density implies n_H = 2 cm^-3, while the pressure implies
# n_particles = 1 cm^-3. This gives an effective mean molecular weight
# mu_eff = rho0 / (m_p * n_from_P) = 2, i.e. the stored P/rho ratio is
# inconsistent with neutral atomic H (mu=1) or fully ionized H (mu=0.5).
# The solver does NOT use mu at all — it computes c_s = sqrt(gamma*P/rho)
# directly (jf1uids/fluid_equations/fluid.py:328) — so the ground-truth
# sound speed for the simulation is sqrt(gamma * p0 / rho0), independent
# of any interpretation of mu or T.

rho0   = 2 * m_p                  # mass density from IC [g/cm^3]
p0     = 1 * k_B * 3e4            # pressure from IC (n=1 cm^-3, T=3e4 K) [erg/cm^3]
n_H    = 2.0                      # number density implied by rho0 [cm^-3]
n_fromP = 1.0                     # number density implied by p0 [cm^-3]
mu_eff = rho0 / (m_p * n_fromP)   # effective mean molecular weight = 2
gamma  = 5.0 / 3.0
U_rms  = 50 * km                  # RMS velocity [cm/s]
L_box  = 3 * pc                   # box size [cm]
N_grid = 128                      # cells per dimension
dx     = L_box / N_grid           # cell size [cm]
k_min  = 2
k_max  = 64
L_force = L_box / k_min           # forcing scale [cm]

# ── Sound speed: solver's definition (ground truth) ──
c_s = np.sqrt(gamma * p0 / rho0)
M   = U_rms / c_s

print(f"rho0              = {rho0:.3e} g/cm^3   (n_H = {n_H} cm^-3)")
print(f"p0                = {p0:.3e} erg/cm^3  (n_fromP = {n_fromP} cm^-3, T = 3e4 K)")
print(f"mu_eff = rho/(m_p*n_fromP) = {mu_eff}  (IC inconsistency, see note)")
print()
print(f"Sound speed (solver: sqrt(gamma*P/rho)) = {c_s/km:.2f} km/s")
print(f"  vs sqrt(gamma*k_B*T/m_p)  [mu=1]      = {np.sqrt(gamma*k_B*3e4/m_p)/km:.2f} km/s  (WRONG — assumes mu=1)")
print(f"  vs sqrt(gamma*k_B*T/(2*m_p)) [mu=2]   = {np.sqrt(gamma*k_B*3e4/(2*m_p))/km:.2f} km/s")
print(f"Mach number       M = {M:.2f}")
print(f"Box size          L = {L_box/pc:.1f} pc")
print(f"Forcing scale     L_f = {L_force/pc:.2f} pc")
print(f"Cell size         dx = {dx:.3e} cm")
print()

# ── Case 1: neutral H, mean-free-path viscosity ──
# Use n_H = 2 cm^-3 (from rho0) for the mean free path, since the
# collision rate depends on the actual particle density.
# Thermal speed from P0/rho0 (consistent with the solver), not from T
# with an assumed mu — same rationale as c_s above.
v_th        = np.sqrt(8 * p0 / (np.pi * rho0))
sigma_HH    = 1e-15                       # cm^2
lambda_mfp  = 1.0 / (n_H * sigma_HH)      # cm
mu_neutral  = 0.499 * rho0 * v_th * lambda_mfp
nu_neutral  = mu_neutral / rho0

print("Neutral H (mean-free-path, n_H = 2 cm^-3, v_th from P0/rho0):")
print(f"  v_th        = {v_th/km:.2f} km/s")
print(f"  nu          = {nu_neutral:.3e} cm^2/s")
print(f"  Re (box)    = {U_rms*L_box/nu_neutral:.2e}")
print(f"  Re (forcing)= {U_rms*L_force/nu_neutral:.2e}")
print()

# ── Case 2: ionized H, Spitzer viscosity ──
# For ionized H at n_H = 2 cm^-3: n_tot = n_p + n_e = 4 cm^-3.
# Spitzer viscosity uses n_i for the ion contribution.
ln_Lambda = 20.0
eta_S     = (5.0/6.0) * np.sqrt(m_p) * (k_B * 3e4)**2.5 / (np.pi**1.5 * e**4 * ln_Lambda)
nu_S      = eta_S / rho0

print("Ionized H (Spitzer, T = 3e4 K, ln Lambda = 20):")
print(f"  nu          = {nu_S:.3e} cm^2/s")
print(f"  Re (box)    = {U_rms*L_box/nu_S:.2e}")
print(f"  Re (forcing)= {U_rms*L_force/nu_S:.2e}")
print()

# ── Numerical Reynolds number (Euler solver) ──
nu_num  = c_s * dx
Re_num  = U_rms * L_box / nu_num

print("Numerical (Euler ILES, c_s from solver):")
print(f"  nu_num ~ c_s*dx = {nu_num:.3e} cm^2/s")
print(f"  Re_num (box)    = {Re_num:.1f}  (M*N = {M*N_grid:.1f})")
print()

# ── Kolmogorov scale ──
Re_neutral = U_rms * L_box / nu_neutral
Re_spitzer = U_rms * L_box / nu_S
eta_neutral = L_box * Re_neutral**(-0.75)
eta_spitzer = L_box * Re_spitzer**(-0.75)

print("Kolmogorov scale:")
print(f"  eta (neutral) = {eta_neutral:.2e} cm   (eta/dx = {eta_neutral/dx:.2e})")
print(f"  eta (ionized) = {eta_spitzer:.2e} cm   (eta/dx = {eta_spitzer/dx:.2e})")
