import numpy as np
import matplotlib.pyplot as plt
import os

# ── Mixing angles ─────────────────────────────────────────────────────────────
t12 = np.deg2rad(33.41)
t13 = np.deg2rad(8.54)
t23 = np.deg2rad(49.1)

# ── Wave-packet parameters ────────────────────────────────────────────────────
sigma = 1e-5   # km
eps   = 0.0    # dimensionless

# ── Mass-squared differences (eV²) ───────────────────────────────────────────
delta12 = 7.42e-5
delta13 = 2.51e-3
delta23 = delta13 - delta12

# ── Fixed baseline, energy array ─────────────────────────────────────────────
L        = 1300.0                         # km  (fixed)
E_GeV    = np.linspace(0.1, 5.0, 5000)   # GeV (start at 0.1 to avoid 1/E → ∞)
E_eV_arr = E_GeV * 1e9                   # eV  (array)

# ── Unit conversion ───────────────────────────────────────────────────────────
km_to_iEV = 5.06773e9          # 1 km = 5.06773e9 eV^{-1}
L_iEV     = L     * km_to_iEV  # scalar  eV^{-1}
sigma_iEV = sigma * km_to_iEV  # scalar  eV^{-1}

# ── Matter potential ──────────────────────────────────────────────────────────
Ye      = 0.5
rho_avg = 2.848   # g/cm³  DUNE average


# ── Vacuum PMNS matrix ────────────────────────────────────────────────────────
def U_matrix(CP):
    s12, s23, s13 = np.sin(t12), np.sin(t23), np.sin(t13)
    c12, c23, c13 = np.cos(t12), np.cos(t23), np.cos(t13)
    eid = np.exp(1j * CP)
    return np.array([
        [ c12*c13,                        s12*c13,                        s13/eid  ],
        [-s12*c23 - c12*s23*s13*eid,       c12*c23 - s12*s23*s13*eid,       s23*c13  ],
        [ s12*s23 - c12*c23*s13*eid,      -c12*s23 - s12*c23*s13*eid,       c23*c13  ]
    ], dtype=complex)

# ── Matter diagonalisation (called per energy point) ─────────────────────────
def get_W_and_lambdas(CP, E_eV_val):
    U = U_matrix(CP)
    V = 7.56e-14 * rho_avg * Ye                    # eV  (independent of E)

    H_vac = U @ np.diag([0.0,
                          delta12 / (2.0 * E_eV_val),
                          delta13 / (2.0 * E_eV_val)]) @ U.conj().T

    H_matter = H_vac + np.diag([V, 0.0, 0.0])
    lambdas, W = np.linalg.eigh(H_matter)          # eigenvalues in eV, sorted
    return W, lambdas

# ── CP phases ────────────────────────────────────────────────────────────────
CP_degrees = [180]
CP_phases  = np.deg2rad(CP_degrees)

# ── Main loop ─────────────────────────────────────────────────────────────────
for CP in CP_phases:

    P_emu = np.zeros(len(E_GeV))
    P_eta = np.zeros(len(E_GeV))
    P_ee  = np.zeros(len(E_GeV))

    for k, E_eV_val in enumerate(E_eV_arr):

        W, lambdas = get_W_and_lambdas(CP, E_eV_val)

        dl_01 = lambdas[1] - lambdas[0]   # eV
        dl_12 = lambdas[2] - lambdas[1]   # eV
        dl_02 = lambdas[2] - lambdas[0]   # eV

        # Phases  [dimensionless] = eV * eV^{-1}
        ph_01 = dl_01 * L_iEV
        ph_12 = dl_12 * L_iEV
        ph_02 = dl_02 * L_iEV

        # Oscillation lengths  [eV^{-1}]
        Lo_01 = 2.0 * np.pi / dl_01
        Lo_12 = 2.0 * np.pi / dl_12
        Lo_02 = 2.0 * np.pi / dl_02

        # Coherence lengths  [eV^{-1}]
        Lc_01 = np.sqrt(2.0) * E_eV_val**2 * sigma_iEV / dl_01**2
        Lc_12 = np.sqrt(2.0) * E_eV_val**2 * sigma_iEV / dl_12**2
        Lc_02 = np.sqrt(2.0) * E_eV_val**2 * sigma_iEV / dl_02**2

        # Damping exponentials  [dimensionless]
        def damp(Lc, Lo):
            return np.exp(-(L_iEV / Lc)**2
                          - 2.0 * np.pi**2 * eps**2 * (sigma_iEV / Lo)**2)

        d01 = damp(Lc_01, Lo_01)
        d12 = damp(Lc_12, Lo_12)
        d02 = damp(Lc_02, Lo_02)

        # W combinations — νe → νμ
        W1m = W[0,1].conj()*W[1,1]*W[0,0]*W[1,0].conj()
        W2m = W[0,2].conj()*W[1,2]*W[0,1]*W[1,1].conj()
        W3m = W[0,2].conj()*W[1,2]*W[0,0]*W[1,0].conj()

        # W combinations — νe → ντ
        W1t = W[0,1].conj()*W[2,1]*W[0,0]*W[2,0].conj()
        W2t = W[0,2].conj()*W[2,2]*W[0,1]*W[2,1].conj()
        W3t = W[0,2].conj()*W[2,2]*W[0,0]*W[2,0].conj()

        # Constant (decoherence-averaged) terms
        cm = (W[0,0]*W[0,0].conj()*W[1,0]*W[1,0].conj()
            + W[0,1]*W[0,1].conj()*W[1,1]*W[1,1].conj()
            + W[0,2]*W[0,2].conj()*W[1,2]*W[1,2].conj()).real

        ct = (W[0,0]*W[0,0].conj()*W[2,0]*W[2,0].conj()
            + W[0,1]*W[0,1].conj()*W[2,1]*W[2,1].conj()
            + W[0,2]*W[0,2].conj()*W[2,2]*W[2,2].conj()).real

        # ── P(νe → νμ) ───────────────────────────────────────────────────────
        P_emu[k] = np.clip((
            cm
            + 2*(W1m.real*np.cos(ph_01)*d01 + W2m.real*np.cos(ph_12)*d12 + W3m.real*np.cos(ph_02)*d02)
            + 2*(W1m.imag*np.sin(ph_01)*d01 + W2m.imag*np.sin(ph_12)*d12 + W3m.imag*np.sin(ph_02)*d02)
        ))

        # ── P(νe → ντ) ───────────────────────────────────────────────────────
        P_eta[k] = np.clip((
            ct
            + 2*(W1t.real*np.cos(ph_01)*d01 + W2t.real*np.cos(ph_12)*d12 + W3t.real*np.cos(ph_02)*d02)
            + 2*(W1t.imag*np.sin(ph_01)*d01 + W2t.imag*np.sin(ph_12)*d12 + W3t.imag*np.sin(ph_02)*d02)
        ))

        # ── P(νe → νe) survival ──────────────────────────────────────────────
        P_ee[k] = np.clip(1 - P_emu[k] - P_eta[k], 0, 1)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r"$E_\nu$ (GeV)", fontsize=13)
    plt.ylabel("Probability", fontsize=13)
    plt.plot(E_GeV, P_emu, label=r"$\nu_e \rightarrow \nu_\mu$",  lw=1.5)
    plt.plot(E_GeV, P_ee,  label=r"$\nu_e \rightarrow \nu_e$",    lw=1.5)
    plt.plot(E_GeV, P_eta, label=r"$\nu_e \rightarrow \nu_\tau$", lw=1.5)
    plt.title(rf"CP = {np.rad2deg(CP):.0f}° | Matter (Wave-Packet) | $L$ = {L:.0f} km",
              fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

print("Done.")
