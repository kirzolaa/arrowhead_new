import numpy as np
import os
import matplotlib.pyplot as plt
from new_bph import Hamiltonian  # This must be defined locally on your machine

def find_exact_symmetric_triple_d(
    d_vals, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-6, save_plots=True
):
    theta_triple = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])

    for d in d_vals:
        h = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_triple)
        R_thetas = np.array(h.R_thetas())
        Va_vals = np.array(h.Va_theta_vals(R_thetas))  # shape (3, 3)
        Vx_vals = np.array(h.Vx_theta_vals(R_thetas))  # shape (3, 3)

        delta_Vs = Va_vals - Vx_vals  # shape (3, 3)

        # Check component-wise equality across 3 symmetric θ-points
        componentwise_equal = [
            np.allclose(delta_Vs[:, i], delta_Vs[0, i], atol=epsilon)
            for i in range(3)
        ]

        if all(componentwise_equal):
            print(f"🎯 Exact triple-point degeneracy found for d = {d:.6f}")
            print("ΔV values at θ = 0, 2π/3, 4π/3:")
            print(delta_Vs)

            if save_plots:
                plot_dir = f"plots_triple_d_{d:.6f}"
                os.makedirs(plot_dir, exist_ok=True)

                # Full sweep for plotting
                theta_full = np.linspace(0, 2*np.pi, 1000)
                h_full = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_full)
                R_full = np.array(h_full.R_thetas())
                Va_full = np.array(h_full.Va_theta_vals(R_full))
                Vx_full = np.array(h_full.Vx_theta_vals(R_full))
                delta_full = Va_full - Vx_full

                # Plot ΔV_i(θ)
                plt.figure(figsize=(10, 5))
                for i in range(3):
                    plt.plot(theta_full, delta_full[:, i], label=fr"$\Delta V_{{{i+1}}}$")
                for θ in theta_triple:
                    plt.axvline(θ, color='k', linestyle='--', alpha=0.3)
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$V_a - V_x$")
                plt.title(f"ΔV components at d = {d:.6f}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"deltaV_full_{d:.6f}.png"))
                plt.close()

d_range = np.linspace(0.001, 0.5, 10000, endpoint=True)
find_exact_symmetric_triple_d(d_range, aVx=1.0, aVa=5.0, x_shift=0.01, c_const=0.01, omega=0.1, R_0=(0, 0, 0))
