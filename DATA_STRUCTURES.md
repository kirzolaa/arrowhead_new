# Data Structures in Berry Phase Calculation

This document visualizes the array shapes and relationships between the key data structures used in the Berry phase calculations.

## Array Shapes and Dimensions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA STRUCTURE FLOWCHART                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐      ┌─────────────────────────────┐
│         PARAMETERS          │      │       HAMILTONIAN           │
│                             │      │                             │
│ theta_vals: (N,)            │─────▶│ H_thetas: (N, M, M)         │
│   N = num_points            │      │   N = num_points            │
│                             │      │   M = matrix_dimension (4)  │
│ R_thetas: (N, 3)            │─────▶│                             │
│   N = num_points            │      └─────────────────────────────┘
│   3 = spatial dimensions    │                    │
│                             │                    ▼
└─────────────────────────────┘      ┌──────────────────────────────┐
                                     │      EIGENDECOMPOSITION      │
                                     │                              │
                                     │ eigenvalues: (N, M)          │
                                     │   N = num_points             │
                                     │   M = num_states (4)         │
                                     │                              │
                                     │ eigenvectors: (N, M, M)      │
                                     │   N = num_points             │
                                     │   M = matrix_dimension (4)   │
                                     │   M = num_states (4)         │
                                     └──────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────┐      ┌──────────────────────────────┐
│    BERRY CONNECTION (TAU)    │      │      BERRY PHASE (GAMMA)     │
│                              │      │                              │
│ tau: (M, M, N)               │─────▶│ gamma: (M, M, N)             │
│   M = num_states (4)         │      │   M = num_states (4)         │
│   M = num_states (4)         │      │   M = num_states (4)         │
│   N = num_points             │      │   N = num_points             │
│                              │      │                              │
│ tau[n,m,i] = Berry connection│      │ gamma[n,m,i] = Accumulated   │
│   from state n to state m    │      │   Berry phase from state n   │
│   at point i                 │      │   to state m up to point i   │
└──────────────────────────────┘      └──────────────────────────────┘
                                                    │
                                                    ▼
                                     ┌─────────────────────────────┐
                                     │    FINAL BERRY PHASES       │
                                     │                             │
                                     │ berry_phases: (M,)          │
                                     │   M = num_states (4)        │
                                     │                             │
                                     │ berry_phase_matrix: (M, M)  │
                                     │   M = num_states (4)        │
                                     │   M = num_states (4)        │
                                     └─────────────────────────────┘
```

## Detailed Explanation

### Parameter Arrays

- **theta_vals**: 1D array of shape (N,) containing the discretized values of θ from θ_min to θ_max
- **R_thetas**: 2D array of shape (N, 3) containing the 3D parameter vectors R(θ) for each θ

### Hamiltonian

- **H_thetas**: 3D array of shape (N, M, M) containing the Hamiltonian matrix for each θ
  - N = number of points (theta values)
  - M = matrix dimension (4×4 arrowhead matrix)

### Eigendecomposition

- **eigenvalues**: 2D array of shape (N, M) containing the eigenvalues for each θ
  - N = number of points
  - M = number of states (4)

- **eigenvectors**: 3D array of shape (N, M, M) containing the eigenvectors for each θ
  - N = number of points
  - First M = matrix dimension (4)
  - Second M = number of states (4)
  - eigenvectors[i, :, j] = j-th eigenvector at θ_i

### Berry Connection (Tau Matrix)

- **tau**: 3D array of shape (M, M, N) containing the Berry connection matrix
  - First M = source state index
  - Second M = target state index
  - N = number of points
  - tau[n, m, i] = ⟨ψ_m(θ_i)|∂/∂θ|ψ_n(θ_i)⟩

### Berry Phase (Gamma Matrix)

- **gamma**: 3D array of shape (M, M, N) containing the accumulated Berry phase
  - First M = source state index
  - Second M = target state index
  - N = number of points
  - gamma[n, m, i] = ∫ tau[n, m, θ] dθ from θ_0 to θ_i

### Final Berry Phases

- **berry_phases**: 1D array of shape (M,) containing the Berry phase for each state
  - M = number of states (4)
  - berry_phases[n] = gamma[n, n, -1] (diagonal elements at the last point)

- **berry_phase_matrix**: 2D array of shape (M, M) containing the final Berry phase matrix
  - M = number of states (4)
  - berry_phase_matrix[n, m] = gamma[n, m, -1] (all elements at the last point)

## Code Implementation

### In `better_or_not_better_bph.py`:

```python
# Computing tau and gamma
tau, gamma = compute_berry_phase(eigvectors_all, theta_vals)
# tau.shape = (M, M, N)
# gamma.shape = (M, M, N)

# Final Berry phases from diagonal elements
berry_phases = np.zeros(M)
for n in range(M):
    berry_phases[n] = gamma[n, n, -1]
# berry_phases.shape = (M,)
```

### In `new_bph.py`:

```python
# NewBerryPhaseCalculator class
A_theta, dR_dtheta = self.calculate_berry_connection_theta_derivative()
# A_theta.shape = (N, M)
# dR_dtheta.shape = (N, 3)

# Berry phase calculation
berry_phases = np.zeros(M, dtype=float)
for n in range(M):
    phase = 0.0
    for i in range(num_points - 1):
        delta_theta = self.theta_range[i+1] - self.theta_range[i]
        phase += A_theta[i, n] * delta_theta
    berry_phases[n] = np.real(phase)
# berry_phases.shape = (M,)
```

### In `check_tau.py`:

```python
# Wilson loop method
berry_phases = np.zeros(M)
for n in range(M):
    phase_sum = 0.0
    for i in range(N):
        # Calculate phase from overlaps
        overlap = np.vdot(psi_i, psi_next)
        phase_diff = np.angle(overlap)
        phase_sum += phase_diff
    berry_phases[n] = phase_sum
# berry_phases.shape = (M,)
```
