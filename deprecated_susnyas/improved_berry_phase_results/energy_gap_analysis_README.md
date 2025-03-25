# Energy Gap Analysis for Berry Phase Systems

This extension to the Berry phase calculation system provides tools for analyzing energy gaps between eigenstates as a function of the adiabatic parameter θ. Understanding these energy gaps is crucial for:

1. **Assessing adiabaticity conditions**: Smaller gaps require slower parameter evolution to maintain adiabaticity
2. **Identifying potential degeneracies**: Points where gaps approach zero can indicate level crossings or avoided crossings
3. **Predicting topological transitions**: Gap closings often coincide with topological phase transitions
4. **Optimizing measurement protocols**: Parameter regions with larger gaps allow for faster, more robust measurements

## Features

- **Energy Spectrum Calculation**: Computes the full energy spectrum as a function of θ
- **Gap Analysis**: Calculates and analyzes gaps between adjacent energy levels
- **Minimum Gap Detection**: Identifies global and local minima in the gap function
- **Statistical Analysis**: Provides statistical measures for each gap (min, max, mean, std dev)
- **Visualization**: Creates comprehensive plots of the energy spectrum, gaps, and Berry phase accumulation
- **Reporting**: Generates detailed reports with analysis results and recommendations

## Usage

### Standalone Analysis

Run the energy gap analysis with custom parameters:

```bash
python energy_gap_analysis.py --c 0.2 --omega 1.0 --a 1.0 --b 0.5 --x_shift 0.2 --y_shift 0.2 --d 1.0 --num_points 1000 --output_dir energy_gap_analysis
```

### Optimal Configuration Analysis

Run the analysis with the optimal parameters that yield 0 parity flips in eigenstate 3:

```bash
python Arrowhead/run_optimal_gap_analysis.py
```

## Output Files

The analysis generates the following outputs in the specified directory:

### Plots
- `energy_spectrum.png`: Energy eigenvalues vs θ
- `energy_gaps.png`: Energy gaps between adjacent levels vs θ
- `comprehensive_analysis.png`: Combined visualization of spectrum, gaps, and Berry phases

### Results
- `gap_analysis_report.txt`: Detailed report with gap statistics and analysis
- `gap_analysis_data.npz`: Raw numerical data for further analysis
- `summary.txt`: Summary of key findings and recommendations

## Integration with Berry Phase Calculation

This analysis complements the Berry phase calculation by:

1. Providing insight into the adiabatic conditions necessary for accurate Berry phase measurement
2. Identifying parameter regions where the Berry phase calculation may be less reliable
3. Correlating gap features with topological properties observed in the Berry phase

## Parameters

- `c`: Fixed coupling constant for the Hamiltonian
- `omega`: Frequency parameter
- `a`, `b`: Coefficients for the potential functions
- `c_const`: Constant term in potentials
- `x_shift`, `y_shift`: Shifts for the Va potential
- `d`: Parameter for R_theta (radius of the circle in parameter space)
- `num_points`: Number of points for θ discretization

## Example Results

For the optimal configuration (0 parity flips in eigenstate 3), the analysis reveals:

- Minimum gap between states 2-3: ~0.05 at θ ≈ 3.14
- Correlation between minimum gaps and Berry phase accumulation rate
- Regions of parameter space where adiabaticity conditions are most stringent

## Further Development

Potential extensions to this analysis include:

1. Time-dependent simulations to assess diabatic transitions
2. Correlation analysis between gap size and Berry phase precision
3. Optimization algorithms to find parameter configurations with desired gap properties
4. Extension to higher-dimensional parameter spaces
