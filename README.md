# Monte Carlo Simulation of the 2D Ising Model

This repository contains a small computational physics project for simulating the two-dimensional Ising model with Monte Carlo methods. The project was written as part of a physics report on phase transitions, critical phenomena, and ferromagnetism.

The main goal is to study how the average magnetization of a spin lattice changes with temperature, lattice size, Monte Carlo iterations, coupling strength, and an optional external magnetic field.

## Project overview

In the Ising model, each lattice site contains a discrete spin variable:

```text
s_i = +1 or -1
```

The spins interact with their nearest neighbors. In the field-free version, the Hamiltonian is proportional to the nearest-neighbor interaction energy. In the version with an external magnetic field, the model also includes a field contribution controlled by `HKb`.

The simulations use random spin updates and a Metropolis-style acceptance rule to evolve the lattice at a given temperature. The main observable computed by the code is the average magnetization per lattice site:

```text
M = (sum of all spins) / N^2
```

where `N` is the lattice dimension of an `N x N` square lattice.

## Main features

- Simulates the 2D Ising model on a square lattice.
- Uses periodic boundary conditions for nearest-neighbor interactions.
- Computes lattice magnetization and lattice energy.
- Runs Monte Carlo spin-flip simulations at fixed temperature.
- Sweeps over temperature to generate magnetization-versus-temperature plots.
- Supports a modified Hamiltonian with an external field.
- Includes utilities for runtime comparisons against lattice size, temperature, and Monte Carlo iteration count.
- Includes an animation-oriented version for visualizing the Monte Carlo evolution.

## Repository structure

```text
.
├── Ising_Model.tex                  # Written report / theoretical background
├── IsingModel2D.py                  # Original 2D Ising model implementation
├── IsingModel2D_H.py                # Main implementation with external field H
├── IsingModel2D_H_animation.py      # Animation-focused simulation script
├── animation.py                     # Standalone animation helper
├── simulations.py                   # Helper experiments and parameter sweeps
└── README.md
```

## File descriptions

### `Ising_Model.tex`

LaTeX report explaining the theoretical background of the project. It covers phase transitions, the Ising model, the one-dimensional exact solution, the Monte Carlo method, runtime behavior, results, and conclusions.

### `IsingModel2D.py`

Original implementation of the two-dimensional Ising model without an explicit external field term. It defines an `isingModel` class with methods for:

- setting the initial lattice state,
- computing nearest-neighbor energy,
- computing lattice energy,
- computing average magnetization,
- running the Monte Carlo simulation,
- plotting magnetization versus temperature.

This file currently contains executable test/demo code at the bottom, so running it directly will execute a sample simulation and save a plot.

### `IsingModel2D_H.py`

Recommended main implementation. This is a modified and more flexible version of `IsingModel2D.py`. It includes:

- NumPy-based lattice representation,
- optional external magnetic field parameter `HKb`,
- configurable initial state,
- critical temperature marker option,
- magnetization-versus-temperature plotting.

The critical temperature used in the plot option is computed as:

```text
Tc = 2J / ln(1 + sqrt(2))
```

where `J` is represented in the code by `JKb`.

### `simulations.py`

Utility functions for running parameter studies:

- `time_size_simulation(...)`: measures runtime and final magnetization as lattice size changes.
- `temperature_simulation(...)`: measures runtime and magnetization over a temperature range.
- `MC_iterations_simulation(...)`: measures runtime and magnetization as the number of Monte Carlo iterations changes.

This file also includes several commented example simulations for different values of `J`, `H`, temperature range, lattice size, and initial magnetization.

### `IsingModel2D_H_animation.py`

Animation-oriented version of the Ising model simulation. It stores snapshots of the lattice while sweeping temperature and can export an MP4 animation of the Monte Carlo evolution.

### `animation.py`

Standalone animation helper that assumes a `snapshots` variable already exists in memory. This file is useful as a prototype, but it is not currently self-contained because `snapshots` is not defined inside the script.

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy matplotlib
```

For MP4 animation export, install `ffmpeg` as well.

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

## Basic usage

The recommended entry point is `IsingModel2D_H.py`.

Example: run a simulation with no external field and plot magnetization versus temperature.

```python
from IsingModel2D_H import isingModel

simulation = isingModel(
    N=50,
    iterations=10000,
    minTemp=0.1,
    maxTemp=10,
    JKb=1,
    HKb=0,
    initial_state=-1.0,
    simulation_name="example_no_field",
    plotTc=True,
)

simulation.plotMvrsT()
```

This saves a plot with a name similar to:

```text
PlotMvT_example_no_field.png
```

## Example with external magnetic field

```python
from IsingModel2D_H import isingModel

simulation = isingModel(
    N=50,
    iterations=10000,
    minTemp=0.1,
    maxTemp=10,
    JKb=1,
    HKb=1,
    initial_state=1.0,
    simulation_name="example_field",
    plotTc=True,
)

simulation.plotMvrsT()
```

Changing `HKb` allows you to explore how the external field biases the system toward one magnetization direction.

## Running parameter studies

The `simulations.py` file provides helper functions for experiments.

### Runtime versus lattice size

```python
from simulations import time_size_simulation
import matplotlib.pyplot as plt

sizes, times, magnetizations = time_size_simulation(
    min_lattice_size=10,
    max_lattice_size=500,
    step=10,
    iterations=10000,
    fixed_T=1,
    JKb=1,
    HKb=0,
)

plt.xlabel("Lattice size N")
plt.ylabel("Computation time")
plt.plot(sizes, times)
plt.savefig("time_vs_lattice_size.png")
```

### Magnetization versus temperature

```python
from simulations import temperature_simulation
import matplotlib.pyplot as plt

temperatures, times, magnetizations = temperature_simulation(
    N=50,
    num_steps=1000,
    iterations=10000,
    minTemp=0.1,
    maxTemp=10,
    JKb=1,
    HKb=0,
)

plt.xlabel("Temperature")
plt.ylabel("Average magnetization per site")
plt.plot(temperatures, magnetizations)
plt.savefig("magnetization_vs_temperature.png")
```

### Runtime versus Monte Carlo iterations

```python
from simulations import MC_iterations_simulation
import matplotlib.pyplot as plt

iterations, times, magnetizations = MC_iterations_simulation(
    N=50,
    min_iterations=100,
    max_iterations=10000,
    T=5,
    JKb=1,
    HKb=0,
)

plt.xlabel("Monte Carlo iterations")
plt.ylabel("Computation time")
plt.plot(list(iterations), times)
plt.savefig("time_vs_iterations.png")
```

## Expected outputs

Depending on the script, the project can generate:

- `PlotMvT.png`
- `PlotMvT_<simulation_name>.png`
- `PlotMvT_v04.png`
- `time_vs_lattice_size.png`
- `magnetization_vs_temperature.png`
- `time_vs_iterations.png`
- `MC_animation.mp4`
- `MC_animation_v04.mp4`

Generated images and videos should normally be kept out of version control unless they are selected examples for the report or project documentation.

A recommended `.gitignore` entry is:

```gitignore
__pycache__/
*.pyc
.venv/
PlotMvT*.png
*.mp4
```

## Notes on the current code

The current code is suitable for experimentation and report reproduction, but some scripts mix reusable class definitions with demo code. For a cleaner GitHub project, it would be better to move demo runs into separate scripts or notebooks.

Recommended future structure:

```text
.
├── src/
│   └── ising/
│       ├── model.py
│       └── animation.py
├── scripts/
│   ├── run_temperature_sweep.py
│   ├── run_size_benchmark.py
│   └── run_animation.py
├── reports/
│   └── Ising_Model.tex
├── figures/
├── README.md
└── requirements.txt
```

## Scientific context

The report discusses the Ising model as an archetypal model for ferromagnetism and continuous phase transitions. The numerical experiments reproduce the qualitative transition from an ordered ferromagnetic phase at low temperature to a disordered paramagnetic phase at high temperature.

The simulations also illustrate finite-size effects: smaller lattices show stronger fluctuations, while larger lattices give smoother behavior closer to the thermodynamic-limit expectation.

## Limitations

- The current Monte Carlo implementation is simple and educational rather than optimized for large-scale simulations.
- Some scripts execute simulations immediately when imported or run.
- The animation helper `animation.py` is not self-contained because it assumes `snapshots` already exists.
- The code does not currently include automated tests.
- Random seeds are not fixed, so exact numerical results may vary between runs.
- Large values such as `N=5000` or millions of iterations may be computationally expensive on a normal laptop.

## Suggested improvements

- Add a `requirements.txt` file.
- Add a command-line interface for choosing `N`, temperature range, `J`, `H`, and iterations.
- Move demo code under `if __name__ == "__main__":`.
- Add reproducibility through optional random seeds.
- Add automated tests for energy, magnetization, boundary conditions, and spin-flip acceptance.
- Save outputs into organized folders such as `figures/` and `animations/`.
- Refactor the code into a package under `src/`.
- Add example figures to the README once final plots are selected.

## Author

**BSc. Julio A. Medina**  
University of San Carlos of Guatemala  
School of Physical Sciences and Mathematics  
Master's Program in Physics
## License

No license file is currently included. 
