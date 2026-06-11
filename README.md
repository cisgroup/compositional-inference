# Subsystem Structure as an Inferential Resource for Coupled Engineered Systems

Code and supporting inputs for the paper:

Ghorbani, E. and Hackl, J. (2026). "Subsystem Structure as an Inferential Resource for Coupled Engineered Systems." arXiv preprint arXiv:2605.27544.

Please cite this work when using the code or results:

```bibtex
@article{ghorbani2026subsystem,
  title={Subsystem Structure as an Inferential Resource for Coupled Engineered Systems},
  author={Ghorbani, Esmaeil and Hackl, J{\"u}rgen},
  journal={arXiv preprint arXiv:2605.27544},
  year={2026}
}
```

## Setup

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run scripts from the repository root so the relative paths resolve correctly.

## Contents

- `src/`: shared filtering and sensitivity utilities.
- `data/`: input data files.
- `models/`: trained NARX model checkpoints.
- `results/`: saved result data used by optional post-processing helpers.
- `01-four-dof-ukf-sindy.py`: 4-DOF UKF and SINDy message-passing example.
- `02-six-dof-ukf.py`: 6-DOF distributed UKF example.
- `03-six-dof-uncertainty-propagation.py`: 6-DOF uncertainty-propagation example.
- `04-direct-problem-4dof.py`: forward 4-DOF direct problem comparing Jacobi, Gauss-Seidel, and AB2 coupling schemes.
- `05-central-vs-distributed-ukf.py`: generalized centralized vs distributed UKF benchmark.
- `06-kuramoto-ukf-wls-wnls.py`: Kuramoto/PYPOWER UKF, WLS, and WNLS benchmark.
- `07-ieee9-turbine-inverse-kf.py`: IEEE-9 turbine inverse Kalman-filter example.
- `08-Narx-training.ipynb`: NARX training notebook.
- `09-SoS-turbine-model.ipynb`: system-of-systems turbine model notebook.

## Run

Run any example directly:

```bash
python 01-four-dof-ukf-sindy.py
python 02-six-dof-ukf.py
python 03-six-dof-uncertainty-propagation.py
python 04-direct-problem-4dof.py
python 05-central-vs-distributed-ukf.py
python 06-kuramoto-ukf-wls-wnls.py
python 07-ieee9-turbine-inverse-kf.py
```

Open the notebooks with:

```bash
jupyter notebook 08-Narx-training.ipynb
jupyter notebook 09-SoS-turbine-model.ipynb
```

Some examples are computationally heavy and may take a long time to finish.
