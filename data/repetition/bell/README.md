## Data format

The `raw/` directory contains data from the "raw" encoded experiment with no other error mitigation. The `dd` contains data from encoded + dynamical decoupling experiments.

Sub-directories, e.g. `ibm_kyiv`, indicate the quantum computer data was collected from.

Sub-sub-directories enumerate trials on that computer, e.g. `00` contains data from the first trial, `01` contains data from the second trial, etc.

There are four files for each trial:

1. `<BASE>_counts.txt`: The sampled bitstrings from the computer.
1. `<BASE>_expectation_values.txt`: The expectation values computed from the bitstrings.
1. `<BASE>_qubits.txt`: The qubits used in the experiment.
1. `<BASE>.pdf`: A plot of the expectation values.

The `BASE` prefix specifies (in this order):

- The computer used
- A descriptor/name for the experiment
- The number of qubits used
- The depth d, meaning the (U^dag U)^d U is performed where U is the state preparation circuit
- The date and time the data was collected
- The job id for the unencoded (i.e., n = 1) experiment
- The batch job id for the encoded experiments (e.g., n = 3, 5, 7)

## Plotting

To visualize data, select `experiment(s)` and `computer(s)` in `plot.ipynb` and run the notebook.
