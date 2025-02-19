import cirq
import qsimcirq
import numpy as np
from typing import Sequence, Dict


def dj(qreg: Sequence[cirq.Qid], oracleType: int, oracleValue: int) -> cirq.Circuit:
    """
    Args:
        qregs: set of N+1 qubits where the N first qubits represent the reqister for querying the oracle and the N+1-th qubit is the register for storing the answer of the oracle
    Returns:
        cirq.Circuit: quantum circuit for the Deutsch Jozsa algorithm
    """

    # implementing quantum electro
    circuit_dj = cirq.Circuit()
    n = len(qreg) - 1

    # initialization
    circuit_dj.append(cirq.H.on_each(qreg[:n]))
    circuit_dj.append(cirq.X(qreg[n]))
    circuit_dj.append(cirq.H(qreg[n]))

    # Oracle
    if oracleType == 0:  # If the oracleType is "0", the oracle returns oracleValue for all input.
        if oracleValue == 1:
            circuit_dj.append(cirq.X(qreg[n]))
    else:  # Otherwise, it returns the inner product of the input with a (non-zero bitstring)
        for i in range(n):
            if oracleValue & (1 << i):
                circuit_dj.append(cirq.CNOT.on(qreg[n - i - 1], qreg[n]))

    # finalization
    circuit_dj.append(cirq.H.on_each(qreg[:n]))

    return circuit_dj
