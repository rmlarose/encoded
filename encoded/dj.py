import cirq
from typing import Sequence
from encoded.steane import encoding_steane, logical_H, logical_X, logical_CNOT


def dj(qreg: Sequence[cirq.Qid], oracleType: int, oracleValue: int) -> cirq.Circuit:
    """
    Args:
        - qregs: set of N+1 qubits where the N first qubits represent the reqister for querying the oracle and the N+1-th qubit is the register for storing the answer of the oracle
        - oracleType: type of oracle to be used. If oracleType is "0", the oracle is unbalanced otherwise it is balanced
        - oracleValue: value to be returned by the oracle if oracleType is "1"
    Returns:
        - cirq.Circuit: quantum circuit for the Deutsch Jozsa algorithm
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


def dj_steane(qreg: Sequence[cirq.Qid], oracleType: int, oracleValue: int) -> cirq.Circuit:
    """
    Implementation of the Deutsch Jozsa algorithm using the Steane code for encoding a logical qubit.
    Args:
        - qregs: set of 7*(k+1) qubits for k logical qubits representing the register for querying the oracle and the last logical qubit is the register for storing the answer of the oracle
        - oracleType: type of oracle to be used. If oracleType is "0", the oracle is unbalanced otherwise it is balanced
        - oracleValue: value to be returned by the oracle if oracleType is "1"
    Returns:
        - cirq.Circuit: quantum circuit for the Deutsch Jozsa algorithm encoded in the steane code
    """

    # implementing quantum electro
    circuit_dj = cirq.Circuit()
    k = len(qreg) // 7 - 1

    # initialization
    for i in range(k + 1):
        circuit_dj.append(encoding_steane(qreg[i * 7 : 7 * (i + 1)]))

    circuit_dj.append(logical_H(qreg[: 7 * k]))
    circuit_dj.append(logical_X(qreg[7 * k : 7 * (k + 1)]))
    circuit_dj.append(logical_H(qreg[7 * k : 7 * (k + 1)]))

    # Oracle
    if oracleType == 0:  # If the oracleType is "0", the oracle returns oracleValue for all input.
        if oracleValue == 1:
            circuit_dj.append(logical_X(qreg[7 * k : 7 * (k + 1)]))
    else:  # Otherwise, it returns the inner product of the input with a (non-zero bitstring)
        for i in range(7 * k):
            if oracleValue & (1 << i):
                circuit_dj.append(logical_CNOT(qreg[7 * i : 7 * (i + 1)], qreg[7 * k : 7 * (k + 1)]))

    # finalization
    circuit_dj.append(logical_H(qreg[: 7 * k]))

    return circuit_dj
