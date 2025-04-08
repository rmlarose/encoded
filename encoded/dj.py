import cirq
from typing import Sequence, List
from encoded.steane import encoding_steane, steane_H, steane_X, steane_CNOT
from encoded.repetition_code import encoding_repetition, repetition_H, repetition_X, repetition_CNOT
from encoded.tcc import tcc_encoding, tcc_H, tcc_X, tcc_CNOT


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
        encoding_steane(circuit_dj, qreg[i * 7 : 7 * (i + 1)])

    circuit_dj.append(steane_H(qreg[: 7 * k]))
    circuit_dj.append(steane_X(qreg[7 * k : 7 * (k + 1)]))
    circuit_dj.append(steane_H(qreg[7 * k : 7 * (k + 1)]))

    # Oracle
    if oracleType == 0:  # If the oracleType is "0", the oracle returns oracleValue for all input.
        if oracleValue == 1:
            circuit_dj.append(steane_X(qreg[7 * k : 7 * (k + 1)]))
    else:  # Otherwise, it returns the inner product of the input with a (non-zero bitstring)
        for i in range(7 * k):
            if oracleValue & (1 << i):
                circuit_dj.append(steane_CNOT(qreg[7 * i : 7 * (i + 1)], qreg[7 * k : 7 * (k + 1)]))

    # finalization
    circuit_dj.append(steane_H(qreg[: 7 * k]))

    return circuit_dj


def dj_tcc(qreg: Sequence[cirq.Qid], distance, n_encoding: int, oracleType: int, oracleValue: int) -> cirq.Circuit:
    """
    Implementation of the Deutsch Jozsa algorithm using the Steane code for encoding a logical qubit.
    Args:
        - qregs: set of n_encoding*(k+1) qubits for k logical qubits representing the register for querying the oracle and the last logical qubit is the register for storing the answer of the oracle
        - distance: distance of the TCC code
        - n_encoding: the number of qubits in the TCC code
        - oracleType: type of oracle to be used. If oracleType is "0", the oracle is unbalanced otherwise it is balanced
        - oracleValue: value to be returned by the oracle if oracleType is "1"

    Returns:
        - cirq.Circuit: quantum circuit for the Deutsch Jozsa algorithm encoded in the steane code
    """

    # implementing quantum electro
    circuit_dj = cirq.Circuit()
    k = len(qreg) // n_encoding - 1

    tmp_circuit = tcc_encoding(distance)
    # initialization
    for i in range(k + 1):
        circuit_dj = cirq.Circuit.concat_ragged(
            circuit_dj,
            tmp_circuit.transform_qubits(
                dict(zip(qreg[:n_encoding], qreg[(i + 1) * n_encoding : (i + 2) * n_encoding]))
            ),
        )
    circuit_dj.append(tcc_H(qreg[: n_encoding * k]))
    circuit_dj.append(tcc_X(qreg[n_encoding * k : n_encoding * (k + 1)]))
    circuit_dj.append(tcc_H(qreg[n_encoding * k : n_encoding * (k + 1)]))

    # Oracle
    if oracleType == 0:  # If the oracleType is "0", the oracle returns oracleValue for all input.
        if oracleValue == 1:
            circuit_dj.append(tcc_X(qreg[n_encoding * k : n_encoding * (k + 1)]))
    else:  # Otherwise, it returns the inner product of the input with a (non-zero bitstring)
        for i in range(n_encoding * k):
            if oracleValue & (1 << i):
                circuit_dj.append(
                    tcc_CNOT(qreg[n_encoding * i : n_encoding * (i + 1)], qreg[n_encoding * k : n_encoding * (k + 1)])
                )

    # finalization
    circuit_dj.append(tcc_H(qreg[: n_encoding * k]))

    return circuit_dj


def dj_repetition(qreg: Sequence[cirq.Qid], n_encoding: int, oracleType: int, oracleValue: int) -> cirq.Circuit:
    """
    Implementation of the Deutsch Jozsa algorithm using the repetition code for encoding a logical qubit. This aims to perform a scaling experiment.
    Args:
        - qregs: set of n_encoding*(k+1) qubits for k logical qubits representing the register for querying the oracle and the last logical qubit is the register for storing the answer of the oracle.
        - n_encoding: the number of qubits in the repetition code
        - oracleType: type of oracle to be used. If oracleType is "0", the oracle is unbalanced otherwise it is balanced
        - oracleValue: value to be returned by the oracle if oracleType is "1"
    Returns:
        - cirq.Circuit: quantum circuit for the Deutsch Jozsa algorithm encoded in the repetition code
    """
    # implementing quantum electro
    circuit_dj = cirq.Circuit()
    k = len(qreg) // n_encoding - 1

    # initialization
    # for i in range(k + 1):
    #     circuit_dj.append(encoding_repetition("0", qreg[i * n_encoding : n_encoding * (i + 1)]))
    for i in range(k):
        repetition_H_half(circuit_dj, qreg, n_encoding, i)
    #
    repetition_H_half(circuit_dj, qreg, n_encoding, k)
    # circuit_dj.append(cirq.Z.on(qreg[k * n_encoding]))

    if oracleType == 0:  # If the oracleType is "0", the oracle returns oracleValue for all input.
        if oracleValue == 1:
            print()
            repetition_X(circuit_dj, qreg, n_encoding, k)
    else:  # Otherwise, it returns the inner product of the input with a (non-zero bitstring)
        for i in range(n_encoding * k):
            if oracleValue & (1 << i):
                repetition_CNOT(circuit_dj, qreg, n_encoding, i, k)

    # finalization
    for i in range(k):
        repetition_H(circuit_dj, qreg, n_encoding, i)
    repetition_X(circuit_dj, qreg, n_encoding, 0)
    return circuit_dj


def repetition_H_half(circuit: cirq.Circuit, qubits: List[cirq.Qid], n: int, qi: int):
    """
    logical hadamard gate opf the repetition code without the first row of CNOT. This is not corresponding to the true logical hadamard gate of the repetition code but this can be used for simplifying the circuit depth.

    """
    circuit.append(cirq.H.on(qubits[qi * n]))
    for i in range(n - 1):
        circuit.append(cirq.CNOT.on(qubits[qi * n + i], qubits[qi * n + i + 1]))
