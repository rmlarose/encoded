import cirq
from typing import Sequence
import stim
import stimcirq


import numpy as np
from typing import List


def parity_check_matrix_to_stabilizers(matrix: np.ndarray) -> List[stim.PauliString]:
    num_rows, num_cols = matrix.shape
    assert num_cols % 2 == 0
    num_qubits = num_cols // 2

    matrix = matrix.astype(np.bool8)
    return [
        stim.PauliString.from_numpy(
            xs=matrix[row, :num_qubits],
            zs=matrix[row, num_qubits:],
        )
        for row in range(num_rows)
    ]


def stabilizers_to_encoder(stabilizers) -> stim.Circuit:
    tableau = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_underconstrained=True,
    )
    return tableau.to_circuit(method="graph_state")


def encoding_steane(circuit, qreg):
    RX_stim = stimcirq.MeasureAndOrResetGate(
        measure=False, reset=True, basis="X", invert_measure=False, key="", measure_flip_probability=0
    )
    circuit.append(RX_stim.on_each(qreg))
    circuit.append(cirq.CZ(qreg[0], qreg[3]))
    circuit.append(cirq.CZ(qreg[0], qreg[4]))
    circuit.append(cirq.CZ(qreg[0], qreg[5]))
    circuit.append(cirq.CZ(qreg[1], qreg[3]))
    circuit.append(cirq.CZ(qreg[1], qreg[4]))
    circuit.append(cirq.CZ(qreg[1], qreg[6]))
    circuit.append(cirq.CZ(qreg[2], qreg[3]))
    circuit.append(cirq.CZ(qreg[2], qreg[5]))
    circuit.append(cirq.CZ(qreg[2], qreg[6]))
    circuit.append(cirq.H.on_each(qreg[3:7]))
    return circuit


def steane_H(qreg: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(qreg))
    return circuit


def steane_X(qreg: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    circuit.append(cirq.X.on_each(qreg))
    return circuit


def steane_CNOT(qreg1: Sequence[cirq.Qid], qreg2: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    for i in range(7):
        circuit.append(cirq.CNOT.on(qreg1[i], qreg2[i]))
    return circuit
