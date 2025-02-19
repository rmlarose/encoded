import cirq
from typing import Sequence


def encoding_steane(qreg: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    circuit = cirq.Circuit()

    circuit.append(cirq.H.on(qreg[0]))
    circuit.append(cirq.H.on(qreg[4]))
    circuit.append(cirq.H.on(qreg[6]))

    circuit.append(cirq.CNOT.on(qreg[0], qreg[1]))
    circuit.append(cirq.CNOT.on(qreg[4], qreg[5]))

    circuit.append(cirq.CNOT.on(qreg[6], qreg[3]))
    circuit.append(cirq.CNOT.on(qreg[6], qreg[5]))
    circuit.append(cirq.CNOT.on(qreg[4], qreg[2]))

    circuit.append(cirq.CNOT.on(qreg[0], qreg[3]))
    circuit.append(cirq.CNOT.on(qreg[4], qreg[1]))
    circuit.append(cirq.CNOT.on(qreg[3], qreg[2]))

    return circuit


def logical_H(qreg: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(qreg))
    return circuit


def logical_X(qreg: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    circuit.append(cirq.X.on_each(qreg))
    return circuit


def logical_CNOT(qreg1: Sequence[cirq.Qid], qreg2: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    for i in range(7):
        circuit.append(cirq.CNOT.on(qreg1[i], qreg2[i]))
    return circuit
