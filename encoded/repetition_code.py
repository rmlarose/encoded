import cirq
import numpy as np
from typing import List

def encoding_repetition(state: str, n: int):
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()
    if state == "0":
        for q in qubits:
            circuit.append(cirq.I.on(q))
    else:
        for q in qubits:
            circuit.append(cirq.X.on(q))

    return circuit, qubits

def encoding_two_qubit(state1: str, state2: str, n: int):
    qubits = cirq.LineQubit.range(2*n)
    circuit = cirq.Circuit()
    if state1 == "0":
        for q in qubits[:n]:
            circuit.append(cirq.I.on(q))
    else:
        for q in qubits[:n]:
            circuit.append(cirq.X.on(q))

    if state2 == "0":
        for q in qubits[n:]:
            circuit.append(cirq.I.on(q))
    else:
        for q in qubits[n:]:
            circuit.append(cirq.X.on(q))

    return circuit, qubits

def logical_H(circuit: cirq.Circuit, qubits: List[cirq.Qid], n: int, qi: int):
    for i in range(n-1, 0, -1):
        circuit.append(cirq.CNOT.on(qubits[qi*n+i-1], qubits[qi*n+i]))

    circuit.append(cirq.H.on(qubits[qi*n]))

    for i in range(n-1):
        circuit.append(cirq.CNOT.on(qubits[qi*n+i], qubits[qi*n+i+1]))

def logical_Z(circuit: cirq.Circuit, qubits: List[cirq.Qid], n: int, qi: int):
    circuit.append(cirq.Z.on(qubits[qi*n]))

def logical_X(circuit: cirq.Circuit, qubits: List[cirq.Qid], n: int, qi: int):
    for i in range(n):
        circuit.append(cirq.X.on(qubits[qi*n+i]))

def logical_CNOT(circuit: cirq.Circuit, qubits: List[cirq.Qid], n: int, qi1: int, qi2: int):
    for i in range(n):
        circuit.append(cirq.CNOT.on(qubits[qi1*n+i], qubits[qi2*n+i]))

def logical_RZ(circuit: cirq.Circuit, qubits: List[cirq.Qid], n: int, qi: int, theta: float):
    circuit.append(cirq.rz(rads=theta).on(qubits[qi*n]))

def logical_RX(circuit: cirq.Circuit, qubits: List[cirq.Qid], n: int, qi: int, theta: float):
    logical_H(circuit, qubits, n, qi)
    logical_RZ(circuit, qubits, n, qi, theta)
    logical_H(circuit, qubits, n, qi)

def logical_RY(circuit: cirq.Circuit, qubits: List[cirq.Qid], n: int, qi: int, theta: float):
    logical_RZ(circuit, qubits, n, qi, -np.pi/2)
    logical_RX(circuit, qubits, n, qi, theta)
    logical_RZ(circuit, qubits, n, qi, np.pi/2)

def generate_stabilizers(n):
    stabilizers = []
    for i in range(n-1):
        stab = 1.
        for j in range(n):
            if 0 <= j-i < 2:
                stab = np.kron(stab,
                    [[1., 0.], [0., -1.]]
                )
            else:
                stab = np.kron(stab,
                    [[1., 0.], [0., 1.]]
                )
        stabilizers.append(stab)
    return stabilizers

def get_projector_mat(n):
    stabs = generate_stabilizers(n)
    pi = np.eye(2**n)
    for s in stabs:
        pi @= (np.eye(2**n) + s) / 2
    return pi