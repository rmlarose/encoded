import cirq
import numpy as np


def encoding_c4(
    state1: str, state2: str, circuit: cirq.Circuit, qubits: cirq.Qid
) -> None:
    """
    Encode four logical qubits into the C4 code (or [4,2,2] code) in the desired state space.
    Args:
        state1: initial state of the first logical qubit among 1 and 0 (string)
        state1: initial state of the second logical qubit among 1 and 0 (string)
        circuit: circuit to apply the encoding (cirq.Circuit)
        qubits: qubits to apply the encoding (cirq.Qid)

    """
    if state1 == "1":
        circuit.append(cirq.X(qubits[0]))
    if state2 == "1":
        circuit.append(cirq.X(qubits[1]))
    circuit.append(cirq.H(qubits[3]))
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))
    circuit.append(cirq.CNOT(qubits[3], qubits[1]))
    circuit.append(cirq.CNOT(qubits[1], qubits[2]))
    circuit.append(cirq.CNOT(qubits[3], qubits[0]))


def logical_X1(circuit: cirq.Circuit, qubits: cirq.Qid) -> None:
    """
    Logical X gate on the first qubit
    circuit: circuit to apply the logical operation (cirq.Circuit)
    qubits: qubits to apply the logical operation (cirq.Qid)
    """
    circuit.append(cirq.X(qubits[0]))
    circuit.append(cirq.X(qubits[2]))


def logical_X2(circuit: cirq.Circuit, qubits: cirq.Qid) -> None:
    """
    Logical X gate on the second qubit
    circuit: circuit to apply the logical operation (cirq.Circuit)
    qubits: qubits to apply the logical operation (cirq.Qid)
    """

    circuit.append(cirq.X(qubits[0]))
    circuit.append(cirq.X(qubits[1]))


def logical_Z1(circuit: cirq.Circuit, qubits: cirq.Qid) -> None:
    """
    Logical Z gate on the first qubit
    circuit: circuit to apply the logical operation (cirq.Circuit)
    qubits: qubits to apply the logical operation (cirq.Qid)
    """
    circuit.append(cirq.Z(qubits[0]))
    circuit.append(cirq.Z(qubits[1]))


def logical_Z2(circuit: cirq.Circuit, qubits: cirq.Qid) -> None:
    """
    Logical Z gate on the second qubit
    circuit: circuit to apply the logical operation (cirq.Circuit)
    qubits: qubits to apply the logical operation (cirq.Qid)
    """
    circuit.append(cirq.Z(qubits[0]))
    circuit.append(cirq.Z(qubits[2]))


def logical_RZ1(circuit: cirq.Circuit, qubits: cirq.Qid, theta: float) -> None:
    """Define the logical rotation on the z axis for the first logical qubit
    args:
        circuit: circuit to apply the logical gate (cirq.Circuit)
        qubits: qubits to apply the logical gate (cirq.Qid)
        theta: angle of rotation (float)
    """
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.rz(rads=theta).on(qubits[1]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))


def logical_RZ2(circuit: cirq.Circuit, qubits: cirq.Qid, theta: float) -> None:
    """Define the logical rotation on the z axis for the second logical qubit
    args:
        circuit: circuit to apply the logical gate (cirq.Circuit)
        qubits: qubits to apply the logical gate (cirq.Qid)
        theta: angle of rotation (float)
    """
    circuit.append(cirq.CNOT(qubits[1], qubits[3]))
    circuit.append(cirq.rz(rads=theta).on(qubits[3]))
    circuit.append(cirq.CNOT(qubits[1], qubits[3]))


def logical_RX1(circuit: cirq.Circuit, qubits: cirq.Qid, theta: float) -> float:
    """Define the logical rotation on the x axis for the first logical qubit
    args:
        circuit: circuit to apply the logical gate (cirq.Circuit)
        qubits: qubits to apply the logical gate (cirq.Qid)
        theta: angle of rotation (float)
    """

    circuit.append(cirq.CNOT(qubits[0], qubits[2]))
    circuit.append(cirq.rx(rads=theta).on(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))


def logical_RX2(circuit: cirq.Circuit, qubits: cirq.Qid, theta: float) -> None:
    """Define the logical rotation on the x axis for the second logical qubit
    args:
        circuit: circuit to apply the logical gate (cirq.Circuit)
        qubits: qubits to apply the logical gate (cirq.Qid)
        theta: angle of rotation (float)
    """
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))
    circuit.append(cirq.rx(rads=theta).on(qubits[1]))
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))


def logical_RY1(circuit: cirq.Circuit, qubits: cirq.Qid, theta: float) -> None:
    """Define the logical rotation on the y axis for the first logical qubit
    args:
        circuit: circuit to apply the logical gate (cirq.Circuit)
        qubits: qubits to apply the logical gate (cirq.Qid)
        theta: angle of rotation (float)
    """
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))
    circuit.append(cirq.CZ(qubits[0], qubits[1]))
    circuit.append(cirq.ry(rads=theta).on(qubits[0]))
    circuit.append(cirq.CZ(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))


def logical_RY2(circuit: cirq.Circuit, qubits: cirq.Qid, theta: float) -> None:
    """Define the logical rotation on the y axis for the second logical qubit
    args:
        circuit: circuit to apply the logical gate (cirq.Circuit)
        qubits: qubits to apply the logical gate (cirq.Qid)
        theta: angle of rotation (float)
    """
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))
    circuit.append(cirq.CZ(qubits[1], qubits[3]))
    circuit.append(cirq.ry(rads=theta).on(qubits[1]))
    circuit.append(cirq.CZ(qubits[1], qubits[3]))
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))


def logical_H1(circuit: cirq.Circuit, qubits: cirq.Qid) -> None:
    """
    Define the logical hadamard operator on the first logical qubit
    Args:
        circuit: circuit where the logical operation is defined (cirq.Circuit)
        qubits: qubits where the logical operation is defined (cirq.Circuit)
    """
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))
    circuit.append(cirq.X(qubits[0]))
    circuit.append(cirq.CZ(qubits[0], qubits[1]))
    circuit.append(cirq.ry(rads=-np.pi / 2).on(qubits[0]))
    circuit.append(cirq.CZ(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))


def logical_H2(circuit: cirq.Circuit, qubits: cirq.Qid) -> None:
    """
    Define the logical hadamard operator on the second logical qubit
    Args:
        circuit: circuit where the logical operation is defined (cirq.Circuit)
        qubits: qubits where the logical operation is defined (cirq.Circuit)
    """
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))
    circuit.append(cirq.X(qubits[1]))
    circuit.append(cirq.CZ(qubits[1], qubits[3]))
    circuit.append(cirq.ry(rads=-np.pi / 2).on(qubits[1]))
    circuit.append(cirq.CZ(qubits[1], qubits[3]))
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))


def logical_CNOT12(circuit: cirq.Circuit, qubits: cirq.Qid) -> None:
    """
    Define the logical CNOT operator with control on the first logical qubit and target on the second logical qubit
    Args:
        circuit: circuit where the logical operation is defined (cirq.Circuit)
        qubits: qubits where the logical operation is defined (cirq.Circuit)
    """
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))


def logical_CNOT21(circuit: cirq.Circuit, qubits: cirq.Qid) -> None:
    """
    Define the logical CNOT operator with control on the second logical qubit and target on the first logical qubit
    Args:
        circuit: circuit where the logical operation is defined (cirq.Circuit)
        qubits: qubits where the logical operation is defined (cirq.Circuit)
    """
    circuit.append(cirq.CNOT(qubits[3], qubits[1]))
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))
    circuit.append(cirq.CNOT(qubits[3], qubits[1]))
