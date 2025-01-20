import cirq
import numpy as np


def encoding_single_qubit(state: str, qubits: list, start_index: int) -> cirq.circuits:
    """Encode five physical qubits into one logical qubit in the [5,1,3] code

    Args:
       circuit: The desired circuit to encode cirq.circuits.
       state: The logical state this will be encoded into.
       qubits: the list of qubits used by the circuit.
       start_index: the starting index where the logical qubit will be encoded. The physical qubits used for the encoding will be in [start_index,start_index+4].

    Return:
        The encoding circuit for the [5,1,3]
    """

    sdg = cirq.S**-1
    circuit = cirq.Circuit()
    if state == "1":
        circuit.append(cirq.X(qubits[start_index + 0]))
    circuit.append(cirq.Z(qubits[start_index + 0]))
    circuit.append(cirq.H(qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 3]))
    circuit.append(sdg(qubits[start_index + 0]))
    circuit.append(cirq.CNOT(qubits[start_index + 2], qubits[start_index + 4]))
    circuit.append(cirq.CNOT(qubits[start_index + 3], qubits[start_index + 1]))
    circuit.append(cirq.CNOT(qubits[start_index + 3], qubits[start_index + 4]))
    circuit.append(cirq.H(qubits[start_index + 1]))
    circuit.append(sdg(qubits[start_index + 2]))
    circuit.append(sdg(qubits[start_index + 4]))
    circuit.append(cirq.S(qubits[start_index + 3]))
    circuit.append(cirq.CNOT(qubits[start_index + 1], qubits[start_index + 0]))

    circuit.append(cirq.S(qubits[start_index + 0]))
    circuit.append(cirq.S(qubits[start_index + 1]))
    circuit.append(cirq.Z(qubits[start_index + 2]))
    circuit.append(cirq.CNOT(qubits[start_index + 4], qubits[start_index + 0]))
    circuit.append(cirq.H(qubits[start_index + 4]))
    circuit.append(cirq.CNOT(qubits[start_index + 4], qubits[start_index + 1]))

    return circuit


def encoding_two_qubit(
    state1: str, state2: str, qubits: list, start_index: int
) -> cirq.circuits:
    """Encode five physical qubits into two logical qubit in the [5,1,3] code

    Args:
       circuit: The desired circuit to encode cirq.circuits.
       state1: First logical state this will be encoded into.
       state2: Second logical state this will be encoded into.
       qubits: the list of qubits used by the circuit.
       start_index: the starting index where the logical qubit will be encoded. The physical qubits used for the encoding will be in [start_index,start_index+4].
       Return:
        The encoding circuit for the two concatenated code in the [5,1,3]
    """
    circuit1 = encoding_single_qubit(state1, qubits, start_index=0)
    circuit2 = encoding_single_qubit(state2, qubits, start_index=5)
    return circuit1 + circuit2


def logical_SH(circuit: cirq.circuits, qubits: list, start_index: int):
    """Prepare the logical SH gate schedule
    Args:
       circuit: The desired circuit to encode cirq.circuits.
       qubits: the list of qubits used by the circuit.
       start_index: the starting index where the logical gate will be encoded. The physical qubits used for the encoding will be in [start_index,start_index+4].
    """

    for i in range(start_index, start_index + 5):
        circuit.append(cirq.H(qubits[i]))
    for i in range(start_index, start_index + 5):
        circuit.append(cirq.S(qubits[i]))


def logical_SH_dag(circuit: cirq.circuits, qubits: list, start_index: int):
    """Prepare the logical SH_dag gate schedule
    Args:
       circuit: The desired circuit to encode cirq.circuits.
       qubits: the list of qubits used by the circuit.
       start_index: the starting index where the logical gate will be encoded. The physical qubits used for the encoding will be in [start_index,start_index+4].
    """
    sdg = cirq.S**-1
    for i in range(start_index, start_index + 5):
        circuit.append(sdg(qubits[i]))
    for i in range(start_index, start_index + 5):
        circuit.append(cirq.H(qubits[i]))


def logical_H(circuit: cirq.circuits, qubits: list, start_index: int):
    """Prepare the logical H gate schedule
    Args:
       circuit: The desired circuit to encode cirq.circuits.
       qubits: the list of qubits used by the circuit.
       start_index: the starting index where the logical gate will be encoded. The physical qubits used for the encoding will be in [start_index,start_index+4].
    """
    for i in range(start_index + 5):
        circuit.append(cirq.H(qubits[i]))
    circuit.append(cirq.SWAP(qubits[start_index + 0], qubits[start_index + 1]))
    circuit.append(cirq.SWAP(qubits[start_index + 1], qubits[start_index + 4]))
    circuit.append(cirq.SWAP(qubits[start_index + 4], qubits[start_index + 3]))


def logical_RZ(circuit: cirq.circuits, qubits: list, theta: float, start_index: int):
    """Prepare the logical RZ gate schedule
    Args:
       circuit: The desired circuit to encode cirq.circuits.
       qubits: the list of qubits used by the circuit.
       theta:  angle of rotation
       start_index: the starting index where the logical gate will be encoded. The physical qubits used for the encoding will be in [start_index,start_index+4].
    """
    sdg = cirq.S**-1
    circuit.append(cirq.H(qubits[start_index + 0]))
    circuit.append(cirq.S(qubits[start_index + 0]))
    circuit.append(cirq.Y(qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 4]))
    circuit.append(cirq.S(qubits[start_index + 4]))
    circuit.append(cirq.CNOT(qubits[start_index + 4], qubits[start_index + 2]))
    circuit.append(cirq.CNOT(qubits[start_index + 0], qubits[start_index + 2]))
    circuit.append(cirq.rz(rads=theta).on(qubits[start_index + 2]))
    circuit.append(cirq.CNOT(qubits[start_index + 0], qubits[start_index + 2]))
    circuit.append(cirq.CNOT(qubits[start_index + 4], qubits[start_index + 2]))
    circuit.append(sdg(qubits[start_index + 0]))
    circuit.append(cirq.H(qubits[start_index + 0]))
    circuit.append(cirq.Y(qubits[start_index + 2]))
    circuit.append(sdg(qubits[start_index + 4]))
    circuit.append(cirq.H(qubits[start_index + 4]))


def logical_RX(circuit: cirq.circuits, qubits: list, theta: float, start_index: int):
    """Prepare the logical RX gate schedule
    Args:
       circuit: The desired circuit to encode cirq.circuits.
       qubits: the list of qubits used by the circuit.
       theta:  angle of rotation
       start_index: the starting index where the logical gate will be encoded. The physical qubits used for the encoding will be in [start_index,start_index+4].
    """

    CZ = cirq.ControlledGate(sub_gate=cirq.Z, num_controls=1)

    circuit.append(cirq.rz(rads=-np.pi).on(qubits[start_index + 2]))
    circuit.append(cirq.X(qubits[start_index + 2]))
    circuit.append(CZ(qubits[start_index + 3], qubits[start_index + 2]))
    circuit.append(CZ(qubits[start_index + 1], qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 2]))
    circuit.append(cirq.rz(rads=theta).on(qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 2]))
    circuit.append(CZ(qubits[start_index + 1], qubits[start_index + 2]))
    circuit.append(CZ(qubits[start_index + 3], qubits[start_index + 2]))
    circuit.append(cirq.rz(rads=-np.pi).on(qubits[start_index + 2]))
    circuit.append(cirq.X(qubits[start_index + 2]))


def logical_RY(circuit: cirq.circuits, qubits: list, theta: float, start_index: int):
    """Prepare the logical RY gate schedule
    Args:
       circuit: The desired circuit to encode cirq.circuits.
       qubits: the list of qubits used by the circuit.
       theta:  angle of rotation
       start_index: the starting index where the logical gate will be encoded. The physical qubits used for the encoding will be in [start_index,start_index+4].
    """

    CZ = cirq.ControlledGate(sub_gate=cirq.Z, num_controls=1)
    circuit.append(cirq.H(qubits[start_index + 1]))
    circuit.append(cirq.H(qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 3]))
    circuit.append(cirq.rz(rads=-np.pi / 2).on(qubits[start_index + 2]))
    circuit.append(cirq.X(qubits[start_index + 2]))
    circuit.append(CZ(qubits[start_index + 3], qubits[start_index + 2]))
    circuit.append(CZ(qubits[start_index + 1], qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 2]))
    circuit.append(cirq.rz(rads=theta).on(qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 2]))
    circuit.append(CZ(qubits[start_index + 1], qubits[start_index + 2]))
    circuit.append(CZ(qubits[start_index + 3], qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 1]))
    circuit.append(cirq.rz(rads=-np.pi).on(qubits[start_index + 2]))
    circuit.append(cirq.X(qubits[start_index + 2]))
    circuit.append(cirq.rz(rads=-np.pi / 2).on(qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 3]))


def logical_CNOT(circuit: cirq.circuits, qubits: list, start_index: int):
    """Prepare the logical CNOT gate schedule
    Args:
       circuit: The desired circuit to encode cirq.circuits.
       qubits: the list of qubits used by the circuit.
       start_index: the starting index where the logical gate will be encoded. The physical qubits used for the encoding will be in [start_index,start_index+9].
    """

    sdg = cirq.S**-1
    circuit.append(cirq.H(qubits[start_index + 0]))
    circuit.append(cirq.S(qubits[start_index + 0]))
    circuit.append(cirq.Y(qubits[start_index + 2]))
    circuit.append(cirq.H(qubits[start_index + 4]))
    circuit.append(cirq.S(qubits[start_index + 4]))

    circuit.append(cirq.H(qubits[start_index + 5]))
    circuit.append(cirq.S(qubits[start_index + 5]))
    circuit.append(cirq.Y(qubits[start_index + 7]))
    circuit.append(cirq.H(qubits[start_index + 9]))
    circuit.append(cirq.S(qubits[start_index + 9]))

    circuit.append(cirq.CNOT(qubits[start_index + 0], qubits[start_index + 5]))
    circuit.append(cirq.CNOT(qubits[start_index + 2], qubits[start_index + 7]))
    circuit.append(cirq.CNOT(qubits[start_index + 4], qubits[start_index + 9]))
    circuit.append(cirq.CNOT(qubits[start_index + 0], qubits[start_index + 9]))
    circuit.append(cirq.CNOT(qubits[start_index + 2], qubits[start_index + 5]))
    circuit.append(cirq.CNOT(qubits[start_index + 4], qubits[start_index + 7]))
    circuit.append(cirq.CNOT(qubits[start_index + 0], qubits[start_index + 7]))
    circuit.append(cirq.CNOT(qubits[start_index + 2], qubits[start_index + 9]))
    circuit.append(cirq.CNOT(qubits[start_index + 4], qubits[start_index + 5]))

    circuit.append(sdg(qubits[start_index + 0]))
    circuit.append(cirq.H(qubits[start_index + 0]))
    circuit.append(cirq.Y(qubits[start_index + 2]))
    circuit.append(sdg(qubits[start_index + 4]))
    circuit.append(cirq.H(qubits[start_index + 4]))
    circuit.append(sdg(qubits[start_index + 5]))
    circuit.append(cirq.H(qubits[start_index + 5]))
    circuit.append(cirq.Y(qubits[start_index + 7]))
    circuit.append(sdg(qubits[start_index + 9]))
    circuit.append(cirq.H(qubits[start_index + 9]))
    return circuit
