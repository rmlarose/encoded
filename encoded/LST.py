import numpy as np
import cirq
import qiskit.circuit
import qsimcirq
import cirq_google
import jax.numpy as jnp
from tqdm import tqdm
import qiskit
import qiskit_ibm_runtime
from numba import njit
from numba import types
from numba.typed import Dict


def int_to_binary(N: int, b: int) -> str:
    """
    Convert a integer to a binary string of size N
    args:
        N: size of the binary (int)
        b: integer to be transformed (int)
    """
    bit = format(b, "b")
    if len(bit) < N:
        for _ in range(N - len(bit)):
            bit = str(0) + bit
    return bit


def base(qubits: cirq.Qid) -> cirq.Circuit:
    """
    Create a circuit initialized in a GHZ state
    args:
        qubits
    returns
    circuit: cirq.Circuit
    """
    return cirq.Circuit(
        cirq.H(qubits[0]),
        *[cirq.CNOT(qubits[i - 1], qubits[i]) for i in range(1, len(qubits))],
    )


def measure(circuit: cirq.Circuit, qubits: cirq.Qid) -> cirq.Circuit:
    """Append to all qubits of the circuit a measurement reconrd
    args:
    """
    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit


def cirq_to_qiskit(circuit: cirq.Circuit) -> qiskit.circuit:
    return qiskit.circuit.QuantumCircuit.from_qasm_str(circuit.to_qasm())


def bitGateMap(c, g, qi, qubits: cirq.Qid):
    """Map X/Y/Z string to cirq ops"""
    if g == 0:
        c.append(cirq.H(qubits[qi]))

    elif g == 1:
        sdg = cirq.S**-1
        c.append(sdg(qubits[qi]))
        c.append(cirq.H(qubits[qi]))

    elif g == 2:
        pass
    else:
        raise NotImplementedError(f"Unknown gate index expected 0,1,2 got {g}")


def generate_measurments(
    circuit, qubits: cirq.Qid, N: int, nsimu: int, probability: float
) -> tuple[list, list]:
    """
    Generate LST nsimu samples from a N-qubit circuit with depolarizing noise.
    args:
        circuit: cirq circuit to sample from
        qubits: the qubits used in the circuit
        N: the qubit size of circuit
        nsimu: number of samples
        probability: depolarizing noise probability
    returns:
        results: list of dictionnaries containing the result with the number of times obtained
        labels: rotation gate set used to obtain the results
    """
    gates = np.random.choice([0, 1, 2], size=(nsimu, N))
    labels, counts = np.unique(gates, axis=0, return_counts=True)
    results = []
    for pauli_index, count in zip(labels, counts):
        c_m = circuit.copy()
        for i, bit in enumerate(pauli_index):
            bitGateMap(c_m, bit, i, qubits)
        measure(c_m, qubits)
        if probability > 0.0:
            c_m = c_m.with_noise(cirq.depolarize(p=probability))
        s = qsimcirq.QSimSimulator()
        samples = s.run(c_m, repetitions=count)
        counts = samples.histogram(key="result")
        dict_param1 = Dict.empty(
            key_type=types.int64,
            value_type=types.int64,
        )
        for key in counts.keys():
            dict_param1[key] = counts[key]
        results.append(dict_param1)
    return labels, results


def generate_measurements_fixed_gates(
    circuit: cirq.Circuit,
    qubits: cirq.Qid,
    N: int,
    N_U: int,
    N_S: int,
    probability: float = 0,
) -> tuple[list, list]:
    """
    Generate LST nsimu samples from a N-qubit circuit with depolarizing noise. We fixed the number of LST rotation gate set by N_U
    args:
        circuit: cirq circuit to sample from
        qubits: the qubits used in the circuit
        N: the qubit size of circuit
        N_U: number of gate set
        nsimu: number of samples per gate set
        probability: depolarizing noise probability
    returns:
        results: list of dictionnaries containing the result with the number of times obtained
        labels: rotation gate set used to obtain the results
    """
    gates = np.random.choice([0, 1, 2], size=(N_U, N))
    labels, counts = np.unique(gates, axis=0, return_counts=True)
    results = []
    for pauli_index, count in zip(labels, tqdm(counts)):
        c_m = circuit.copy()
        for i, bit in enumerate(pauli_index):
            bitGateMap(c_m, bit, i, qubits)
        measure(c_m, qubits)
        if probability > 0.0:
            c_m = c_m.with_noise(cirq.depolarize(p=probability))
        s = qsimcirq.QSimSimulator()
        samples = s.run(c_m, repetitions=(N_S))
        counts = samples.histogram(key="result")
        results.append(counts)
    return labels, results


def generate_measurements_QVM_google(
    circuit: cirq.Circuit,
    qubits: cirq.Qid,
    device_qubit_chain,
    processor_id,
    gate_type,
    N: int,
    N_U: int,
    N_S: int,
) -> tuple[dict, dict]:
    """
    Generate measurement records with the QVM from google. We use N_S shots per gate set with N_U gate set
    args:
        circuit: circuit to sample cirq.Circuit,
        qubits: qubits used in the QVM cirq.Qid,
        device_qubit_chain: topology of the qubit architecture,
        processor_id: id of the QVM processor,
        gate_type: type of gate the QVM is using (ISWAP,...),
        N: size of the circuit int,
        N_U: number of gateset int,
        N_S: number of shots per gate set int,
    returns:
        results: list of dictionnaries containing the measurements as well number of time obtained
        labels: gateset used for the results
    """

    cal = cirq_google.engine.load_median_device_calibration(processor_id)
    noise_props = cirq_google.noise_properties_from_calibration(cal)
    noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
    sim = qsimcirq.QSimSimulator(noise=noise_model)
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    sim_processor = cirq_google.engine.SimulatedLocalProcessor(
        processor_id=processor_id,
        sampler=sim,
        device=device,
        calibrations={cal.timestamp // 1000: cal},
    )
    sim_engine = cirq_google.engine.SimulatedLocalEngine([sim_processor])

    gates = np.random.choice([0, 1, 2], size=(N_U, N))
    labels, counts = np.unique(gates, axis=0, return_counts=True)
    results = []
    for pauli_index, count in zip(labels, tqdm(counts)):
        c_m = circuit.copy()
        for i, bit in enumerate(pauli_index):
            bitGateMap(c_m, bit, i, qubits)
        measure(c_m, qubits)

        translated_ghz_circuit = cirq.optimize_for_target_gateset(
            c_m, context=cirq.TransformerContext(deep=True), gateset=gate_type
        )
        qubit_map = dict(zip(qubits, device_qubit_chain))
        device_ready_ghz_circuit = translated_ghz_circuit.transform_qubits(
            lambda q: qubit_map[q]
        )
        samples = sim_engine.get_sampler(processor_id).run(
            device_ready_ghz_circuit, repetitions=[N_S * count]
        )
        counts = samples.histogram(key="result")
        results.append(counts)
    return labels, results


def run_batch(
    circuit_array: list[cirq.Circuit],
    shots: int,
    backend: qiskit_ibm_runtime,
) -> list:
    """
    Sample a list of circuit using the IBM quantum compute. This permit sending many circuit at once (maximum circuit set possible).
    args:
        circuit_array: list of the circuit to sample list[cirq.Circuit],
        shots: number of shots per circuit int,
        backend: id of the quantum computer needed qiskit_ibm_runtime,
    returns:
        results: measurement reconrd as well as the number of times obtained

    """

    circuit_array = np.array_split(
        np.array(circuit_array), len(circuit_array) // 300 + 1
    )
    results = []
    for circ in circuit_array:
        pm = qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager(
            backend=backend, optimization_level=3
        )
        isa_circuit = [pm.run(c) for c in circ]
        result = backend.run(isa_circuit, shots=shots).result()
        counts = result.get_counts()
        for c in counts:
            results.append(c)
    return results


def generate_measurements_IBM(
    circuit: cirq.Circuit,
    qubits: cirq.Qid,
    N: int,
    nsimu: int,
    backend: qiskit_ibm_runtime,
    shots: int,
) -> tuple[list, list]:
    """
    Generate measurement records with the IBM quantum computers.
    args:
        circuit: circuit to be sample with LST cirq.Circuit,
        qubits: qubits used in the circuit cirq.Qid,
        N: size of the circuit int,
        nsimu: number of  LST samples int,
        backend: id of the IBM quantum computer qiskit_ibm_runtime,
        shots: number of shots per gateset int,
    returns:
        results: list of dictionnaries containing the measurements as well number of time obtained
        labels: gateset used for the results

    """

    gates = np.random.choice([0, 1, 2], size=(nsimu, N))
    labels = np.unique(gates, axis=0, return_counts=False)
    results = []
    circuit_array = np.empty(shape=len(labels), dtype=qiskit.QuantumCircuit)
    for index, pauli_index in tqdm(enumerate(labels)):
        c_m = circuit.copy()
        for i, bit in enumerate(pauli_index):
            bitGateMap(c_m, bit, i, qubits)
        measure(c_m, qubits)
        c_m = cirq_to_qiskit(c_m)
        circuit_array[index] = c_m

    results = run_batch(circuit_array, shots=shots, backend=backend)
    return labels, results


# @njit()
def int_to_bin_list(x, length):
    result = [0] * length
    for i in range(length):
        if x & (1 << i):
            result[length - 1 - i] = 1
    return result


# @njit(nopython=True)
def reconstruct_matrix(
    labels: np.ndarray[np.ndarray[int]], results: np.ndarray[float], N: int
) -> np.ndarray[float]:
    """Reconstruct the density matrix from the measurement results and labels of the gates
    args:
        labels: type of gates used for shadow tomography
        results: measurement reconrds
        N: number of qubits
    """
    shadows = np.zeros((2**N, 2**N), dtype=np.complex128)
    shots = 0
    Identity = np.eye(2, dtype=np.complex128)
    rotGate = np.empty((3, 2, 2), dtype=np.complex128)
    rotGateconj = np.empty((3, 2, 2), dtype=np.complex128)

    rotGate[0] = (
        1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    )
    rotGate[1] = (
        1 / np.sqrt(2) * np.array([[1.0, -1.0j], [1.0, 1.0j]], dtype=np.complex128)
    )
    rotGate[2] = np.eye(2, dtype=np.complex128)

    rotGateconj[0] = (
        1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    )
    rotGateconj[1] = (
        1 / np.sqrt(2) * np.array([[1.0, 1.0j], [1.0, -1.0j]], dtype=np.complex128)
    )
    rotGateconj[2] = np.eye(2, dtype=np.complex128)

    for pauli_index, counts in zip(labels, results):
        for bit, count in counts.items():
            bit = int_to_bin_list(bit, N)
            mat = np.array([[1.0]], dtype=np.complex128)
            for i, bi in enumerate(bit):
                b = rotGate[pauli_index[i]][bi, :]
                bconj = rotGateconj[pauli_index[i]][bi, :]
                mat = np.kron(mat, 3 * np.outer(bconj, b) - Identity)
            shadows += mat * count
            shots += count
    return shadows / shots


def reconstruct_matrix_IBM(
    labels: np.ndarray[np.ndarray[int]], results: np.ndarray[float], N: int
) -> np.ndarray[float]:
    """Reconstruct the density matrix from the measurement results and labels of the gates.
    This use the results provided by qiskit / IBM Quantum Computers.
    args:
        labels: type of gates used for shadow tomography
        results: measurement reconrds
        N: number of qubits
    """
    shadows = []
    shots = 0
    Identity = np.eye(2)
    rotGate = [
        1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]]),
        1 / np.sqrt(2) * np.array([[1.0, -1.0j], [1.0, 1.0j]]),
        Identity,
    ]

    for pauli_string, counts in zip(labels, results):
        for bit, count in counts.items():
            mat = 1.0
            for i, bi in enumerate(bit[::-1]):
                b = rotGate[pauli_string[i]][int(bi), :]
                mat = np.kron(mat, 3 * np.outer(b.conj(), b) - Identity)
            shadows.append(mat * count)
            shots += count

    return np.sum(shadows, axis=0) / (shots)


def trace_dist(rho_exact: np.ndarray, rho_exp: np.ndarray) -> float:
    """returns normalized trace distance between rho_exact and rho_exp"""
    mid = (rho_exact - rho_exp).conj().T @ (rho_exact - rho_exp)
    N = 2 ** int(np.log2(rho_exact.shape[0]) / 2)
    U1, d, U2 = np.linalg.svd(mid)
    sqrt_mid = U1 @ np.diag(np.sqrt(d)) @ U2
    dist = np.trace(sqrt_mid) / 2
    return dist / N


def measure_observable(
    rho: np.ndarray[np.ndarray], observable: np.ndarray[np.ndarray]
) -> float:
    """Compute the value of an observable for a certain density matrix
    args:
        rho: density matrix
        observable: the observable to measure"""
    return (np.trace(rho @ observable) / np.trace(rho)).real


def obs_gate(g: str):
    """
    Help to build an observable by using a chain of strings

    return: the corresponding matrix to the gate
    """
    if g == "I":
        return cirq.unitary(cirq.I)
    if g == "X":
        return cirq.unitary(cirq.X)
    elif g == "Y":
        return cirq.unitary(cirq.Y)
    elif g == "Z":
        return cirq.unitary(cirq.Z)
    else:
        raise NotImplementedError(f"Unknown gate {g}")
