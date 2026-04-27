"""Unit tests for tcc.py."""

import cirq
import stimcirq

from tcc import tcc_encoding


# Parameters.
distance: int = 5
nshots: int = 1_000_000

# Prepare the logical zero state from the physical zero state.
encode_zero = tcc_encoding(distance)
qubits = encode_zero.all_qubits()

# Replace RESET-X gates with H gates.
encode_zero = cirq.H.on_each(qubits) + encode_zero[1:]

# Apply logical X to prepare the logical one state from the logical zero state.
encode_one = encode_zero + cirq.Moment(cirq.X.on_each(qubits))

# Display circuits.
print("Logical 0 encoding:")
print(encode_zero)
print("\n" * 2)
print("Logical 1 encoding:")
print(encode_one)

# Sample bitstrings with Stim.
sampler_zero = stimcirq.cirq_circuit_to_stim_circuit(encode_zero + cirq.measure(qubits)).compile_sampler()
sampler_one = stimcirq.cirq_circuit_to_stim_circuit(encode_one + cirq.measure(qubits)).compile_sampler()

zero_states = sorted(set([int.from_bytes(bytes(b), 'little') for b in sampler_zero.sample(nshots, bit_packed=True)]))
one_states = sorted(set([int.from_bytes(bytes(b), 'little') for b in sampler_one.sample(nshots, bit_packed=True)]))

# Display output.
print("Sampled logical 0 states:")
print(len(zero_states))
print("\n" * 2)
print("Sampled logical 1 states:")
print(len(one_states))
for zero_state in zero_states:
    if zero_state in one_states:
        raise AssertionError("Overlapping codewords")
print("Codewords are unique.")
