import cirq
from typing import Sequence
import stim
import stimcirq

import numpy as np
from typing import List, Dict
import dataclasses


@dataclasses.dataclass
class Tile:
    qubits: list
    color: str


def make_color_code_tiles(*, base_data_width):
    if not (base_data_width % 2 == 1 and base_data_width >= 3):
        raise ValueError(f"{base_data_width=} wasn't an odd number at least as large as 3.")
    w = base_data_width * 2 - 1

    def is_in_bounds(q: complex) -> bool:
        if q.imag < 0:
            return False
        if q.imag * 2 > q.real * 3:
            return False
        if q.imag * 2 > (w - q.real) * 3:
            return False
        return True

    tiles = []
    hexagon_offsets = [-1, +1j, +1j + 1, +2, -1j + 1, -1j]
    for x in range(1, w, 2):
        for y in range((x // 2) % 2, w, 2):
            q = x + 1j * y

            tile = Tile(
                color=["red", "green", "blue"][y % 3],
                qubits=[q + d for d in hexagon_offsets if is_in_bounds(q + d)],
            )

            if len(tile.qubits) in [4, 6]:
                tiles.append(tile)

    return tiles


def get_stabilizer_generators(distance: int):
    tiles = make_color_code_tiles(base_data_width=distance)
    all_qubits = {q for tile in tiles for q in tile.qubits}

    # Only difference here is with the chromobius notebook that we rever
    sorted_qubits = reversed(sorted(all_qubits, key=lambda q: (q.imag, q.real)))
    q2i = {q: i for i, q in enumerate(sorted_qubits)}

    sorted_tiles = []
    for tile in tiles:
        sorted_tiles.append([q2i[q] for q in tile.qubits])

    stabilizers_x = []
    stabilizers_z = []
    for tile in sorted_tiles:
        stab_x = ""
        stab_z = ""
        for i in range(int((3 * distance**2 + 1) / 4)):
            if i in tile:
                stab_x += "X"
                stab_z += "Z"
            else:
                stab_x += "I"
                stab_z += "I"
        stabilizers_x.append(stab_x)
        stabilizers_z.append(stab_z)

    return stabilizers_x + stabilizers_z


def stabilizers_to_encoder(stabilizers) -> stim.Circuit:
    tableau = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_underconstrained=True,
    )

    return tableau.to_circuit(method="graph_state")


def tcc_encoding(distance):
    generator_strs = get_stabilizer_generators(distance)[::-1]
    return stimcirq.stim_circuit_to_cirq_circuit(stabilizers_to_encoder([stim.PauliString(s) for s in generator_strs]))


def tcc_H(qreg: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(qreg))
    return circuit


def tcc_X(qreg: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    circuit.append(cirq.X.on_each(qreg))
    return circuit


def tcc_CNOT(qreg1: Sequence[cirq.Qid], qreg2: Sequence[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    for i in range(len(qreg1)):
        circuit.append(cirq.CNOT.on(qreg1[i], qreg2[i]))
    return circuit
