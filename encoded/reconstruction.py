import stim
import cirq
import stimcirq
import qsimcirq
import numpy as np
import tqdm
from typing import List

from LST import *

def generate_clifford_tableaus(nqubits, ntableaus):
    tableaus = np.empty(ntableaus, dtype=stim.Tableau)
    for i in range(ntableaus):
        tableaus[i] = stim.Tableau.random(nqubits)
    return tableaus

def apply_clifford_measurement(circuit, tableau, key="result"):
    clifford_c = stimcirq.stim_circuit_to_cirq_circuit(tableau.to_circuit())
    circuit += clifford_c
    circuit.append(cirq.measure(*circuit.all_qubits(), key=key))

def generate_clifford_measurements(
    circuit, nsimu, noise_rate: float=0, shots=1, seed=None
) -> tuple[list, list]:
    N = len(circuit.all_qubits())
    if isinstance(nsimu, int):
        gates = generate_clifford_tableaus(N, nsimu)
    else: gates = nsimu
    results = []
    for clifford in tqdm(gates):
        c_m = circuit.copy()
        apply_clifford_measurement(c_m, clifford)
        if noise_rate > 0.0:
            c_m = c_m.with_noise(cirq.depolarize(p=noise_rate))
        s = qsimcirq.QSimSimulator(seed=seed)
        samples = s.run(c_m, repetitions=shots)
        counts = samples.histogram(key="result")
        results.append(counts)
    return gates, results

#################################################################
# Hong-Ye GitHub code
# https://github.com/hongyehu/Sim-Clifford/blob/lst/base/utils.py
#################################################################
def acq(g1, g2):
    '''Calculate Pauli operator anticmuunation indicator.

    Parameters:
    g1: int (2*N) - the first Pauli string in binary repr.
    g2: int (2*N) - the second Pauli string in binary repr.
    
    Returns:
    acq: int - acq = 0 if g1, g2 commute, acq = 1 if g1, g2 anticommute.'''
    assert g1.shape == g2.shape
    (N2,) = g1.shape
    N = N2//2
    acq = 0
    for i in range(N):
        acq += g1[2*i+1]*g2[2*i] - g1[2*i]*g2[2*i+1]
    return acq % 2

#################################################################
# Hong-Ye GitHub code
# https://github.com/hongyehu/Sim-Clifford/blob/lst/base/utils.py
#################################################################
def ipow(g1, g2):
    '''Phase indicator for the product of two Pauli strings.

    Parameters:
    g1: int (2*N) - the first Pauli string in binary repr.
    g2: int (2*N) - the second Pauli string in binary repr.
    
    Returns:
    ipow: int - the phase indicator (power of i) when product 
        sigma[g1] with sigma[g2].'''
    assert g1.shape == g2.shape
    (N2,) = g1.shape
    N = N2//2
    ipow = 0
    for i in range(N):
        g1x = g1[2*i  ]
        g1z = g1[2*i+1]
        g2x = g2[2*i  ]
        g2z = g2[2*i+1]
        gx = g1x + g2x
        gz = g1z + g2z 
        ipow += g1z * g2x - g1x * g2z + 2*((gx//2) * gz + gx * (gz//2))
    return ipow % 4

#################################################################
# Hong-Ye GitHub code
# https://github.com/hongyehu/Sim-Clifford/blob/lst/base/utils.py
#################################################################
def stabilizer_projection_full(gs_stb, ps_stb, gs_obs, ps_obs, r):
    '''Measure Pauli operators on a stabilizer state.

    Parameters:
    gs_stb: int (2*N, 2*N) - Pauli strings in original stabilizer tableau.
    ps_stb: int (N) - phase indicators of (de)stabilizers.
    gs_obs: int (L, 2*N) - strings of Pauli operators to be measured.
    ps_obs: int (L) - phase indicators of Pauli operators to be measured.
    r: int - log2 rank of density matrix (num of standby stablizers).

    Returns:
    gs_stb: int (2*N, 2*N) - Pauli strings in updated stabilizer tableau.
    ps_stb: int (N) - phase indicators of (de)stabilizers.
    r: int - updated log2 rank of density matrix.
    trace: float - Tr(P * rho1 * P*)'''
    (L, Ng) = gs_obs.shape
    N = Ng//2
#     assert L==N # Current implementation is full state projection
    assert 0<=r<=N
    ga = np.empty(2*N, dtype=np.int_) # workspace for stabilizer accumulation
    pa = 0 # workspace for phase accumulation
    trace = 1
    for k in range(L): # for each observable gs_obs[k]
        update = False
        extend = False
        p = 0 # pointer
        ga[:] = 0
        pa = 0
        for j in range(2*N):
            if acq(gs_stb[j], gs_obs[k]): # find gs_stb[j] anticommute with gs_obs[k]
                if update: # if gs_stb[j] is not the first anticommuting operator
                    # update gs_stb[j] to commute with gs_obs[k]
                    if j < N: # if gs_stb[j] is a stablizer, phase matters
                        ps_stb[j] = (ps_stb[j] + ps_stb[p] + ipow(gs_stb[j], gs_stb[p]))%4
                    gs_stb[j] = (gs_stb[j] + gs_stb[p])%2
                else: # if gs_stb[j] is the first anticommuting operator
                    if j < N + r: # if gs_stb[j] is not an active destabilizer
                        p = j # move pointer to j
                        update = True
                        if not r <= j < N: # if gs_stb[j] is a standby operator
                            extend = True
                    else: # gs_stb[j] anticommute with destabilizer, meaning gs_obs[k] already a combination of active stabilizers
                        # collect corresponding stabilizer component to ga
                        pa = (pa + ps_stb[j-N] + ipow(ga, gs_stb[j-N]))%4
                        ga = (ga + gs_stb[j-N])%2
        if update:
            # now gs_stb[p] and gs_obs[k] anticommute
            q = (p+N)%(2*N) # get q as dual of p 
            gs_stb[q] = gs_stb[p] # move gs_stb[p] to gs_stb[q]
            gs_stb[p] = gs_obs[k] # add gs_obs[k] to gs_stb[p]
            if extend:
                r -= 1 # rank will reduce under extension
                # bring new stabilizer from p to r
                if p == r:
                    pass
                elif q == r:
                    gs_stb[np.array([p,q])] = gs_stb[np.array([q,p])] # swap p,q
                else:
                    s = (r+N)%(2*N) # get s as dual of r
                    gs_stb[np.array([p,r])] = gs_stb[np.array([r,p])] # swap p,r
                    gs_stb[np.array([q,s])] = gs_stb[np.array([s,q])] # swap q,s
                p = r
            # the projection will change phase of stabilizer
            ps_stb[p] = ps_obs[k]
            trace = trace/2.
        else: # no update, gs_obs[k] is eigen, result is in pa
            assert((ga == gs_obs[k]).all())
            if not pa == ps_obs[k]:
                trace = 0.
    return gs_stb, ps_stb, r, trace

#################################################################
# Hong-Ye GitHub code
# https://github.com/hongyehu/Sim-Clifford/blob/lst/base/utils.py
#################################################################
def stabilizer_expect(gs_stb, ps_stb, gs_obs, ps_obs, r):
    '''Evaluate the expectation values of Pauli operators on a stabilizer state.

    Parameters:
    gs_stb: int (2*N, 2*N) - Pauli strings in original stabilizer tableau.
    ps_stb: int (N) - phase indicators of (de)stabilizers.
    gs_obs: int (L, 2*N) - strings of Pauli operators to be measured.
    ps_obs: int (L) - phase indicators of Pauli operators to be measured.
    r: int - log2 rank of density matrix (num of standby stablizers).

    Returns:
    xs: int (L) - expectation values of Pauli operators.'''
    (L, Ng) = gs_obs.shape
    N = Ng//2
    assert 0<=r<=N
    xs = np.empty(L, dtype=np.int_) # expectation values
    ga = np.empty(2*N, dtype=np.int_) # workspace for stabilizer accumulation
    pa = 0 # workspace for sign accumulation
    for k in range(L): # for each observable gs_obs[k] 
        ga[:] = 0
        pa = 0
        trivial = True # assuming gs_obs[k] is trivial in code subspace
        for j in range(2*N):
            if acq(gs_stb[j], gs_obs[k]): 
                if j < N + r: # if gs_stb[j] is active stablizer or standby.
                    xs[k] = 0 # gs_obs[k] is logical or error operator.
                    trivial = False # gs_obs[k] is not trivial
                    break
                else: # accumulate stablizer components
                    pa = (pa + ps_stb[j-N] + ipow(ga, gs_stb[j-N]))%4
                    ga = (ga + gs_stb[j-N])%2
        if trivial:
            xs[k] = (-1)**(((pa - ps_obs[k])%4)//2)
    return xs

def tabeau_to_stabilizers(tableau):
    t_mat = tableau.to_numpy()
    N = len(t_mat[0])

    # interleave X and Z terms
    zs = np.empty((N, 2*N))
    zs[:,0::2] = t_mat[2]*1
    zs[:,1::2] = t_mat[3]*1
    xs = np.empty((N, 2*N))
    xs[:,0::2] = t_mat[0]*1
    xs[:,1::2] = t_mat[1]*1

    gs = np.concatenate((zs, xs))
    ps = np.concatenate((t_mat[5]*1, t_mat[4]*1))
    return gs, ps

def reconstruct_trace(
    tableaus: np.ndarray[np.ndarray[int]], results: np.ndarray[float], proj_gs, proj_ps, obs_gs, obs_ps, obs_cs
) -> np.ndarray[float]:
    tr_full = 0
    tot_id_sh = 0
    shots = 0
    N = len(tableaus[0].to_numpy()[0])
    print(2**N+1)
    for clifford, counts in zip(tableaus, tqdm(results)):
        for bit, count in counts.items():
            bit = np.array(int_to_bin_list(bit, N))
            gs, ps = tabeau_to_stabilizers(clifford.inverse())
            tableau = clifford.inverse().to_numpy() # U_dag

            # Modify Stabilizer state to reflect measurements
            ps[:N] = (bit^ps[:N])*2
            ps[N:] = ((bit^ps[N:])*2 + 2)%2

            # gs, ps, _, tr = stabilizer_projection_full(gs, ps, proj_gs, proj_ps, 0)
            # print(tr)

            xs = stabilizer_expect(gs, ps, obs_gs, obs_ps, 0)
            # print(xs)
            ev = 0
            for c, x in zip(obs_cs, xs):
                ev += c*x
            id_sh = 0
            for c, g, p in zip(obs_cs, obs_gs, obs_ps):
                if not g.any(): id_sh += (-1)**p * 2**N * c
            tr_full += ev * count
            tot_id_sh += id_sh * count
            shots += count
    print(tot_id_sh, shots)
    return (tr_full * (2**N+1) - tot_id_sh) / shots
    # return (tr_full * 25) / shots

# def reconstruct_pauli_trace(labels, results):


def reconstruct_state(
        tableaus: np.ndarray[np.ndarray[int]], results: np.ndarray[float]
):
    N = len(tableaus[0].to_numpy()[0])
    shadows = np.zeros((2**N, 2**N), dtype=np.complex128)
    shots = 0
    Identity = np.eye(2**N, dtype=np.complex128)

    for tableau, counts in zip(tableaus, results):
        # tab_c = stimcirq.stim_circuit_to_cirq_circuit(tableau.to_circuit())
        # for i in range(N):
        #     tab_c.append(cirq.I.on(cirq.LineQubit(i)))
        # U = tab_c.unitary()
        
        U = tableau.to_unitary_matrix(endian="big")
        # print(U.shape)
        for bit, count in counts.items():
            # bit = int_to_bin_list(bit, N)
            b = U[bit, :]
            bconj = np.conj(b)
            mat = (2**N + 1) * np.outer(bconj, b) - Identity
            shadows += mat * count
            shots += count
    return shadows / shots