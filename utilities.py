#!/usr/bin/env python3
import qiskit
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile
)
from qiskit.quantum_info import (
    Kraus, 
    SuperOp,
)
from qiskit_aer.noise import(
    depolarizing_error,
    phase_damping_error,
    NoiseModel,
    QuantumError,
    amplitude_damping_error,
)
# from qiskit_ibm_provider import IBMProvider
from qiskit import Aer
# from qiskit_aer import AerSimulator
# import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import math
from qiskit.quantum_info import Pauli

from qiskit.circuit.library import RZGate


BACKEND_SIM=Aer.get_backend("aer_simulator")
SHOTS=10000
BASIS=["h", "x", "rz", "ccx", "cx", "cz", "id"]

# def stripLeadingCRegs(dist, keep_regs):
#     '''Strips the leading classical registers.
#     args:
#     dist (dict): key, value pair of a distribution. 
#     keep_regs (int): the number of ClassicalRegister objects to keep in the key 
#     starting from the right end of the keys. '''
#     filteredDist={}
#     for k, v in dist.items():
#         print("original: ", k)
#         splitK=k.split(" ")
#         print("split: ", splitK)
#         splitK=splitK[-keep_regs::]
#         print("popped: ", splitK)
#         newK=" ".join(splitK)
#         print("new: ", newK)
#         filteredDist[newK]=filteredDist.get(newK,0)+v
#     return filteredDist

def _createQECSubcirc(qecCirc, qecDataQR, qecRepCR, qecQubit):
    '''Helper function for creating the repetition code for one half of an
    EPR pair.
    args: 
    qecCirc (QuantumCircuit): the circuit to apply the code to.
    qecDatQR (QuantumRegister): the registers for the epr pair and the ancillas for
    the repetition code.
    qecRegCR (ClassicalRegister): the registers to store the rep code measurements to after
    decoding. These measurement results are not necessary because we use a ccx gate.
    qecQubit (Qubit): the qubit that is protected with the repetition code'''
    qecCirc.barrier()
    # qecQubit will be the used as part of the BSM at the charlie repeater (middle node)
    qecCirc.h(qecDataQR[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQR[0], qecQubit)
    # QEC encoding
    qecCirc.cx(qecQubit,qecDataQR[1])
    qecCirc.cx(qecQubit,qecDataQR[2]) 
    qecCirc.barrier()
    for idx in list(range(1,reps+1)):
        qecCirc.id(qecDataQR[idx])
    qecCirc.id(qecQubit)
    qecCirc.barrier()
    # QEC step
    qecCirc.cx(qecQubit, qecDataQR[1])
    qecCirc.cx(qecQubit, qecDataQR[2])
    qecCirc.barrier()
    # QEC correcting
    qecCirc.ccx(qecDataQR[1], qecDataQR[2], qecQubit)
    qecCirc.barrier()
    # Measurements
    qecCirc.measure(qecDataQR[1], qecRepCR[0])
    qecCirc.measure(qecDataQR[2], qecRepCR[1])

def createQECCirc():
    '''Creates a qec verion of the entanglement swapping scenario a-c-b. one half of 
    each of the a and b local epr pairs are encoded in the repetition code. The encoded
    qubits are sent to the c repeater which decodes, corrects, and does the entanglement swapping.'''
    qecDataQRa=QuantumRegister(3, "a")
    qecRepCRa=ClassicalRegister(2, "rep_ancillas_a")
    qecDataQRb=QuantumRegister(3, "b")
    qecRepCRb=ClassicalRegister(2, "rep_ancillas_b")
    qecQRc=QuantumRegister(2, "c")
    qecCRc=ClassicalRegister(2, "entanglement_swap")
    qecEPRCR=ClassicalRegister(2, "final_epr_pair")
    qecCirc=QuantumCircuit(qecDataQRa, qecDataQRb, qecQRc, qecEPRCR, qecRepCRa, qecRepCRb, qecCRc)
    _createQECSubcirc(qecCirc, qecDataQRa, qecRepCRa, qecQRc[0])
    _createQECSubcirc(qecCirc, qecDataQRb, qecRepCRb, qecQRc[1])
    qecCirc.barrier()
    # Entanglement swap
    qecCirc.cx(qecQRc[0], qecQRc[1])
    qecCirc.h(qecQRc[0])
    qecCirc.measure(qecQRc[0], qecCRc[0])
    qecCirc.measure(qecQRc[1], qecCRc[1])
    _add_fixing_gates(qecCirc, qecDataQRa, qecCRc, 0)
    # qecCirc.measure(qecDataQRa[0], cr_h[0])
    # qecCirc.measure(qecDataQRa[1], cr_h[1])
   
    return qecCirc, qecDataQRa, qecRepCRa, qecDataQRb, qecRepCRb, qecEPRCR

def _createQedSubcirc(qecCirc, qecDataQR, qecRepCR):
    '''Helper function for creating the repetition code for one half of an
    EPR pair.
    args: 
    qecCirc (QuantumCircuit): the circuit to apply the code to.
    qecDataQR (QuantumRegister): the registers for the epr pair and the ancillas for
    the repetition code.
    qecRegCR (ClassicalRegister): the registers to store the rep code measurements to after
    decoding. These measurement results are not necessary because we use a ccx gate.
    qecQubit (Qubit): the qubit that is protected with the repetition code'''
    qubits=qecDataQR.size
    qecCirc.barrier()
    # qecQubit will be the used as part of the BSM at the charlie repeater (middle node)
    qecCirc.h(qecDataQR[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQR[0], qecDataQR[1])
    # QEC encoding
    #the first two indexes are epr pair and the rest are ancillas for the repetition code
    ancil_idxs=range(qubits)[2:] 
    for idx in ancil_idxs:
        qecCirc.cx(qecDataQR[1], qecDataQR[idx])
    # qecCirc.cx(qecQubit,qecDataQR[2]) 
    qecCirc.barrier()
    qecCirc.id(qecDataQR[1])
    for idx in ancil_idxs:
        qecCirc.id(qecDataQR[idx])
    qecCirc.barrier()
    # QEC decoding
    for idx in ancil_idxs:
        qecCirc.cx(qecDataQR[1], qecDataQR[idx])
    qecCirc.barrier()
    # Measurements
    for idx_ancil, idx_m in zip(ancil_idxs, list(range(qecRepCR.size))):
        qecCirc.measure(qecDataQR[idx_ancil], qecRepCR[idx_m])
    # qecCirc.measure(qecDataQR[2], qecRepCR[1])

def createQedCirc(reps):
    '''Creates a qec verion of the entanglement swapping scenario a-c-b. one half of 
    each of the a and b local epr pairs are encoded in the repetition code. The encoded
    qubits are sent to the c repeater which decodes, corrects, and does the entanglement swapping.'''
    #the number of qubits at a and b both have at 2 for the local epr pairs and then the number
    # of ancillas for the repetition code.
    qecQRa=QuantumRegister(reps+2, "a")
    qecQRb=QuantumRegister(reps+2, "b")
    qecRepCRa=ClassicalRegister(reps, "rep_ancillas_a")
    qecRepCRb=ClassicalRegister(reps, "rep_ancillas_b")
    # qecQRc=QuantumRegister(2, "swap_qubits")
    qecCRc=ClassicalRegister(2, "entanglement_swap")
    qecEPRCR=ClassicalRegister(2, "final_epr_pair")
    qecCirc=QuantumCircuit(qecQRa, qecQRb, qecEPRCR, qecRepCRa, qecRepCRb, qecCRc)
    _createQedSubcirc(qecCirc, qecQRa, qecRepCRa)
    _createQedSubcirc(qecCirc, qecQRb, qecRepCRb)
    qecCirc.barrier()
    # Entanglement swap
    qecCirc.cx(qecQRa[1], qecQRb[1])
    qecCirc.h(qecQRa[1])
    qecCirc.measure(qecQRa[1], qecCRc[0])
    qecCirc.measure(qecQRb[1], qecCRc[1])
    epr_idx=0
    _add_fixing_gates(qecCirc, qecQRa, qecCRc, epr_idx)
    # qecCirc.measure(qecDataQRa[0], cr_h[0])
    # qecCirc.measure(qecDataQRa[1], cr_h[1])
   
    return qecCirc, qecQRa, qecRepCRa, qecQRb, qecRepCRb, qecEPRCR

def createPCSRepCircTest():
    '''Creates a qec + pcs x checks verion of the entanglement swapping scenario a-c-b. the x-checks
     are conditioned only one acilla. one half of each of the a and b local epr pairs are encoded in 
     the repetition code and protected by all x checks with one ancilla for a and one ancilla for b. 
     The encoded qubits are sent to the c repeater which decodes, corrects, and 
     does the entanglement swapping.'''
    qecDataQRa=QuantumRegister(3, "a")
    qecRepCRa=ClassicalRegister(2, "rep_ancillas_a")
    qecDataQRb=QuantumRegister(3, "b")
    qecRepCRb=ClassicalRegister(2, "rep_ancillas_b")
    qecQRc=QuantumRegister(2, "c")
    qecCRc=ClassicalRegister(2, "entanglement_swap")
    qecEPRCR=ClassicalRegister(2, "final_epr_pair")
    pcsQRa=QuantumRegister(2, "pcs_a")
    pcsQRb=QuantumRegister(2, "pcs_b")
    pcsCR=ClassicalRegister(4, "pcs")
    qecCirc=QuantumCircuit(qecDataQRa, qecDataQRb, qecQRc, pcsQRa, pcsQRb, qecEPRCR, pcsCR, qecRepCRa, qecRepCRb, qecCRc)
    #QEC
    # _createQECSubcirc(qecCirc, qecDataQRa, qecRepCRa, qecQRc[0])
    # _createQECSubcirc(qecCirc, qecDataQRb, qecRepCRb, qecQRc[1])
    qecCirc.h(qecDataQRa[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQRa[0], qecQRc[0])
    qecCirc.h(qecDataQRb[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQRb[0], qecQRc[1])
    qecCirc.barrier()
    # QEC encoding
    qecCirc.cx(qecQRc[0],qecDataQRa[1])
    qecCirc.cx(qecQRc[0],qecDataQRa[2])
    qecCirc.cx(qecQRc[1],qecDataQRb[1])
    qecCirc.cx(qecQRc[1],qecDataQRb[2]) 
    qecCirc.barrier()
    #PCS
    qecCirc.h((pcsQRa[0], pcsQRb[0], pcsQRa[1], pcsQRb[1]))
    qecCirc.cx(pcsQRa[0], qecQRc[0])
    qecCirc.cx(pcsQRa[1], qecDataQRa[1])
    qecCirc.cx(pcsQRb[0], qecQRc[1])
    qecCirc.cx(pcsQRb[1], qecDataQRb[1])
    qecCirc.barrier()
    # for idx in list(range(1,3)):
    #     qecCirc.id(qecDataQR[idx])
    qecCirc.id([qecDataQRa[1], qecDataQRa[2], qecDataQRb[1], 
                qecDataQRb[2], pcsQRa[0], pcsQRb[0], qecQRc[0], qecQRc[1],
                pcsQRa[1], pcsQRb[1]])
    #PCS
    qecCirc.barrier()
    qecCirc.cx(pcsQRa[1], qecDataQRa[1])
    qecCirc.cx(pcsQRa[0], qecQRc[0])
    qecCirc.cx(pcsQRb[1], qecDataQRb[1])
    qecCirc.cx(pcsQRb[0], qecQRc[1])
    qecCirc.h((pcsQRa[0], pcsQRa[1], pcsQRb[0], pcsQRb[1]))
    qecCirc.barrier()
    # QEC decode step
    qecCirc.cx(qecQRc[0],qecDataQRa[2])
    qecCirc.cx(qecQRc[0],qecDataQRa[1])
    qecCirc.cx(qecQRc[1],qecDataQRb[2])
    qecCirc.cx(qecQRc[1],qecDataQRb[1])
    qecCirc.barrier()
    # QEC correcting
    qecCirc.ccx(qecDataQRa[1], qecDataQRa[2], qecQRc[0])
    qecCirc.ccx(qecDataQRb[1], qecDataQRb[2], qecQRc[1])
    qecCirc.barrier()
    # Measurements
    qecCirc.measure(qecDataQRa[1], qecRepCRa[0])
    qecCirc.measure(qecDataQRa[2], qecRepCRa[1])
    qecCirc.measure(qecDataQRb[1], qecRepCRb[0])
    qecCirc.measure(qecDataQRb[2], qecRepCRb[1])

    qecCirc.barrier()
    # Entanglement swap
    qecCirc.cx(qecQRc[0], qecQRc[1])
    qecCirc.h(qecQRc[0])
    qecCirc.measure(qecQRc[0], qecCRc[0])
    qecCirc.measure(qecQRc[1], qecCRc[1])
    _add_fixing_gates(qecCirc, qecDataQRa, qecCRc, 0)
    # Measure pcs
    qecCirc.measure(pcsQRa[0], pcsCR[0])
    qecCirc.measure(pcsQRb[0], pcsCR[1])
    qecCirc.measure(pcsQRa[1], pcsCR[2])
    qecCirc.measure(pcsQRb[1], pcsCR[3])
   
    return qecCirc, qecDataQRa, qecRepCRa, qecDataQRb, qecRepCRb, qecEPRCR

def createPCSSepQedRepCirc(reps):
    '''Creates a qec + pcs x checks verion of the entanglement swapping scenario a-c-b. the x-checks
     are conditioned only one acilla. one half of each of the a and b local epr pairs are encoded in 
     the repetition code and protected by all x checks with one ancilla for a and one ancilla for b. 
     The encoded qubits are sent to the c repeater which decodes, corrects, and 
     does the entanglement swapping.'''
    # ancillas + 2 qubits for local epr
    dataQubits=reps+2
    qecDataQRa=QuantumRegister(dataQubits, "a")
    qecDataQRb=QuantumRegister(dataQubits, "b")
    qecRepCRa=ClassicalRegister(reps, "rep_ancillas_a")
    qecRepCRb=ClassicalRegister(reps, "rep_ancillas_b")
    # qecQRc=QuantumRegister(2, "c")
    qecCRc=ClassicalRegister(2, "entanglement_swap")
    qecEPRCR=ClassicalRegister(2, "final_epr_pair")
    # zip(list(range(dataQubits))[1:], list(range(pcsCR.size)))
    pcsQRa=QuantumRegister(reps+1, "pcs_a")
    pcsQRb=QuantumRegister(reps+1, "pcs_b")
    pcsCR=ClassicalRegister(2*(reps+1), "pcs")
    qecCirc=QuantumCircuit(qecDataQRa, qecDataQRb, pcsQRa, pcsQRb, qecEPRCR, pcsCR, qecRepCRa, qecRepCRb, qecCRc)
    
    # create local epr pairs
    qecCirc.h(qecDataQRa[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQRa[0], qecDataQRa[1])
    qecCirc.h(qecDataQRb[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQRb[0], qecDataQRb[1])
    qecCirc.barrier()
    # QEC encoding
    ancil_reg_idxs=list(range(dataQubits))[2:]
    for idx in ancil_reg_idxs:
        qecCirc.cx(qecDataQRa[1], qecDataQRa[idx])
        qecCirc.cx(qecDataQRb[1], qecDataQRb[idx])
    qecCirc.barrier()
    #PCS
    for idx1, idx2 in zip(list(range(pcsQRa.size)), list(range(dataQubits))[1:]):
        qecCirc.h((pcsQRa[idx1], pcsQRb[idx1]))
        qecCirc.cx(pcsQRa[idx1], qecDataQRa[idx2])
        qecCirc.cx(pcsQRb[idx1], qecDataQRb[idx2])
    for idx in range(dataQubits)[1:]:
        qecCirc.id([qecDataQRa[idx], qecDataQRb[idx]])
    for idx in range(pcsQRa.size):
        qecCirc.id([pcsQRa[idx], pcsQRb[idx]])
    qecCirc.barrier()
    #PCS
    for idx1, idx2 in zip(list(range(pcsQRa.size)), list(range(dataQubits))[1:]):
        qecCirc.cx(pcsQRb[idx1], qecDataQRb[idx2])
        qecCirc.cx(pcsQRa[idx1], qecDataQRa[idx2])
        qecCirc.h((pcsQRa[idx1], pcsQRb[idx1]))
    # Measure pcs
    for idx1, idx2 in zip(list(range(pcsQRa.size)), list(range(pcsCR.size))):
        qecCirc.measure(pcsQRa[idx1], pcsCR[idx2])
        qecCirc.measure(pcsQRb[idx1], pcsCR[idx2+1])
    qecCirc.barrier()
    # QEC decode step
    for idx in ancil_reg_idxs:
        qecCirc.cx(qecDataQRa[1], qecDataQRa[idx])
        qecCirc.cx(qecDataQRb[1], qecDataQRb[idx])
    qecCirc.barrier()
    # Qec measurements
    for idx1, idx2 in zip(ancil_reg_idxs, list(range(reps))):
        qecCirc.measure(qecDataQRa[idx1], qecRepCRa[idx2])
        qecCirc.measure(qecDataQRb[idx1], qecRepCRb[idx2])
    qecCirc.barrier()
    # Entanglement swap
    qecCirc.cx(qecDataQRa[1], qecDataQRb[1])
    qecCirc.h(qecDataQRa[1])
    qecCirc.measure(qecDataQRa[1], qecCRc[0])
    qecCirc.measure(qecDataQRb[1], qecCRc[1])
    _add_fixing_gates(qecCirc, qecDataQRa, qecCRc, 0)
   
    return qecCirc, qecDataQRa, qecRepCRa, qecDataQRb, qecRepCRb, qecEPRCR

def createPCS1QedRepCirc(reps):
    '''Creates a qec + pcs x checks verion of the entanglement swapping scenario a-c-b. the x-checks
     are conditioned only one acilla. one half of each of the a and b local epr pairs are encoded in 
     the repetition code and protected by all x checks with one ancilla for a and one ancilla for b. 
     The encoded qubits are sent to the c repeater which decodes, corrects, and 
     does the entanglement swapping.'''
    # ancillas + 2 qubits for local epr
    dataQubits=reps+2
    qecDataQRa=QuantumRegister(dataQubits, "a")
    qecDataQRb=QuantumRegister(dataQubits, "b")
    qecRepCRa=ClassicalRegister(reps, "rep_ancillas_a")
    qecRepCRb=ClassicalRegister(reps, "rep_ancillas_b")
    # qecQRc=QuantumRegister(2, "c")
    qecCRc=ClassicalRegister(2, "entanglement_swap")
    qecEPRCR=ClassicalRegister(2, "final_epr_pair")
    pcsQRa=QuantumRegister(1, "pcs_a")
    pcsQRb=QuantumRegister(1, "pcs_b")
    pcsCR=ClassicalRegister(2, "pcs")
    qecCirc=QuantumCircuit(qecDataQRa, qecDataQRb, pcsQRa, pcsQRb, qecEPRCR, pcsCR, qecRepCRa, qecRepCRb, qecCRc)
    #PCS
    qecCirc.h((pcsQRa[0], pcsQRb[0]))
    qecCirc.cx(pcsQRa[0], qecDataQRa[1])
    qecCirc.cx(pcsQRb[0], qecDataQRb[1])
    #QEC
    # _createQECSubcirc(qecCirc, qecDataQRa, qecRepCRa, qecQRc[0])
    # _createQECSubcirc(qecCirc, qecDataQRb, qecRepCRb, qecQRc[1])
    # create local epr pairs
    qecCirc.h(qecDataQRa[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQRa[0], qecDataQRa[1])
    qecCirc.h(qecDataQRb[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQRb[0], qecDataQRb[1])
    qecCirc.barrier()
    # QEC encoding
    ancil_reg_idxs=list(range(dataQubits))[2:]
    for idx in ancil_reg_idxs:
        qecCirc.cx(qecDataQRa[1], qecDataQRa[idx])
        qecCirc.cx(qecDataQRb[1], qecDataQRb[idx])
    qecCirc.barrier()
    # for idx in list(range(1,3)):
    #     qecCirc.id(qecDataQR[idx])
    for idx in range(dataQubits)[1:]:
        qecCirc.id([qecDataQRa[idx], qecDataQRb[idx]])
    qecCirc.id([pcsQRa[0], pcsQRb[0]])
    qecCirc.barrier()
    # QEC decode step
    for idx in ancil_reg_idxs:
        qecCirc.cx(qecDataQRa[1], qecDataQRa[idx])
        qecCirc.cx(qecDataQRb[1], qecDataQRb[idx])
    qecCirc.barrier()
    # QEC correcting
    # qecCirc.ccx(qecDataQRa[1], qecDataQRa[2], qecQRc[0])
    # qecCirc.ccx(qecDataQRb[1], qecDataQRb[2], qecQRc[1])
    # qecCirc.barrier()
    # Qec measurements
    for idx1, idx2 in zip(ancil_reg_idxs, list(range(reps))):
        qecCirc.measure(qecDataQRa[idx1], qecRepCRa[idx2])
        qecCirc.measure(qecDataQRb[idx1], qecRepCRb[idx2])

    #PCS
    qecCirc.barrier()
    qecCirc.cx(pcsQRa[0], qecDataQRa[1])
    qecCirc.cx(pcsQRb[0], qecDataQRb[1])
    qecCirc.h((pcsQRa[0], pcsQRb[0]))
    qecCirc.barrier()
    # Entanglement swap
    qecCirc.cx(qecDataQRa[1], qecDataQRb[1])
    qecCirc.h(qecDataQRa[1])
    qecCirc.measure(qecDataQRa[1], qecCRc[0])
    qecCirc.measure(qecDataQRb[1], qecCRc[1])
    _add_fixing_gates(qecCirc, qecDataQRa, qecCRc, 0)
    # Measure pcs
    qecCirc.measure(pcsQRa[0], pcsCR[0])
    qecCirc.measure(pcsQRb[0], pcsCR[1])
   
    return qecCirc, qecDataQRa, qecRepCRa, qecDataQRb, qecRepCRb, qecEPRCR

def createPCSRepCirc():
    '''Creates a qec + pcs x checks verion of the entanglement swapping scenario a-c-b. the x-checks
     are conditioned only one acilla. one half of each of the a and b local epr pairs are encoded in 
     the repetition code and protected by all x checks with one ancilla for a and one ancilla for b. 
     The encoded qubits are sent to the c repeater which decodes, corrects, and 
     does the entanglement swapping.'''
    qecDataQRa=QuantumRegister(3, "a")
    qecRepCRa=ClassicalRegister(2, "rep_ancillas_a")
    qecDataQRb=QuantumRegister(3, "b")
    qecRepCRb=ClassicalRegister(2, "rep_ancillas_b")
    qecQRc=QuantumRegister(2, "c")
    qecCRc=ClassicalRegister(2, "entanglement_swap")
    qecEPRCR=ClassicalRegister(2, "final_epr_pair")
    pcsQRa=QuantumRegister(1, "pcs_a")
    pcsQRb=QuantumRegister(1, "pcs_b")
    pcsCR=ClassicalRegister(2, "pcs")
    qecCirc=QuantumCircuit(qecDataQRa, qecDataQRb, qecQRc, pcsQRa, pcsQRb, qecEPRCR, pcsCR, qecRepCRa, qecRepCRb, qecCRc)
    #PCS
    qecCirc.h((pcsQRa[0], pcsQRb[0]))
    qecCirc.cx(pcsQRa[0], qecQRc[0])
    qecCirc.cx(pcsQRb[0], qecQRc[1])
    #QEC
    # _createQECSubcirc(qecCirc, qecDataQRa, qecRepCRa, qecQRc[0])
    # _createQECSubcirc(qecCirc, qecDataQRb, qecRepCRb, qecQRc[1])
    qecCirc.h(qecDataQRa[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQRa[0], qecQRc[0])
    qecCirc.h(qecDataQRb[0]) # this qubit will not travel to charlie.
    qecCirc.cx(qecDataQRb[0], qecQRc[1])
    qecCirc.barrier()
    # QEC encoding
    qecCirc.cx(qecQRc[0],qecDataQRa[1])
    qecCirc.cx(qecQRc[0],qecDataQRa[2])
    qecCirc.cx(qecQRc[1],qecDataQRb[1])
    qecCirc.cx(qecQRc[1],qecDataQRb[2]) 
    qecCirc.barrier()
    # for idx in list(range(1,3)):
    #     qecCirc.id(qecDataQR[idx])
    qecCirc.id([qecDataQRa[1], qecDataQRa[2], qecDataQRb[1], qecDataQRb[2], pcsQRa[0], pcsQRb[0], qecQRc[0], qecQRc[1]])
    qecCirc.barrier()
    # QEC decode step
    qecCirc.cx(qecQRc[0],qecDataQRa[2])
    qecCirc.cx(qecQRc[0],qecDataQRa[1])
    qecCirc.cx(qecQRc[1],qecDataQRb[2])
    qecCirc.cx(qecQRc[1],qecDataQRb[1])
    qecCirc.barrier()
    # QEC correcting
    qecCirc.ccx(qecDataQRa[1], qecDataQRa[2], qecQRc[0])
    qecCirc.ccx(qecDataQRb[1], qecDataQRb[2], qecQRc[1])
    qecCirc.barrier()
    # Measurements
    qecCirc.measure(qecDataQRa[1], qecRepCRa[0])
    qecCirc.measure(qecDataQRa[2], qecRepCRa[1])
    qecCirc.measure(qecDataQRb[1], qecRepCRb[0])
    qecCirc.measure(qecDataQRb[2], qecRepCRb[1])

    #PCS
    qecCirc.barrier()
    qecCirc.cx(pcsQRa[0], qecQRc[0])
    qecCirc.cx(pcsQRb[0], qecQRc[1])
    qecCirc.h((pcsQRa[0], pcsQRb[0]))
    qecCirc.barrier()
    # Entanglement swap
    qecCirc.cx(qecQRc[0], qecQRc[1])
    qecCirc.h(qecQRc[0])
    qecCirc.measure(qecQRc[0], qecCRc[0])
    qecCirc.measure(qecQRc[1], qecCRc[1])
    _add_fixing_gates(qecCirc, qecDataQRa, qecCRc, 0)
    # Measure pcs
    qecCirc.measure(pcsQRa[0], pcsCR[0])
    qecCirc.measure(pcsQRb[0], pcsCR[1])
   
    return qecCirc, qecDataQRa, qecRepCRa, qecDataQRb, qecRepCRb, qecEPRCR

def get_noise_model(errorp1=0.001):
    '''Creates the base noise model. Every gate that appears in the circuit should be
    here.'''
    # errorp1=0.0003
    errorp2=errorp1*10
    error1q=depolarizing_error(errorp1, 1)
    errorm=depolarizing_error(errorp1, 1)
    error2q=depolarizing_error(errorp2, 2)
    error3q=depolarizing_error(errorp2, 3)
    noise_model=NoiseModel()
    noise_model.add_all_qubit_quantum_error(error2q, ["cx", "cz"])
    noise_model.add_all_qubit_quantum_error(error3q, ["ccx"])
    noise_model.add_all_qubit_quantum_error(error1q, ["h", "x", "y", "z", "rz"])
    noise_model.add_all_qubit_quantum_error(errorm, ["measure"])
    return noise_model

# def add_fixing_gates(qc, qr, qr_bsm, cr, idx=0):
#     '''
#     Adds the local rotations to get the EPR pair between Alice and Bob.
#     The local rotations can be sent from Charlie to either Alice or Bob.

#     qc (QuantumCircuit): circuit to append gates to
#     qr (QuantumRegister): Either the qubit register for Alice or Bob.
#     qr_bsm (QuantumRegister): The qubits where BSM is performed for heralding.
#     cr (ClassicalRegister): Classical register containing the Heralded 
#     entanglement measurement. The right most register should be the
#     outcome of the qubit that has the Hadamard and Cnot control 
#     when doing the Bell measurement.
#     idx (int): index of qr to apply corrections to.
#     '''
#     # Use classical communication from Charlie to correct locally.
#     # Note that it is assumed that the right most bit had Hadamard act
#     # on it in the Bell measurement.
#     # Using if_test (note: I'm sure that this code or the Qiskit if_test is bugged.)
#     # # Phi^-
#     # with qc.if_test((cr, 0b01)):
#     #     qc.z(qr[idx])
#     # # Psi^+
#     # with qc.if_test((cr, 0b10)):
#     #     qc.x(qr[idx])
#     # # Psi^-
#     # with qc.if_test((cr, 0b11)):
#     #     qc.x(qr[idx])
#     #     qc.z(qr[idx])

#     # Phi^-: 01 syndrome
#     qc.barrier()
#     qc.x(qr_bsm[1])
#     qc.mcrz(-math.pi, qr_bsm, qr[idx])
#     qc.x(qr_bsm[1])
#     qc.barrier()
#     # Psi^+: 10 syndrome
#     qc.x(qr_bsm[0])
#     qc.mcx(qr_bsm, qr[idx])
#     qc.x(qr_bsm[0])
#     qc.barrier()
#     # Psi^-: 11
#     qc.mcx(qr_bsm, qr[idx])
#     qc.mcrz(-math.pi, qr_bsm, qr[idx])

def _add_fixing_gates(qc, qr, cr_bsm, idx=0):
    '''
    Adds the local rotations to get the EPR pair between Alice and Bob.
    The local rotations can be sent from Charlie to either Alice or Bob.

    qc (QuantumCircuit): circuit to append gates to
    qr (QuantumRegister): Either the qubit register for Alice or Bob.
    qr_bsm (QuantumRegister): The qubits where BSM is performed for heralding.
    idx (int): index of qr to apply corrections to.
    '''
    # Use classical communication from Charlie to correct locally.
    # Note that it is assumed that the left most bit had Hadamard act
    # on it in the Bell measurement.

    qc.barrier()
    # Phi^-: 10 syndrome
    # qc.x(qr_bsm[1])
    # qc.mcrz(-math.pi, qr_bsm, qr[idx])
    # qc.x(qr_bsm[1])
    # with qc.if_test((cr_bsm, 0b01)): #bits need to reversed because qiskit stores index 0 at the right.
    #     qc.z(qr[idx])
    # qc.barrier()
    # Psi^+: 01 syndrome
    # qc.x(qr_bsm[0])
    # qc.ccx(qr_bsm[0], qr_bsm[1], qr[idx])
    # qc.x(qr_bsm[0])
    # with qc.if_test((cr_bsm, 0b10)):
    #     qc.x(qr[idx])
    # qc.barrier()
    # Psi^-: 11
    # qc.ccx(qr_bsm[0], qr_bsm[1], qr[idx])
    # qc.mcrz(-math.pi, qr_bsm, qr[idx])
    # with qc.if_test((cr_bsm, 0b11)):
    #     qc.x(qr[idx])
    #     qc.z(qr[idx])
    with qc.switch(cr_bsm) as case:
        with case(0b01):
            qc.z(qr[idx])
        with case(0b10):
            qc.x(qr[idx])
        with case(0b11):
            qc.x(qr[idx])
            qc.z(qr[idx])
    


def meas_xx(qc, qr, cr, backend, shots):
    '''
    Performs the xx measurement on qr.

    qc (QuantumCircuit): circuit to append gates to
    qr (QuantumRegister): registers to measure in the xx basis
    cr (ClassicalRegister):

    return: the simulation result
    '''
    temp_qc=deepcopy(qc)
    temp_qc.h(qr)
    temp_qc.measure(qr[0], cr[0])
    temp_qc.measure(qr[1], cr[1])
    # temp_qc=transpile(temp_qc, backend, optimization_level=0, basis_gates=BASIS)
    # temp_qc=transpile(temp_qc, backend, optimization_level=0)
    print("xx circ: ")
    print(temp_qc)
    print("xx circ: ", temp_qc.count_ops())
    return backend.run(temp_qc, shots=shots, optimization=0).result()

def meas_yy(qc, qr, cr, backend, shots):
    '''
    Performs the yy measurement on qr.

    qc (QuantumCircuit): circuit to append gates to
    qr (QuantumRegister): registers to measure in the yy basis
    cr (ClassicalRegister):

    return: the simulation result
    '''
    temp_qc=deepcopy(qc)
    temp_qc.s(qr)
    temp_qc.h(qr)
    temp_qc.measure(qr[0], cr[0])
    temp_qc.measure(qr[1], cr[1])
    # temp_qc=transpile(temp_qc, backend, optimization_level=0, basis_gates=BASIS)
    # temp_qc=transpile(temp_qc, backend, optimization_level=0)
    print("yy circ: ")
    print(temp_qc)
    print("yy circ: ", temp_qc.count_ops())
    return backend.run(temp_qc, shots=shots, optimization=0).result()

def meas_zz(qc, qr, cr, backend, shots):
    '''
    Performs the zz measurement on qr.

    qc (QuantumCircuit): circuit to append gates to
    qr (QuantumRegister): registers to measure in the zz basis
    cr (ClassicalRegister):

    return: the simulation result
    '''
    temp_qc=deepcopy(qc)
    temp_qc.measure(qr[0], cr[0])
    temp_qc.measure(qr[1], cr[1])
    # temp_qc=transpile(temp_qc, backend, optimization_level=0, basis_gates=BASIS)
    # temp_qc=transpile(temp_qc, backend, optimization_level=0)
    print("zz circ: ")
    print(temp_qc)
    print("zz circ: ", temp_qc.count_ops())
    return backend.run(temp_qc, shots=shots, optimization=0).result()

def repeated_pcs_circ_xz_checks(reps):
    '''a-c-b entanglement swapping. PCS checks are x and z checks.'''
    qr_a=QuantumRegister(1, "alice")
    qr_c_xpcs=QuantumRegister(2+2*reps, "charliex")
    qr_c_zpcs=QuantumRegister(2*reps, "charliez")
    qr_b=QuantumRegister(1, "bob")
    cr_epr=ClassicalRegister(2, "cr_epr")
    cr_c=ClassicalRegister(2*reps, "cr_pcs_ancillas")
    # cr_c=ClassicalRegister(4*reps, "cr_pcs_ancillas")
    cr_h=ClassicalRegister(2, "cr_heralded")

    qc=QuantumCircuit(qr_a, qr_b, qr_c_xpcs, qr_c_zpcs, cr_epr, cr_h, cr_c)

    # pcs x checks.
    c_qubits=qr_c_xpcs.size
    # print("size: ", c_qubits)
    for idx in list(range(c_qubits))[2:2+reps]:
        # print("idx1, ", idx)
        qc.h(qr_c_xpcs[idx])
        qc.cx(qr_c_xpcs[idx], qr_c_xpcs[0])
    for idx in list(range(c_qubits))[2+reps:]:
        # print("idx2, ", idx)
        qc.h(qr_c_xpcs[idx])
        qc.cx(qr_c_xpcs[idx], qr_c_xpcs[1])
    # # PCS X checks
    # qc.h([qr_c[3],qr_c[5]])
    # qc.cx(qr_c[3], qr_c[0])
    # qc.cx(qr_c[5], qr_c[1])
    # Bell states
    qc.h(qr_a[0])
    qc.cx(qr_a[0], qr_c_xpcs[0])
    qc.h(qr_b[0])
    qc.cx(qr_b[0], qr_c_xpcs[1])
    # speed of light 300,000km/s. 100/300,000=3*10^-4=0.3millisec
    qc.barrier()
    # PCS Z checks
    for idx in list(range(reps)): #z checks
        # print("idx1, ", idx)
        qc.h(qr_c_zpcs[idx])
        qc.cz(qr_c_zpcs[idx], qr_c_xpcs[0])
    for idx in list(range(2*reps))[reps:]: #z checks
        # print("idx1, ", idx)
        qc.h(qr_c_zpcs[idx])
        qc.cz(qr_c_zpcs[idx], qr_c_xpcs[1])
    # qc.h((qr_c[2], qr_c[4]))
    # qc.cz(qr_c[2], qr_c[0])
    # qc.cz(qr_c[4], qr_c[1])
    qc.barrier()
    for idx in list(range(qr_c_xpcs.size)):
        qc.id(qr_c_xpcs[idx])
    for idx in list(range(qr_c_zpcs.size)):
        qc.id(qr_c_zpcs[idx])
    qc.barrier()
    # PCS Z checks
    for idx in list(range(2*reps))[reps:][::-1]: #z checks
        # print("idx1, ", idx)
        qc.cz(qr_c_zpcs[idx], qr_c_xpcs[1])
        qc.h(qr_c_zpcs[idx])
    for idx in list(range(reps))[::-1]: #z checks
        # print("idx1, ", idx)
        qc.cz(qr_c_zpcs[idx], qr_c_xpcs[0])
        qc.h(qr_c_zpcs[idx])
    

    # qc.cz(qr_c[4], qr_c[1])
    # qc.cz(qr_c[2], qr_c[0])
    # qc.cx(qr_c[5], qr_c[1])
    # qc.cx(qr_c[3], qr_c[0])
    # qc.h((qr_c[2], qr_c[3], qr_c[4], qr_c[5]))
    # pcs left checks.
    # c_qubits=qr_c.size
    # pcs left checks.
    c_qubits=qr_c_xpcs.size
    # print("size: ", c_qubits)
    for idx in list(range(c_qubits))[2+reps:][::-1]:
        # print("idx2, ", idx)
        qc.cx(qr_c_xpcs[idx], qr_c_xpcs[1])
        qc.h(qr_c_xpcs[idx])
    for idx in list(range(c_qubits))[2:2+reps][::-1]:
        # print("idx1, ", idx)
        qc.cx(qr_c_xpcs[idx], qr_c_xpcs[0])
        qc.h(qr_c_xpcs[idx])

    # Entanglement swap
    qc.barrier()
    qc.cx(qr_c_xpcs[0], qr_c_xpcs[1])
    qc.h(qr_c_xpcs[0])
    qc.measure(qr_c_xpcs[0], cr_h[0])
    qc.measure(qr_c_xpcs[1], cr_h[1])
    # PCS measurements
    # for idx1, idx2 in zip(list(range(qr_c_xpcs.size))[2:], list(range(cr_c.size))[:2*reps]):
    #     qc.measure(qr_c_xpcs[idx1], cr_c[idx2])
    # for idx1, idx2 in zip(list(range(qr_c_zpcs.size)), list(range(cr_c.size))[2*reps:]):
    #     qc.measure(qr_c_zpcs[idx1], cr_c[idx2])    
    for idx1, idx2 in zip(list(range(qr_c_xpcs.size))[2:2+int(reps/2)], list(range(cr_c.size))[:int(reps/2)]):
        qc.measure(qr_c_xpcs[idx1], cr_c[idx2])
    for idx1, idx2 in zip(list(range(qr_c_xpcs.size))[2+reps:2+reps+int(reps/2)], list(range(cr_c.size))[int(reps/2):]):
        qc.measure(qr_c_xpcs[idx1], cr_c[idx2])
    for idx1, idx2 in zip(list(range(qr_c_zpcs.size))[:int(reps/2)], list(range(cr_c.size))[reps:]):
        qc.measure(qr_c_zpcs[idx1], cr_c[idx2])   
    for idx1, idx2 in zip(list(range(qr_c_zpcs.size))[reps: reps+int(reps/2)], list(range(cr_c.size))[reps+int(reps/2):]):
        qc.measure(qr_c_zpcs[idx1], cr_c[idx2])  
    # qc.measure(qr_c[3], cr_c[1])
    # qc.measure(qr_c[4], cr_c[2])
    # qc.measure(qr_c[5], cr_c[3])
    bsm_ancillas=qr_c_xpcs[:2]
    _add_fixing_gates(qc, qr_a, cr_h)
    return qc, qr_a, qr_b, cr_epr

def cat_pcs_circ_xz_checks(reps):
    '''a-c-b entanglement swapping. PCS checks are x and z checks.'''
    qr_a=QuantumRegister(1, "alice")
    qr_c_xpcs=QuantumRegister(2+2*reps, "charliex")
    qr_c_zpcs=QuantumRegister(2*reps, "charliez")
    qr_b=QuantumRegister(1, "bob")
    cr_epr=ClassicalRegister(2, "cr_epr")
    cr_c=ClassicalRegister(2*reps, "cr_pcs_ancillas")
    # cr_c=ClassicalRegister(4*reps, "cr_pcs_ancillas")
    cr_h=ClassicalRegister(2, "cr_heralded")

    qc=QuantumCircuit(qr_a, qr_b, qr_c_xpcs, qr_c_zpcs, cr_epr, cr_h, cr_c)

    # pcs x checks.
    x_qubits=qr_c_xpcs.size
    qc.h(qr_c_xpcs[2])
    qc.h(qr_c_xpcs[2+reps])
    # for idx in list(range(x_qubits))[3:2+reps]:
    #     qc.cx(qr_c_xpcs[2], qr_c_xpcs[idx])
    # for idx in list(range(x_qubits))[2+reps+1:]:
    #     qc.cx(qr_c_xpcs[2+reps], qr_c_xpcs[idx])

    qc.cx(qr_c_xpcs[2], qr_c_xpcs[0])
    qc.cx(qr_c_xpcs[2+reps], qr_c_xpcs[1])

    # Bell states
    qc.h(qr_a[0])
    qc.cx(qr_a[0], qr_c_xpcs[0])
    qc.h(qr_b[0])
    qc.cx(qr_b[0], qr_c_xpcs[1])
    # speed of light 300,000km/s. 100/300,000=3*10^-4=0.3millisec
    qc.barrier()
    # PCS Z checks
    qc.h(qr_c_zpcs[0])
    qc.h(qr_c_zpcs[reps])
    z_qubits=qr_c_zpcs.size
    # for idx in list(range(z_qubits))[1:reps]:
    #     qc.cx(qr_c_zpcs[0], qr_c_zpcs[idx])
    # for idx in list(range(z_qubits))[1+reps:]:
    #     qc.cx(qr_c_zpcs[reps], qr_c_zpcs[idx])
    qc.cz(qr_c_zpcs[0], qr_c_xpcs[0])
    qc.cz(qr_c_zpcs[reps], qr_c_xpcs[1])

    qc.barrier()
    for idx in list(range(qr_c_xpcs.size)):
        qc.id(qr_c_xpcs[idx])
    for idx in list(range(qr_c_zpcs.size)):
        qc.id(qr_c_zpcs[idx])
    qc.barrier()
    # PCS Z checks
    qc.cz(qr_c_zpcs[reps], qr_c_xpcs[1])
    qc.cz(qr_c_zpcs[0], qr_c_xpcs[0])
    qc.h(qr_c_zpcs[0])
    qc.h(qr_c_zpcs[reps])
    
    # pcs x checks.
    qc.cx(qr_c_xpcs[2], qr_c_xpcs[0])
    qc.cx(qr_c_xpcs[2+reps], qr_c_xpcs[1])
    qc.h(qr_c_xpcs[2])
    qc.h(qr_c_xpcs[2+reps])

    # Entanglement swap
    qc.barrier()
    qc.cx(qr_c_xpcs[0], qr_c_xpcs[1])
    qc.h(qr_c_xpcs[0])
    qc.measure(qr_c_xpcs[0], cr_h[0])
    qc.measure(qr_c_xpcs[1], cr_h[1])
    # PCS measurements
    # for idx1, idx2 in zip(list(range(qr_c_xpcs.size))[2:], list(range(cr_c.size))[:2*reps]):
    #     qc.measure(qr_c_xpcs[idx1], cr_c[idx2])
    # for idx1, idx2 in zip(list(range(qr_c_zpcs.size)), list(range(cr_c.size))[2*reps:]):
    #     qc.measure(qr_c_zpcs[idx1], cr_c[idx2])    
    for idx1, idx2 in zip(list(range(qr_c_xpcs.size))[2:2+int(reps/2)], list(range(cr_c.size))[:int(reps/2)]):
        qc.measure(qr_c_xpcs[idx1], cr_c[idx2])
    for idx1, idx2 in zip(list(range(qr_c_xpcs.size))[2+reps:2+reps+int(reps/2)], list(range(cr_c.size))[int(reps/2):]):
        qc.measure(qr_c_xpcs[idx1], cr_c[idx2])
    for idx1, idx2 in zip(list(range(qr_c_zpcs.size))[:int(reps/2)], list(range(cr_c.size))[reps:]):
        qc.measure(qr_c_zpcs[idx1], cr_c[idx2])   
    for idx1, idx2 in zip(list(range(qr_c_zpcs.size))[reps: reps+int(reps/2)], list(range(cr_c.size))[reps+int(reps/2):]):
        qc.measure(qr_c_zpcs[idx1], cr_c[idx2])  
    # qc.measure(qr_c[3], cr_c[1])
    # qc.measure(qr_c[4], cr_c[2])
    # qc.measure(qr_c[5], cr_c[3])
    bsm_ancillas=qr_c_xpcs[:2]
    _add_fixing_gates(qc, qr_a, cr_h)
    return qc, qr_a, qr_b, cr_epr

def repeated_pcs_circ_x_check(reps):
    '''a-c-b repeaters for entanglement swapping at c. x pauli checks are applied to flying qubits from
    a and b.'''
    qr_a=QuantumRegister(1, "alice")
    qr_c=QuantumRegister(2+reps*2, "charlie")
    qr_b=QuantumRegister(1, "bob")
    cr_epr=ClassicalRegister(2, "cr_epr")
    cr_c=ClassicalRegister(reps*2, "cr_pcs_ancillas")
    cr_h=ClassicalRegister(2, "cr_heralded")

    qc=QuantumCircuit(qr_a, qr_b, qr_c, cr_epr, cr_h, cr_c)

    # pcs left checks.
    c_qubits=qr_c.size
    # print("size: ", c_qubits)
    for idx in list(range(c_qubits))[2:2+reps]:
        # print("idx1, ", idx)
        qc.h(qr_c[idx])
        qc.cx(qr_c[idx], qr_c[0])
    for idx in list(range(c_qubits))[2+reps:]:
        # print("idx2, ", idx)
        qc.h(qr_c[idx])
        qc.cx(qr_c[idx], qr_c[1])

    qc.barrier()
    #Bell states
    qc.h(qr_a[0])
    qc.cx(qr_a[0], qr_c[0] )
    qc.h((qr_b[0]))
    qc.cx(qr_b[0], qr_c[1])
    # speed of light 300,000km/s. 100/300,000=3*10^-4=0.3millisec
    qc.barrier()
    for idx in range(qr_c.size):
        qc.id(qr_c[idx])
    qc.barrier()
    # pcs right checks.
    for idx in list(range(c_qubits))[2+reps:][::-1]:
        qc.cx(qr_c[idx], qr_c[1])
        qc.h(qr_c[idx])
    for idx in list(range(c_qubits))[2:2+reps][::-1]:
        qc.cx(qr_c[idx], qr_c[0])
        qc.h(qr_c[idx])
    

    # qc.cx(qr_c[3], qr_c[1])
    # qc.cx(qr_c[2], qr_c[0])
    # qc.h((qr_c[2], qr_c[3]))
    qc.barrier()
    qc.cx(qr_c[0], qr_c[1])
    qc.h(qr_c[0])

    qc.measure(qr_c[0], cr_h[0])
    qc.measure(qr_c[1], cr_h[1])

    for idx1, idx2 in zip(list(range(c_qubits))[2:2+reps], list(range(reps))):
        qc.measure(qr_c[idx1], cr_c[idx2])
    for idx1, idx2 in zip(list(range(c_qubits))[2+reps:], list(range(reps,2*reps))):
        print("meas idx: ",  idx1, idx2)
        qc.measure(qr_c[idx1], cr_c[idx2])

    bsm_ancillas=qr_c[:2]
    _add_fixing_gates(qc, qr_a, cr_h)
    return qc, qr_a, qr_b, cr_epr

def heralded_pcs_circ_x_check():
    '''a-c-b repeaters for entanglement swapping at c. x pauli checks are applied to flying qubits from
    a and b.'''
    qr_a=QuantumRegister(1, "alice")
    qr_c=QuantumRegister(4, "charlie")
    qr_b=QuantumRegister(1, "bob")
    cr_epr=ClassicalRegister(2, "cr_epr")
    cr_c=ClassicalRegister(2, "cr_pcs_ancillas")
    cr_h=ClassicalRegister(2, "cr_heralded")

    qc=QuantumCircuit(qr_a, qr_b, qr_c, cr_epr, cr_h, cr_c)

    # pcs left checks.
    qc.h((qr_c[2], qr_c[3]))
    qc.cx(qr_c[2], qr_c[0])
    qc.cx(qr_c[3], qr_c[1])
    qc.barrier()
    #Bell states
    qc.h(qr_a[0])
    qc.cx(qr_a[0], qr_c[0] )
    qc.h((qr_b[0]))
    qc.cx(qr_b[0], qr_c[1])
    # speed of light 300,000km/s. 100/300,000=3*10^-4=0.3millisec
    qc.barrier()
    qc.id((qr_c[0],qr_c[1],qr_c[2],qr_c[3]))
    qc.barrier()
    qc.cx(qr_c[3], qr_c[1])
    qc.cx(qr_c[2], qr_c[0])
    qc.h((qr_c[2], qr_c[3]))
    qc.barrier()
    qc.cx(qr_c[0], qr_c[1])
    qc.h(qr_c[0])

    qc.measure(qr_c[0], cr_h[0])
    qc.measure(qr_c[1], cr_h[1])
    qc.measure(qr_c[2], cr_c[0])
    qc.measure(qr_c[3], cr_c[1])
    bsm_ancillas=qr_c[:2]
    _add_fixing_gates(qc, qr_a, cr_h)
    return qc, qr_a, qr_b, cr_epr

def heralded_pcs_circ_y_check():
    '''a-c-b repeaters for entanglement swapping at c. y pauli checks are applied to flying qubits from
    a and b.'''
    qr_a=QuantumRegister(1, "alice")
    qr_c=QuantumRegister(4, "charlie")
    qr_b=QuantumRegister(1, "bob")
    cr_epr=ClassicalRegister(2, "cr_epr")
    cr_c=ClassicalRegister(2, "cr_pcs_ancillas")
    cr_h=ClassicalRegister(2, "cr_heralded")

    qc=QuantumCircuit(qr_a, qr_b, qr_c, cr_epr, cr_h, cr_c)

    #Bell states
    qc.h(qr_a[0])
    qc.cx(qr_a[0], qr_c[0] )
    qc.h((qr_b[0]))
    qc.cx(qr_b[0], qr_c[1])
    # speed of light 300,000km/s. 100/300,000=3*10^-4=0.3millisec
    qc.barrier()
    qc.h((qr_c[2], qr_c[3]))
    qc.cy(qr_c[2], qr_c[0])
    qc.cy(qr_c[3], qr_c[1])
    qc.id((qr_c[0],qr_c[1],qr_c[2],qr_c[3]))
    qc.cy(qr_c[3], qr_c[1])
    qc.cy(qr_c[2], qr_c[0])
    qc.h((qr_c[2], qr_c[3]))
    qc.barrier()
    qc.cx(qr_c[0], qr_c[1])
    qc.h(qr_c[0])

    qc.measure(qr_c[0], cr_h[0])
    qc.measure(qr_c[1], cr_h[1])
    qc.measure(qr_c[2], cr_c[0])
    qc.measure(qr_c[3], cr_c[1])
    bsm_ancillas=qr_c[:2]
    _add_fixing_gates(qc, qr_a, cr_h)
    return qc, qr_a, qr_b, cr_epr

def heralded_pcs_circ_z_check():
    '''a-c-b repeaters for entanglement swapping at c. z pauli checks are applied to flying qubits from
    a and b.'''
    qr_a=QuantumRegister(1, "alice")
    qr_c=QuantumRegister(4, "charlie")
    qr_b=QuantumRegister(1, "bob")
    cr_epr=ClassicalRegister(2, "cr_epr")
    cr_c=ClassicalRegister(2, "cr_pcs_ancillas")
    cr_h=ClassicalRegister(2, "cr_heralded")

    qc=QuantumCircuit(qr_a, qr_b, qr_c, cr_epr, cr_h, cr_c)

    #Bell states
    qc.h(qr_a[0])
    qc.cx(qr_a[0], qr_c[0] )
    qc.h((qr_b[0]))
    qc.cx(qr_b[0], qr_c[1])
    # speed of light 300,000km/s. 100/300,000=3*10^-4=0.3millisec
    qc.barrier()
    qc.h((qr_c[2], qr_c[3]))
    qc.cz(qr_c[2], qr_c[0])
    qc.cz(qr_c[3], qr_c[1])
    qc.id((qr_c[0],qr_c[1],qr_c[2],qr_c[3]))
    qc.cz(qr_c[3], qr_c[1])
    qc.cz(qr_c[2], qr_c[0])
    qc.h((qr_c[2], qr_c[3]))
    qc.barrier()
    qc.cx(qr_c[0], qr_c[1])
    qc.h(qr_c[0])

    qc.measure(qr_c[0], cr_h[0])
    qc.measure(qr_c[1], cr_h[1])
    qc.measure(qr_c[2], cr_c[0])
    qc.measure(qr_c[3], cr_c[1])
    bsm_ancillas=qr_c[:2]
    _add_fixing_gates(qc, qr_a, cr_h)
    return qc, qr_a, qr_b, cr_epr

# def heralded_circ_depol(error):
#     antiCommSet=["I","X"]
#     numOps=len(antiCommSet)
#     print(antiCommSet)
#     # error=1
#     # antiCommSet=[math.sqrt(1-error)*Pauli(elem).to_matrix() if idx==0 else math.sqrt(error)*Pauli(elem).to_matrix() for idx, elem in enumerate(antiCommSet)]
#     antiCommSet=[math.sqrt(1-error)*Pauli(elem).to_matrix() if idx==0 else math.sqrt(error/(numOps-1))*Pauli(elem).to_matrix() for idx, elem in enumerate(antiCommSet)]
#     # print(type(antiCommSet[0]))
#     testset=[elem.conj().T @ elem for elem in antiCommSet]
#     # print(sum(testset))
#     krausOps=antiCommSet

#     QuantumError.atol = 1E-04
#     error_map=QuantumError(Kraus(krausOps)).to_instruction()
#     qr_a=QuantumRegister(2, "alice")
#     # qr_c=QuantumRegister(4, "charlie")
#     qr_b=QuantumRegister(2, "bob")
#     cr_epr=ClassicalRegister(2, "cr_epr")
#     cr_h=ClassicalRegister(2, "cr_heralded")

#     qc=QuantumCircuit(qr_a, qr_b, cr_epr, cr_h)
#     # Bell States
#     qc.h(qr_a[1])
#     qc.cx(qr_a[1], qr_a[0])
#     qc.h(qr_b[1])
#     qc.cx(qr_b[1], qr_b[0])
#     # Channel
#     qc.id((qr_a[0],qr_b[0]))
#     qc.append(error_map, [qr_a[0]])
#     qc.append(error_map, [qr_b[0]])
#     # Entanglement swap
#     qc.cx(qr_a[0], qr_b[0])
#     qc.h(qr_a[0])
#     qc.barrier()
#     qc.measure(qr_a[0], cr_h[0])
#     qc.measure(qr_b[0], cr_h[1])
#     bsm_ancillas=[qr_a[0], qr_b[0]]
#     _add_fixing_gates(qc, qr_a, bsm_ancillas, 1)
#     return qc, qr_a, qr_b, cr_epr

# def heraldedPSCcircRedundantZChecksDepol(error):
#     # error=np.longdouble(error)
#     # noise_model=NoiseModel()
#     # print(f"depolarizing error: {error}")
#     # prob1=error
#     # prob2=10*prob1
#     # error_id=phase_damping_error(1, 1)
#     # errorData=depolarizing_error(0.3, 1)
#     # errorAncilla=depolarizing_error(prob1, 1)

#     # antiCommSet=["II","IX", "IY", "XX", "XY", "YI", "YZ", "ZI", "ZZ"]
#     antiCommSetData=["I","X"]
#     antiCommSetAncilla=["I","Z"]

#     numOps=len(antiCommSetData)
#     # print(antiCommSet)
#     # error=1
#     # antiCommSet=[math.sqrt(1-error)*Pauli(elem).to_matrix() if idx==0 else math.sqrt(error)*Pauli(elem).to_matrix() for idx, elem in enumerate(antiCommSet)]
#     antiCommSetData=[math.sqrt(1-error)*Pauli(elem).to_matrix() if idx==0 else math.sqrt(error/(numOps-1))*Pauli(elem).to_matrix() for idx, elem in enumerate(antiCommSetData)]
#     antiCommSetAncilla=[math.sqrt(1-error)*Pauli(elem).to_matrix() if idx==0 else math.sqrt(error/(numOps-1))*Pauli(elem).to_matrix() for idx, elem in enumerate(antiCommSetAncilla)]

#     # print(type(antiCommSet[0]))
#     # testset=[elem.conj().T @ elem for elem in antiCommSet]
#     # print(sum(testset))
#     # krausOps=antiCommSet
#     krausOpsData=antiCommSetData
#     krausOpsAncilla=antiCommSetAncilla

#     QuantumError.atol = 1E-04
#     # error_map=QuantumError(Kraus(krausOps)).to_instruction()
#     error_mapData=QuantumError(Kraus(krausOpsData)).to_instruction()
#     error_mapAncilla=QuantumError(Kraus(krausOpsAncilla)).to_instruction()

#     # noise_model.add_all_qubit_quantum_error(errorData, ["id"], list(range(2,4)))
#     # noise_model.add_all_qubit_quantum_error(errorAncilla, ["id"], list(range(4,8)))
#     # for i in list(range(2,4)):
#     #         noise_model.add_quantum_error(error_map, ["id"], [i])
#     # for i in list(range(4,8)):
#     #         noise_model.add_quantum_error(error_map, ["id"], [i])

#     qr_a=QuantumRegister(1, "alice")
#     qr_c=QuantumRegister(6, "charlie")
#     qr_b=QuantumRegister(1, "bob")
#     cr_epr=ClassicalRegister(2, "cr_epr")
#     cr_c=ClassicalRegister(4, "cr_ancilla")
#     cr_h=ClassicalRegister(2, "cr_heralded")

#     qc=QuantumCircuit(qr_a, qr_b, qr_c, cr_epr, cr_h, cr_c)

#     # Initial graph states
#     qc.h((qr_a[0], qr_c[0]))
#     # qc.h(qr_a[0])
#     qc.cz(qr_a[0], qr_c[0])
#     qc.h((qr_b[0], qr_c[1]))
#     # qc.h(qr_b[0])
#     qc.cz(qr_b[0], qr_c[1])
#     qc.barrier()
#     qc.h((qr_c[2], qr_c[3], qr_c[4], qr_c[5]))
#     qc.cz(qr_c[2], qr_c[0])
#     qc.cz(qr_c[3], qr_c[0])
#     qc.cz(qr_c[4], qr_c[1])
#     qc.cz(qr_c[5], qr_c[1])
#     qc.append(error_mapData, [qr_c[0]])
#     qc.barrier()
#     qc.id((qr_c[0],qr_c[1],qr_c[2],qr_c[3], qr_c[4], qr_c[5]))
#     # qc.append(error_map, [qr_c[2], qr_c[0]])
#     # qc.append(error_map, [qr_c[3], qr_c[0]])
#     # qc.append(error_map, [qr_c[4], qr_c[1]])
#     # qc.append(error_map, [qr_c[5], qr_c[1]])
#     ##############
#     qc.append(error_mapData, [qr_c[0]])
#     qc.append(error_mapAncilla, [qr_c[2]])
#     qc.append(error_mapAncilla, [qr_c[3]])
#     qc.append(error_mapData, [qr_c[1]])
#     qc.append(error_mapAncilla, [qr_c[4]])
#     qc.append(error_mapAncilla, [qr_c[5]])
#     # qc.id((qr_c[0],qr_c[1]))
#     qc.barrier()
#     qc.cz(qr_c[5], qr_c[1])
#     # qc.cz(qr_c[4], qr_c[1])
#     qc.cz(qr_c[3], qr_c[0])
#     # qc.cz(qr_c[2], qr_c[0])
#     # qc.h((qr_c[2], qr_c[3], qr_c[4], qr_c[5]))
#     qc.h((qr_c[3], qr_c[5]))
    
#     # qc.measure(qr_c[2], cr_c[0])
#     qc.measure(qr_c[3], cr_c[1])
#     # qc.measure(qr_c[4], cr_c[2])
#     qc.measure(qr_c[5], cr_c[3])

#     # # Dependent Z PCS
#     # with qc.if_test((cr_c[3],1)) as else_:
#     #     qc.cz(qr_c[4], qr_c[1])
#     #     qc.h(qr_c[4])
#     #     qc.measure(qr_c[4], cr_c[2])
#     # with else_:
#     #     qc.measure(qr_c[4], cr_c[2])
#     #     with qc.if_test((cr_c[2],1)) as else_:
#     #         qc.z(qr_c[1])
#     #     with else_:
#     #         print("passing")

#     # with qc.if_test((cr_c[1],1)) as else_:
#     #     qc.cz(qr_c[2], qr_c[0])
#     #     qc.h(qr_c[2])
#     #     qc.measure(qr_c[2], cr_c[0])
#     # with else_:
#     #     qc.measure(qr_c[2], cr_c[0])
#     #     with qc.if_test((cr_c[0],1)) as else_:
#     #         qc.z(qr_c[0])
#     #     with else_:
#     #         print("passing")

#     qc.cz(qr_c[2], qr_c[0])
#     # qc.cz(qr_c[3], qr_c[0])
#     qc.cz(qr_c[4], qr_c[1])
#     qc.h((qr_c[2], qr_c[4]))
#     # qc.cz(qr_c[5], qr_c[1])

#     # # Redundant checks.
#     # # target qr_c[1]
#     # qc.ccz(qr_c[5], qr_c[4], qr_c[1])
#     # qc.ch(qr_c[5], qr_c[4])
#     # qc.x(qr_c[5])
#     # qc.ccz(qr_c[5], qr_c[4], qr_c[1])
#     qc.measure(qr_c[4], cr_c[2])
#     # # target qr_c[0]
#     # qc.ccz(qr_c[3], qr_c[2], qr_c[0])
#     # qc.ch(qr_c[3], qr_c[2])
#     # qc.x(qr_c[3])
#     # qc.ccz(qr_c[3], qr_c[2], qr_c[0])
#     qc.measure(qr_c[2], cr_c[0])
#     # Convert the graph states to Bell States    
#     qc.h((qr_c[0], qr_c[1]))

#     # Entanglement swap
#     qc.barrier()
#     qc.cx(qr_c[0], qr_c[1])
#     qc.h(qr_c[0])
#     qc.measure(qr_c[0], cr_h[0])
#     qc.measure(qr_c[1], cr_h[1])
        
#     bsm_ancillas=qr_c[:2]
#     _add_fixing_gates(qc, qr_a, bsm_ancillas)
#     return qc, qr_a, qr_b, cr_epr

# def heraldedPSCcircRedundantZChecksGraphSt():
#     qr_a=QuantumRegister(1, "alice")
#     qr_c=QuantumRegister(6, "charlie")
#     qr_b=QuantumRegister(1, "bob")
#     cr_epr=ClassicalRegister(2, "cr_epr")
#     cr_c=ClassicalRegister(4, "cr_ancilla")
#     cr_h=ClassicalRegister(2, "cr_heralded")

#     qc=QuantumCircuit(qr_a, qr_b, qr_c, cr_epr, cr_h, cr_c)

#     # Initial graph states
#     qc.h((qr_a[0], qr_c[0]))
#     # qc.h(qr_a[0])
#     qc.cz(qr_a[0], qr_c[0])
#     qc.h((qr_b[0], qr_c[1]))
#     # qc.h(qr_b[0])
#     qc.cz(qr_b[0], qr_c[1])
#     qc.barrier()
#     qc.h((qr_c[2], qr_c[3], qr_c[4], qr_c[5]))
#     qc.cx(qr_c[2], qr_c[0])
#     qc.cx(qr_c[3], qr_c[0])
#     qc.cx(qr_c[4], qr_c[1])
#     qc.cx(qr_c[5], qr_c[1])
#     qc.barrier()
#     qc.id((qr_c[0],qr_c[1],qr_c[2],qr_c[3], qr_c[4], qr_c[5]))
#     # qc.id((qr_c[0],qr_c[1]))
#     qc.barrier()
#     qc.cx(qr_c[5], qr_c[1])
#     # qc.cz(qr_c[4], qr_c[1])
#     qc.cx(qr_c[3], qr_c[0])
#     # qc.cz(qr_c[2], qr_c[0])
#     # qc.h((qr_c[2], qr_c[3], qr_c[4], qr_c[5]))
#     qc.h((qr_c[3], qr_c[5]))
    
#     # qc.measure(qr_c[2], cr_c[0])
#     qc.measure(qr_c[3], cr_c[1])
#     # qc.measure(qr_c[4], cr_c[2])
#     qc.measure(qr_c[5], cr_c[3])

#     # # Dependent Z PCS
#     # with qc.if_test((cr_c[3],1)) as else_:
#     #     qc.cz(qr_c[4], qr_c[1])
#     #     qc.h(qr_c[4])
#     #     qc.measure(qr_c[4], cr_c[2])
#     # with else_:
#     #     qc.measure(qr_c[4], cr_c[2])
#     #     with qc.if_test((cr_c[2],1)) as else_:
#     #         qc.z(qr_c[1])
#     #     with else_:
#     #         print("passing")

#     # with qc.if_test((cr_c[1],1)) as else_:
#     #     qc.cz(qr_c[2], qr_c[0])
#     #     qc.h(qr_c[2])
#     #     qc.measure(qr_c[2], cr_c[0])
#     # with else_:
#     #     qc.measure(qr_c[2], cr_c[0])
#     #     with qc.if_test((cr_c[0],1)) as else_:
#     #         qc.z(qr_c[0])
#     #     with else_:
#     #         print("passing")

#     qc.cx(qr_c[2], qr_c[0])
#     # qc.cz(qr_c[3], qr_c[0])
#     qc.cx(qr_c[4], qr_c[1])
#     qc.h((qr_c[2], qr_c[4]))
#     # qc.cz(qr_c[5], qr_c[1])

#     # # Redundant checks.
#     # # target qr_c[1]
#     # qc.ccz(qr_c[5], qr_c[4], qr_c[1])
#     # qc.ch(qr_c[5], qr_c[4])
#     # qc.x(qr_c[5])
#     # qc.ccz(qr_c[5], qr_c[4], qr_c[1])
#     qc.measure(qr_c[4], cr_c[2])
#     # # target qr_c[0]
#     # qc.ccz(qr_c[3], qr_c[2], qr_c[0])
#     # qc.ch(qr_c[3], qr_c[2])
#     # qc.x(qr_c[3])
#     # qc.ccz(qr_c[3], qr_c[2], qr_c[0])
#     qc.measure(qr_c[2], cr_c[0])
#     # Convert the graph states to Bell States    
#     qc.h((qr_c[0], qr_c[1]))

#     # Entanglement swap
#     qc.barrier()
#     qc.cx(qr_c[0], qr_c[1])
#     qc.h(qr_c[0])
#     qc.measure(qr_c[0], cr_h[0])
#     qc.measure(qr_c[1], cr_h[1])
        
#     bsm_ancillas=qr_c[:2]
#     _add_fixing_gates(qc, qr_a, bsm_ancillas)
#     return qc, qr_a, qr_b, cr_epr

def heraldedPSCcircRedundantXChecks():
    qr_a=QuantumRegister(1, "alice")
    qr_c=QuantumRegister(6, "charlie")
    qr_b=QuantumRegister(1, "bob")
    cr_epr=ClassicalRegister(2, "cr_epr")
    cr_c=ClassicalRegister(4, "cr_ancilla")
    cr_h=ClassicalRegister(2, "cr_heralded")

    qc=QuantumCircuit(qr_a, qr_b, qr_c, cr_epr, cr_h, cr_c)

    # Initial graph states
    #PCS checks
    qc.h((qr_c[2], qr_c[3], qr_c[4], qr_c[5]))
    # qc.cx(qr_c[2], qr_c[0])
    # qc.cx(qr_c[3], qr_c[0])
    # qc.cx(qr_c[4], qr_c[1])
    # qc.cx(qr_c[5], qr_c[1])
    qc.cx(qr_c[2], qr_a[0])
    qc.cx(qr_c[3], qr_a[0])
    qc.cx(qr_c[4], qr_b[0])
    qc.cx(qr_c[5], qr_b[0])
    qc.barrier()
    qc.h((qr_a[0]))
    # qc.h(qr_a[0])
    qc.cx(qr_a[0], qr_c[0])
    qc.h((qr_b[0]))
    # qc.h(qr_b[0])
    qc.cx(qr_b[0], qr_c[1])
    
    qc.barrier()
    qc.id((qr_c[0],qr_c[1],qr_c[2],qr_c[3], qr_c[4], qr_c[5]))
    # qc.id((qr_c[0],qr_c[1]))
    qc.barrier()
    qc.cx(qr_c[5], qr_c[1])
    # qc.cz(qr_c[4], qr_c[1])
    qc.cx(qr_c[3], qr_c[0])
    # qc.cz(qr_c[2], qr_c[0])
    # qc.h((qr_c[2], qr_c[3], qr_c[4], qr_c[5]))
    qc.h((qr_c[3], qr_c[5]))
    
    # qc.measure(qr_c[2], cr_c[0])
    qc.measure(qr_c[3], cr_c[1])
    # qc.measure(qr_c[4], cr_c[2])
    qc.measure(qr_c[5], cr_c[3])

    qc.cx(qr_c[2], qr_c[0])
    # qc.cz(qr_c[3], qr_c[0])
    qc.cx(qr_c[4], qr_c[1])
    qc.h((qr_c[2], qr_c[4]))
    # qc.cz(qr_c[5], qr_c[1])

    qc.measure(qr_c[4], cr_c[2])
    qc.measure(qr_c[2], cr_c[0])
    # Convert the graph states to Bell States    
    # qc.h((qr_c[0], qr_c[1]))

    # Entanglement swap
    qc.barrier()
    qc.cx(qr_c[0], qr_c[1])
    qc.h(qr_c[0])
    qc.measure(qr_c[0], cr_h[0])
    qc.measure(qr_c[1], cr_h[1])
        
    bsm_ancillas=qr_c[:2]
    _add_fixing_gates(qc, qr_a, cr_h)
    return qc, qr_a, qr_b, cr_epr

# def heraldedPSCcircRedundantXZChecks():
#     qr_a=QuantumRegister(1, "alice")
#     qr_c=QuantumRegister(10, "charlie")
#     qr_b=QuantumRegister(1, "bob")
#     cr_epr=ClassicalRegister(2, "cr_epr")
#     cr_c=ClassicalRegister(8, "cr_ancilla")
#     cr_h=ClassicalRegister(2, "cr_heralded")

#     qc=QuantumCircuit(qr_a, qr_b, qr_c, cr_epr, cr_h, cr_c)

#     # Initial graph states
#     qc.h((qr_a[0], qr_c[0]))
#     qc.cz(qr_a[0], qr_c[0])
#     qc.h((qr_b[0], qr_c[1]))
#     qc.cz(qr_b[0], qr_c[1])
#     qc.barrier()
#     # Left PCS
#     qc.h((qr_c[2], qr_c[3], qr_c[4], qr_c[5], qr_c[6], qr_c[7], qr_c[8], qr_c[9]))
#     qc.cz(qr_c[2], qr_c[0])
#     qc.cz(qr_c[3], qr_c[0])
#     qc.cx(qr_c[4], qr_c[0])
#     qc.cx(qr_c[5], qr_c[0])
#     qc.cz(qr_c[6], qr_c[1])
#     qc.cz(qr_c[7], qr_c[1])
#     qc.cx(qr_c[8], qr_c[1])
#     qc.cx(qr_c[9], qr_c[1])
#     qc.id((qr_c[0],qr_c[1],qr_c[2],qr_c[3], qr_c[4], qr_c[5]))
#     # Right PCS
#     qc.cz(qr_c[2], qr_c[0])
#     qc.cx(qr_c[4], qr_c[0])
#     qc.cz(qr_c[6], qr_c[1])
#     qc.cx(qr_c[8], qr_c[1])
#     # Redundant checks.
#     # target: qr_c[0]
#     qc.ccz(qr_c[2], qr_c[3], qr_c[0])
#     qc.ch(qr_c[2], qr_c[3])
#     qc.x(qr_c[2])
#     qc.ccz(qr_c[2], qr_c[3], qr_c[0])
#     qc.ccx(qr_c[4], qr_c[5], qr_c[0])
#     qc.ch(qr_c[4], qr_c[5])
#     qc.x(qr_c[4])
#     qc.ccx(qr_c[4], qr_c[5], qr_c[0])
#     # target: qr_c[1]
#     qc.ccz(qr_c[6], qr_c[7], qr_c[1])
#     qc.ch(qr_c[6], qr_c[7])
#     qc.x(qr_c[6])
#     qc.ccz(qr_c[6], qr_c[7], qr_c[1])
#     qc.ccx(qr_c[8], qr_c[9], qr_c[1])
#     qc.ch(qr_c[8], qr_c[9])
#     qc.x(qr_c[8])
#     qc.ccx(qr_c[8], qr_c[9], qr_c[1])


#     #Convert the graph states to Bell States    
#     qc.h((qr_c[0], qr_c[1]))

#     # Entanglement swap
#     qc.barrier()
#     qc.cx(qr_c[0], qr_c[1])
#     qc.h(qr_c[0])
#     qc.measure(qr_c[0], cr_h[0])
#     qc.measure(qr_c[1], cr_h[1])

#     # measure
#     qc.measure(qr_c[2:], cr_c)
        
#     bsm_ancillas=qr_c[:2]
#     _add_fixing_gates(qc, qr_a, bsm_ancillas)
#     return qc, qr_a, qr_b, cr_epr

def heralded_pcs_circ_xz_checks():
    '''a-c-b entanglement swapping. PCS checks are x and z checks.'''
    qr_a=QuantumRegister(1, "alice")
    qr_c=QuantumRegister(6, "charlie")
    qr_b=QuantumRegister(1, "bob")
    cr_epr=ClassicalRegister(2, "cr_epr")
    cr_c=ClassicalRegister(4, "cr_pcs_ancillas")
    cr_h=ClassicalRegister(2, "cr_heralded")

    qc=QuantumCircuit(qr_a, qr_b, qr_c, cr_epr, cr_h, cr_c)

    # PCS X checks
    qc.h([qr_c[3],qr_c[5]])
    qc.cx(qr_c[3], qr_c[0])
    qc.cx(qr_c[5], qr_c[1])
    # Bell states
    qc.h(qr_a[0])
    qc.cx(qr_a[0], qr_c[0])
    qc.h(qr_b[0])
    qc.cx(qr_b[0], qr_c[1])
    # speed of light 300,000km/s. 100/300,000=3*10^-4=0.3millisec
    qc.barrier()
    # PCS Z checks
    qc.h((qr_c[2], qr_c[4]))
    qc.cz(qr_c[2], qr_c[0])
    qc.cz(qr_c[4], qr_c[1])
    qc.barrier()
    qc.id((qr_c[0],qr_c[1],qr_c[2],qr_c[3], qr_c[4], qr_c[5]))
    qc.barrier()
    qc.cz(qr_c[4], qr_c[1])
    qc.cz(qr_c[2], qr_c[0])
    qc.cx(qr_c[5], qr_c[1])
    qc.cx(qr_c[3], qr_c[0])
    qc.h((qr_c[2], qr_c[3], qr_c[4], qr_c[5]))
    # Entanglement swap
    qc.barrier()
    qc.cx(qr_c[0], qr_c[1])
    qc.h(qr_c[0])

    qc.measure(qr_c[0], cr_h[0])
    qc.measure(qr_c[1], cr_h[1])
    qc.measure(qr_c[2], cr_c[0])
    qc.measure(qr_c[3], cr_c[1])
    qc.measure(qr_c[4], cr_c[2])
    qc.measure(qr_c[5], cr_c[3])
    bsm_ancillas=qr_c[:2]
    _add_fixing_gates(qc, qr_a, cr_h)
    return qc, qr_a, qr_b, cr_epr

def heralded_circ():
    '''raw entanglment swapping.'''
    qr_a=QuantumRegister(2, "alice")
    # qr_c=QuantumRegister(4, "charlie")
    qr_b=QuantumRegister(2, "bob")
    cr_epr=ClassicalRegister(2, "cr_epr")
    cr_h=ClassicalRegister(2, "cr_heralded")

    qc=QuantumCircuit(qr_a, qr_b, cr_epr, cr_h)
    # Bell States
    qc.h(qr_a[1])
    qc.cx(qr_a[1], qr_a[0])
    qc.h(qr_b[1])
    qc.cx(qr_b[1], qr_b[0])
    # Channel
    qc.id((qr_a[0],qr_b[0]))
    # Entanglement swap
    qc.cx(qr_a[0], qr_b[0])
    qc.h(qr_a[0])
    qc.barrier()
    qc.measure(qr_a[0], cr_h[0])
    qc.measure(qr_b[0], cr_h[1])
    bsm_ancillas=[qr_a[0], qr_b[0]]
    _add_fixing_gates(qc, qr_a, cr_h, 1)
    return qc, qr_a, qr_b, cr_epr

# def kraus_conjsum(ops):
#     sum=np.zeros((2,2), dtype=np.complex_)
#     for op in ops:
#         sum+=op.T.conj() @ op
#     return sum

def collect_meas(results, epr_idx):
    '''Collects the measurements results that appear at epr_idx.
    args: results (dict): dictionary of the distribution
    epr_idx (int): the key is separated by spaces for each classicalregister.
    this number tells us which classicalregister to focus on when collapsing
    the distribution.
    returns: dict: collapsed results.'''
    collect_vals={}
    for key, val in results.items():
        key_epr=key.split(" ")[epr_idx]
        # print("key: ", key)
        # print("key epr: ", key_epr)

        collect_vals[key_epr]=collect_vals.get(key_epr, 0)+val
    print("collected vals: ", collect_vals)
    return collect_vals

def to_percent(result):
    '''converts dictionary distribution to percentages.
    args: result(dict): dictionary of distribution counts.'''
    total = sum(result.values())
    return {k: v/ total for k, v in result.items()}

def expect_of_vals(vals):
    '''Uses vals to calculate the expectation value. 
    expect=p(00)+p(11)-p(01)-p(10).'''
    return vals.get("00",0)+vals.get("11", 0)-vals.get("01", 0)-vals.get("10",0)

def fidelity_phip(expect_xx, expect_yy, expect_zz):
    '''fidelity=tr(rho*|phi+><phi+|).
    |phi+><phi+|=1/4(I+xx-yy+zz).'''
    return 1/4*(1+expect_xx-expect_yy+expect_zz)

def post_select(result, tot_ancillas, pcs_ancilla_pos):
    '''Post select based on PCS ancilla outcomes.
     args: result (dist): raw distribution
     tot_ancillas (int): the number of ancillas for all the pcs schemes.
     pcs_ancilla_pos (int): the position of the pcs classical registers.
     returns (dict): post selected dictionary'''
    filtered_result={}
    for key, value in result.items():
        # print(f"total ancillas: {tot_ancillas}")
        print(f"raw key value: {key}: {value}")
        ancilla_vals=key.split(" ")[pcs_ancilla_pos]
        print(f"pcs key: {ancilla_vals}")
        # print(f"key after split {key}")
        # compute_key=key[tot_ancillas+1:] #There is a space separating the ancillas from data
        # print(f"commpute key: {compute_key}")
        # val="0"*tot_ancillas
        # print(f"comparing {ancilla_vals} and {val}")
        if ancilla_vals=="0"*tot_ancillas:
            filtered_result[key]=filtered_result.get(key, 0)+value
            print(f"filtered results: {filtered_result}")
    return filtered_result

def post_select_some(result, tot_ancillas, pcs_ancilla_pos):
    '''Post select based on PCS ancilla outcomes.
     args: result (dist): raw distribution
     tot_ancillas (int): the number of ancillas for all the pcs schemes.
     pcs_ancilla_pos (int): the position of the pcs classical registers.
     returns (dict): post selected dictionary'''
    filtered_result={}
    for key, value in result.items():
        print(f"raw key value: {key}: {value}")
        ancilla_vals=key.split(" ")[pcs_ancilla_pos]
        print(f"pcs key: {ancilla_vals}")
        print(f"key after split {key}")
        # compute_key=key[tot_ancillas+1:] #There is a space separating the ancillas from data
        # print(f"commpute key: {compute_key}")
        print("aniclla pos: ", int(tot_ancillas/2))
        if getZeroWeight(ancilla_vals[:2:])==2 and getZeroWeight(ancilla_vals[2::])==2:
            filtered_result[key]=filtered_result.get(key, 0)+value
            print(f"filtered results: {filtered_result}")
    return filtered_result

def post_select_redundant(result, redundancy, pcs_ancilla_pos):
    '''Repeated measurement post selection. 
    args: 
    result (dict): the dictionary to filter
    redundancy (int): the number of times the same check repeats.
    pcs_ancilla_pos (int): position of the pcs classical registers'''
    filtered_result={}
    for key, value in result.items():
        ancilla_vals=key.split(" ")[pcs_ancilla_pos]
        print(f"raw key value: {key}: {value}")
        print(f"pcs key: {ancilla_vals}")
        # compute_key=key[tot_ancillas+1:] #There is a space separating the ancillas from data
        check=[getZeroWeight(chunk)>=1 for chunk in chunkwise(ancilla_vals, redundancy)]
        print(f"pcs redundant checks results: {check}")
        if all(check):
            filtered_result[key]=filtered_result.get(key, 0)+value
            print(f"filtered results: {filtered_result}")
    return filtered_result

def getZeroWeight(bitStr):
    '''counts the number of zero bits.'''
    return bitStr.count("0")


def chunkwise(data, l):
    '''Splits iterable into sublists of length l.'''
    for idx in range(0, len(data), l):
        print("yielding: ", data[idx:idx + l]) 
        yield data[idx:idx + l]