# ---------------------------------------------------------
# quantum_money.py — BB84 Quantum Money Simulation
# ---------------------------------------------------------

from qiskit import QuantumCircuit
from qiskit_aer import Aer
import numpy as np
import random

# Backend
backend = Aer.get_backend("aer_simulator")


# ---------------------------------------------------------
# Minting the Quantum Banknote
# ---------------------------------------------------------

def mint_quantum_money(n=100):
    """
    Mint a quantum banknote consisting of n BB84 states.
    Returns:
        secret_record: list of (bit, basis)
        circuits: list of prepared QuantumCircuit objects
    """
    bits = np.random.randint(2, size=n)
    bases = np.random.choice(['Z', 'X'], size=n)
    circuits = []

    for bit, basis in zip(bits, bases):
        qc = QuantumCircuit(1, 1)
        if bit == 1:
            qc.x(0)
        if basis == 'X':
            qc.h(0)
        circuits.append(qc)

    secret_record = list(zip(bits, bases))  # Bank keeps classical info
    return secret_record, circuits


# ---------------------------------------------------------
# Verification of the Banknote
# ---------------------------------------------------------

def verify_banknote(secret_record, circuits):
    """
    Verify the quantum banknote using the secret classical record.
    Returns:
        fidelity: fraction of correctly measured qubits
    """
    correct = 0
    total = len(secret_record)

    for (bit, basis), qc in zip(secret_record, circuits):
        meas_qc = qc.copy()
        if basis == 'X':
            meas_qc.h(0)
        meas_qc.measure(0, 0)

        job = backend.run(meas_qc, shots=1)
        result = job.result().get_counts()
        measured = int(max(result, key=result.get))

        if measured == bit:
            correct += 1

    return correct / total


# ---------------------------------------------------------
# Forgery Attempt (Counterfeiter)
# ---------------------------------------------------------

def forge_banknote(circuits):
    """
    A counterfeiter attempts to copy the banknote by measuring qubits.
    They do NOT know the correct basis.
    Returns:
        forged_circuits: list of new quantum circuits
    """
    forged = []

    for qc in circuits:
        # Forger picks a random basis
        forged_basis = random.choice(['Z', 'X'])

        measure_qc = qc.copy()
        if forged_basis == 'X':
            measure_qc.h(0)
        measure_qc.measure(0, 0)

        job = backend.run(measure_qc, shots=1)
        result = job.result().get_counts()
        bit_guess = int(max(result, key=result.get))

        # Forger re-prepares the guessed qubit
        new_qc = QuantumCircuit(1, 1)
        if bit_guess == 1:
            new_qc.x(0)
        if forged_basis == 'X':
            new_qc.h(0)

        forged.append(new_qc)

    return forged


# ---------------------------------------------------------
# Demo function (for testing)
# ---------------------------------------------------------

def demo_quantum_money():
    print("\n=== Quantum Money Demo ===")

    secret, note = mint_quantum_money(n=100)
    auth_fidelity = verify_banknote(secret, note)
    print(f"Authentic note verification fidelity: {auth_fidelity:.4f}")

    forged_note = forge_banknote(note)
    forged_fidelity = verify_banknote(secret, forged_note)
    print(f"Forged note verification fidelity: {forged_fidelity:.4f}")

    print(f"Forgery error rate: {1 - forged_fidelity:.4f}")


if __name__ == "__main__":
    demo_quantum_money()
