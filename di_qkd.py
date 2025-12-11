# ---------------------------------------------------------
# di_qkd.py — Device-Independent QKD (E91 CHSH Simulation)
# ---------------------------------------------------------

from qiskit import QuantumCircuit
from qiskit_aer import Aer
import numpy as np
import random
import math

# Backend
backend = Aer.get_backend("aer_simulator")


# ---------------------------------------------------------
# Generate Bell Pair
# ---------------------------------------------------------

def bell_pair():
    """Create a |Φ+⟩ Bell pair."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


# ---------------------------------------------------------
# Measurement in Arbitrary Basis (Angle on Bloch Sphere)
# ---------------------------------------------------------

def measure_in_basis(qc, qubit, angle):
    """Rotate by angle then measure, returning ±1."""
    meas = qc.copy()
    meas.ry(2 * angle, qubit)
    meas.measure(qubit, qubit)

    job = backend.run(meas, shots=1)
    result = job.result().get_counts()
    bit = int(max(result, key=result.get))

    return 1 - 2 * bit  # Map {0,1} → {+1,−1}


# ---------------------------------------------------------
# CHSH Bell Test
# ---------------------------------------------------------

def chsh_experiment(N=500, eve=False):
    """
    Perform CHSH inequality test using Bell pairs.
    eve=True simulates an intercept-resend attacker.
    """
    # E91 measurement settings
    a0 = 0
    a1 = math.pi / 4
    b0 = math.pi / 8
    b1 = -math.pi / 8

    settings = [(a0, b0), (a0, b1), (a1, b0), (a1, b1)]
    correlations = []

    for angleA, angleB in settings:
        E = 0
        for _ in range(N):
            qc = bell_pair()

            if eve:
                # Eve collapses qubit 0 by measuring in random basis
                eve_angle = random.choice([0, math.pi / 4])
                measure_in_basis(qc, 0, eve_angle)

            A = measure_in_basis(qc, 0, angleA)
            B = measure_in_basis(qc, 1, angleB)
            E += A * B

        correlations.append(E / N)

    # Compute CHSH S value:
    S = correlations[0] + correlations[1] + correlations[2] - correlations[3]
    return S


# ---------------------------------------------------------
# Demo run (for testing)
# ---------------------------------------------------------

def demo_diqkd():
    print("\n=== DI-QKD (E91 CHSH Test) ===")

    S_clean = chsh_experiment(N=300, eve=False)
    print(f"CHSH S-value without Eve:  {S_clean:.4f}")

    S_eve = chsh_experiment(N=300, eve=True)
    print(f"CHSH S-value with Eve:     {S_eve:.4f}")

    print("\nQuantum limit (Tsirelson bound): S = 2.828")
    print("Classical local realism bound:   S ≤ 2.0")
    print("If S > 2 → Bell violation → security guaranteed.")


if __name__ == "__main__":
    demo_diqkd()
