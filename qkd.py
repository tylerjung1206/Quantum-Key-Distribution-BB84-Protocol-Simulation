# ---------------------------------------------------------
# BB84 Quantum Key Distribution Simulation in Qiskit
# Works in VS Code or Jupyter Notebook
# ---------------------------------------------------------

from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
import numpy as np
import random
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def prepare_qubit(bit, basis):
    """
    Prepare a single BB84 qubit.
    bit: 0 or 1
    basis: 'Z' or 'X'
    """
    qc = QuantumCircuit(1, 1)

    # Encode bit
    if bit == 1:
        qc.x(0)

    # Change basis
    if basis == 'X':
        qc.h(0)

    return qc


def measure_qubit(qc, basis):
    """
    Measure a qubit in a given basis.
    """
    if basis == 'X':
        qc.h(0)
    qc.measure(0, 0)
    backend = Aer.get_backend("qasm_simulator")
    result = execute(qc, backend, shots=1).result()
    return int(result.get_counts().most_common()[0][0])


def eve_intercept(bit, basis, eve_prob):
    """
    Simulates Eve intercept-resend with probability eve_prob.
    """
    if random.random() > eve_prob:
        return None  # Eve does nothing

    eve_basis = random.choice(['Z', 'X'])
    eve_qc = prepare_qubit(bit, basis)
    eve_bit = measure_qubit(eve_qc, eve_basis)

    # Eve resends the qubit she *thinks* is correct
    resend_qc = prepare_qubit(eve_bit, eve_basis)
    return resend_qc


# ---------------------------------------------------------
# Main BB84 Simulation
# ---------------------------------------------------------

def run_bb84(N=200, eve_prob=0, noise_model=None):
    """
    Run BB84 simulation.
    eve_prob: probability Eve intercepts each qubit
    noise_model: Qiskit noise model
    """
    backend = Aer.get_backend("qasm_simulator")

    alice_bits = np.random.randint(2, size=N)
    alice_bases = np.random.choice(['Z', 'X'], size=N)
    bob_bases = np.random.choice(['Z', 'X'], size=N)

    bob_results = []

    for i in range(N):
        bit = alice_bits[i]
        a_basis = alice_bases[i]
        b_basis = bob_bases[i]

        # Prepare Alice's qubit
        qc = prepare_qubit(bit, a_basis)

        # Eve intercepts?
        eve_qc = eve_intercept(bit, a_basis, eve_prob)
        if eve_qc is not None:
            qc = eve_qc

        # Bob measures
        meas_qc = qc.copy()
        if b_basis == 'X':
            meas_qc.h(0)
        meas_qc.measure(0, 0)

        if noise_model:
            job = execute(meas_qc, backend, shots=1, noise_model=noise_model)
        else:
            job = execute(meas_qc, backend, shots=1)

        result = job.result()
        measured_bit = int(result.get_counts().most_common()[0][0])
        bob_results.append(measured_bit)

    # Sift key: keep only where bases match
    mask = alice_bases == bob_bases
    sifted_alice = alice_bits[mask]
    sifted_bob = np.array(bob_results)[mask]

    # Compute QBER
    errors = np.sum(sifted_alice != sifted_bob)
    QBER = errors / len(sifted_alice) if len(sifted_alice) > 0 else 0

    return QBER, len(sifted_alice)


# ---------------------------------------------------------
# Noise Model
# ---------------------------------------------------------

def make_noise_model(p=0.05):
    """
    Simple depolarizing noise model for BB84.
    """
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p, 1), ['u3'])
    return noise_model


# ---------------------------------------------------------
# Experiments for plotting
# ---------------------------------------------------------

def plot_eve_qber():
    eve_probs = np.linspace(0, 1, 10)
    qbers = []

    for p in eve_probs:
        QBER, _ = run_bb84(N=500, eve_prob=p)
        qbers.append(QBER)

    plt.figure(figsize=(8,5))
    plt.plot(eve_probs, qbers, marker='o')
    plt.xlabel("Eavesdropping Probability")
    plt.ylabel("QBER")
    plt.title("QBER vs Eve Intercept Probability (BB84)")
    plt.grid()
    plt.show()


def plot_noise_qber():
    noise_vals = np.linspace(0, 0.2, 10)
    qbers = []

    for p in noise_vals:
        noise = make_noise_model(p=p)
        QBER, _ = run_bb84(N=500, eve_prob=0, noise_model=noise)
        qbers.append(QBER)

    plt.figure(figsize=(8,5))
    plt.plot(noise_vals, qbers, marker='o', color="red")
    plt.xlabel("Noise Strength p")
    plt.ylabel("QBER")
    plt.title("QBER vs Channel Noise (Depolarizing)")
    plt.grid()
    plt.show()


# ---------------------------------------------------------
# Run All Experiments (Uncomment to use)
# ---------------------------------------------------------

if __name__ == "__main__":
    print("\nRunning ideal BB84 channel...")
    QBER, keylen = run_bb84(N=300, eve_prob=0)
    print(f"Ideal QBER = {QBER:.4f}, sifted key length = {keylen}")

    print("\nRunning with Eve intercepting 20% of qubits...")
    QBER, keylen = run_bb84(N=300, eve_prob=0.2)
    print(f"Eavesdropping QBER = {QBER:.4f}")

    print("\nRunning noisy channel...")
    noise = make_noise_model(p=0.05)
    QBER, keylen = run_bb84(N=300, eve_prob=0, noise_model=noise)
    print(f"Noisy channel QBER = {QBER:.4f}")

    # Plots
    plot_eve_qber()
    plot_noise_qber()

