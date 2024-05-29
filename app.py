import streamlit as st
import numpy as np
import pandas as pd

def try_import_qiskit():
    try:
        import qiskit
        return qiskit
    except ImportError as e:
        st.warning("Qiskit not available.")
        return None

def try_import_cirq():
    try:
        import cirq
        return cirq
    except ImportError as e:
        st.warning("Cirq not available.")
        return None

def try_import_pyquil():
    try:
        import pyquil
        from pyquil import Program
        return pyquil, Program
    except ImportError as e:
        st.warning("PyQuil not available.")
        return None, None

def try_import_pennylane():
    try:
        import pennylane as qml
        return qml
    except ImportError as e:
        st.warning("PennyLane not available.")
        return None

st.title("Quantum Computing with Multiple Libraries")

qiskit = try_import_qiskit()
cirq = try_import_cirq()
pyquil, Program = try_import_pyquil()
qml = try_import_pennylane()

if qiskit:
    st.write("Using Qiskit for quantum computing.")
    # Qiskit example code
    from qiskit import QuantumCircuit, transpile, Aer, execute

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator).result()
    counts = result.get_counts()

    st.write("Qiskit Result:", counts)

elif cirq:
    st.write("Using Cirq for quantum computing.")
    # Cirq example code
    qubit = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit(
        cirq.H(qubit),
        cirq.measure(qubit, key='m')
    )

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=100)
    counts = result.histogram(key='m')

    st.write("Cirq Result:", counts)

elif pyquil and Program:
    st.write("Using PyQuil for quantum computing.")
    # PyQuil example code
    program = Program('H 0', 'CNOT 0 1', 'MEASURE 0 0', 'MEASURE 1 1')
    from pyquil.api import get_qc
    qc = get_qc('2q-qvm')
    result = qc.run_and_measure(program, trials=10)

    st.write("PyQuil Result:", result)

elif qml:
    st.write("Using PennyLane for quantum computing.")
    # PennyLane example code
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    result = circuit()

    st.write("PennyLane Result:", result)

else:
    st.error("No quantum computing library available.")
