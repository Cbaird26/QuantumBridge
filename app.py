import streamlit as st
import qiskit
import pandas as pd
import numpy as np

st.title("QuantumBridge")

st.write("""
## Welcome to QuantumBridge
This is a simple demonstration of a Streamlit app with Qiskit integration.
""")

# Example of a simple Qiskit circuit
st.write("### Quantum Circuit Example")
qr = qiskit.QuantumRegister(2)
cr = qiskit.ClassicalRegister(2)
circuit = qiskit.QuantumCircuit(qr, cr)

circuit.h(qr[0])
circuit.cx(qr[0], qr[1])
circuit.measure(qr, cr)

st.text(circuit.draw())

st.write("### Simulating the circuit")

backend = qiskit.Aer.get_backend('qasm_simulator')
result = qiskit.execute(circuit, backend).result()
counts = result.get_counts(circuit)

st.write(f"Counts: {counts}")

# Example DataFrame
st.write("### Example DataFrame")
data = {
    'Column 1': np.random.randn(10),
    'Column 2': np.random.randn(10)
}
df = pd.DataFrame(data)

st.dataframe(df)

st.write("### Example Chart")
st.line_chart(df)
