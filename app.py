import streamlit as st
import pennylane as qml
from pennylane import numpy as np

# Create a quantum device
dev = qml.device('default.qubit', wires=2)

# Define a quantum node
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# Define the parameters
params = np.array([0.543, 0.123], requires_grad=True)

# Execute the quantum circuit
result = circuit(params)

# Streamlit app
st.title("Quantum Circuit Simulation with PennyLane")
st.write("Quantum Circuit Results:")
st.write(f"PauliZ expectation values: {result}")

# Allow user to change the parameters
param1 = st.slider("RX Parameter", min_value=0.0, max_value=2*np.pi, value=params[0])
param2 = st.slider("RY Parameter", min_value=0.0, max_value=2*np.pi, value=params[1])

# Recompute the result with new parameters
new_params = np.array([param1, param2], requires_grad=True)
new_result = circuit(new_params)
st.write(f"New PauliZ expectation values: {new_result}")
