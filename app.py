import streamlit as st
import sys
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import Aer
import pennylane as qml
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI

# Set your OpenAI API key directly
API_KEY = "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"

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY
)

# Initialize Quantum Device
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

def run_quantum_circuit(params):
    result = circuit(params)
    return result

def optimize_params(model, X, y):
    model.fit(X, y)
    optimized_params = model.predict([[1, 2]])  # Example input, adjust as needed
    return optimized_params

def chat_with_gpt(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message['content'].strip()

# Streamlit app
st.title("QuantumBridge: AI-Powered Quantum Supercomputer with ChatGPT")
st.write("This app demonstrates quantum computations with AI and ChatGPT integration.")

param_0 = st.slider("RX rotation (radians)", 0.0, 6.28, 0.0)
param_1 = st.slider("RY rotation (radians)", 0.0, 6.28, 0.0)

qc = QuantumCircuit(2)
qc.rx(param_0, 0)
qc.ry(param_1, 1)
qc.cx(0, 1)
backend = Aer.get_backend("qasm_simulator")
qc_transpiled = transpile(qc, backend)
qobj = assemble(qc_transpiled)
job = backend.run(qobj)
result = job.result()
counts = result.get_counts()

st.write("Quantum Circuit:")
st.write(qc.draw())

st.write("Results:")
st.write(counts)
st.bar_chart([counts.get('00', 0), counts.get('01', 0), counts.get('10', 0), counts.get('11', 0)])

user_input = st.text_input("Ask GPT-3 a question about quantum computing or AI:")
if st.button("Ask GPT-3"):
    if user_input:
        gpt_response = chat_with_gpt(user_input)
        st.write("ChatGPT Response:", gpt_response)
    else:
        st.write("Please enter a question for GPT-3.")
