import streamlit as st
import pennylane as qml
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI
import os

# Set up OpenAI API key
api_key = "sk-proj-UJYGzw4N2bpofXQBgjzxT3BlbkFJas5lBlQTAVq48FQNAPlo"  # Replace this with the correct API key
client = OpenAI(api_key=api_key)

# Initialize Quantum Device
dev = qml.device("default.qubit", wires=2)

# Define a simple quantum circuit
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

# Optimize the quantum circuit parameters
def optimize_circuit(params):
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    steps = 100
    for i in range(steps):
        params = opt.step(circuit, params)
    return params

def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].text

st.title("QuantumBridge")

st.write("Welcome to QuantumBridge, where quantum computing meets AI.")

# Quantum Circuit Optimization Section
st.header("Optimize Quantum Circuit")
params = np.random.rand(2)
optimized_params = optimize_circuit(params)
st.write("Optimized parameters:", optimized_params)

# AI Chat Section
st.header("Chat with GPT AI")
user_input = st.text_input("Ask the AI a question about quantum computing:")
if st.button("Ask AI"):
    if user_input:
        ai_response = chat_with_gpt(user_input)
        st.write("AI Response:", ai_response)
    else:
        st.write("Please enter a question for the AI.")
