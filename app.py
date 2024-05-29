import streamlit as st
import pennylane as qml
from transformers import pipeline

# Initialize Hugging Face pipeline
model_name = "gpt2"  # You can replace this with another model if needed
generator = pipeline("text-generation", model=model_name)

# Define a simple quantum function using PennyLane
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def quantum_function(theta):
    qml.RX(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

# Streamlit app
st.title('Quantum Super Computer with Hugging Face')
st.write("Generate text using Hugging Face's GPT-2 model integrated with quantum computing")

# Quantum parameters
theta = [st.slider("Theta 0", 0.0, 3.14, 0.1), st.slider("Theta 1", 0.0, 3.14, 0.1)]

# Compute quantum function
quantum_result = quantum_function(theta)
st.write("Quantum Result:", quantum_result)

# Input from the user
user_input = st.text_input("Enter a prompt:", "")

# Generate text when the button is clicked
if st.button("Generate"):
    if user_input:
        with st.spinner("Generating text..."):
            generated_text = generator(user_input, max_length=100)
            st.write("Generated Text:")
            st.write(generated_text[0]['generated_text'])
    else:
        st.write("Please enter a prompt to generate text.")

# Display the quantum results
st.write("Quantum Computation has been applied.")
