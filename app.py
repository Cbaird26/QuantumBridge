import streamlit as st
import pennylane as qml
from transformers import BloomTokenizerFast, BloomForCausalLM

# Initialize Hugging Face Bloom model
model_name = "bigscience/bloom-560m"  # You can choose a different Bloom model if needed
tokenizer = BloomTokenizerFast.from_pretrained(model_name)
model = BloomForCausalLM.from_pretrained(model_name)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
st.write("Generate text using Hugging Face's Bloom model integrated with quantum computing")

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
            generated_text = generate_text(user_input)
            st.write("Generated Text:")
            st.write(generated_text)
    else:
        st.write("Please enter a prompt to generate text.")

# Display the quantum results
st.write("Quantum Computation has been applied.")
