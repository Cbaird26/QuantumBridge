import streamlit as st

st.title("Quantum Computing Library Diagnostic")

libraries = {
    "Qiskit": "qiskit",
    "Cirq": "cirq",
    "Pennylane": "pennylane",
    "ProjectQ": "projectq",
    "PyQuil": "pyquil"
}

results = {}

for lib_name, lib_import in libraries.items():
    try:
        __import__(lib_import)
        results[lib_name] = "Success"
    except ImportError as e:
        results[lib_name] = f"Failed: {e}"

st.write("Diagnostic Results:")
for lib_name, result in results.items():
    st.write(f"{lib_name}: {result}")

st.write("Ensure you have the necessary libraries installed in your environment.")
