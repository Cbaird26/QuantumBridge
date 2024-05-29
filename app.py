import streamlit as st
from transformers import BloomTokenizerFast, AutoModelForCausalLM
from transformers.pipelines import pipeline

# Define the model name
model_name = "bigscience/bloom-560m"

# Function to load the tokenizer and model
def load_model(model_name):
    try:
        tokenizer = BloomTokenizerFast.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load the tokenizer and model
tokenizer, model = load_model(model_name)

# Check if the model is loaded successfully
if tokenizer and model:
    # Create the pipeline
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    # Streamlit app
    st.title("QuantumBridge Text Generator")
    prompt = st.text_input("Enter your prompt:", "Once upon a time")
    if st.button("Generate"):
        try:
            output = generator(prompt, max_length=50)
            st.write(output[0]['generated_text'])
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
else:
    st.error("Model loading failed. Please check the logs for more details.")
