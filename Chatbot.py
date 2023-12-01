# Streamlit UI
st.title("Question Answering with PaLM")

# User input for context
context = st.text_area("Context:", "The quick brown fox jumps over the lazy dog.")

# User input for message
message = st.text_input("Your question:", "What is the meaning of life?")

# Button to generate response
if st.button("Generate Response"):
    # Generate response using PaLMWrapper
    response = palm_wrapper.generate_response(context, message)

    # Display the response
    st.subheader("Response:")
    st.write(response)