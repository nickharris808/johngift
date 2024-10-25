import streamlit as st
import pandas as pd
import openai
from typing import List, Dict

# Configure page settings
st.set_page_config(page_title="Shoe Felt Books", layout="wide")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'openai_model' not in st.session_state:
    st.session_state.openai_model = "gpt-4"

def load_data() -> pd.DataFrame:
    """Load and return the book data from CSV."""
    return pd.read_csv('data.csv')

def initialize_chat(book_summary: str):
    """Initialize chat with system prompt including book context."""
    system_prompt = {
        "role": "system",
        "content": f"""You are a knowledgeable assistant who has read and deeply understands this book. 
        Use the following summary as context for our discussion:
        {book_summary}
        
        Provide thoughtful, relevant responses based on the book's content and themes. 
        When appropriate, reference specific examples or quotes from the book to support your points."""
    }
    st.session_state.messages = [system_prompt]

def assistant_response(messages: List[Dict[str, str]], model: str) -> str:
    """Get response from OpenAI API."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=True
    )
    return response

def display_chat_interface():
    """Display and handle the chat interface."""
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("What would you like to ask about the book?")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for response in assistant_response(st.session_state.messages, st.session_state.openai_model):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

def main():
    """Main application function."""
    st.title("Shoe Felt Books")

    # Load data
    df = load_data()

    # Book selection
    selected_book = st.selectbox(
        "Select Book",
        options=df['Book'].tolist(),
        index=None,
        placeholder="Choose a book..."
    )

    if selected_book:
        book_data = df[df['Book'] == selected_book].iloc[0]

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary", 
            "Quotes", 
            "Applicable Advice",
            "Chat"
        ])

        with tab1:
            st.markdown("### Book Summary")
            st.write(book_data['Summary'])

        with tab2:
            st.markdown("### Notable Quotes")
            st.write(book_data['Quotes'])

        with tab3:
            st.markdown("### Applicable Advice")
            st.write(book_data['Personalized Takeaway'])

        with tab4:
            st.markdown("### Chat about the Book")
            if not st.session_state.messages:
                initialize_chat(book_data['Summary'])
            display_chat_interface()

if __name__ == "__main__":
    main()
