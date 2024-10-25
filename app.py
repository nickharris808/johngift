import streamlit as st
import pandas as pd
import openai
from typing import List, Dict
import os
openai.api_key = os.getenv("OPENAI_API_KEY")  
# Configure page settings
st.set_page_config(page_title="John's Books", layout="wide")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'openai_model' not in st.session_state:
    st.session_state.openai_model = "gpt-4o"

# Define categories
CATEGORIES = [
    "All Categories",
    "Self-help",
    "History",
    "Biography & Memoir",
    "Science",
    "Non-fiction",
    "Fiction",
    "Biography",
    "Health",
    "Business",
    "Economics",
    "Memoir",
    "Classic",
    "Thriller",
    "Philosophy",
    "Fantasy",
    "Historical",
    "Business & Economics",
    "Travel"
]

# Sample data in case CSV is not found
SAMPLE_DATA = {
    'Title': ["Atomic Habits", "Deep Work", "Think Again"],
    'Summary': [
        "A guide about building good habits and breaking bad ones.",
        "How to develop the ability to focus without distraction.",
        "The power of knowing what you don't know and how to rethink and unlearn."
    ],
    'Category': ["Self-help", "Business", "Non-fiction"],
    'Personalized Takeaway': [
        "Focus on building systems rather than setting goals.",
        "Schedule deep work sessions and protect them zealously.",
        "Embrace the joy of being wrong and learning from mistakes."
    ]
}

def load_data() -> pd.DataFrame:
    """Load book data from CSV or use sample data if file not found."""
    try:
        # First try to load from the current directory
        if os.path.exists('data.csv'):
            return pd.read_csv('data.csv')
        
        # Then try to load from the app's directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(app_dir, 'data.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        
        # If no CSV found, use sample data
        st.warning("data.csv not found. Using sample data for demonstration.")
        return pd.DataFrame(SAMPLE_DATA)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(SAMPLE_DATA)

def initialize_chat(book_summary: str):
    """Initialize chat with system prompt including book context."""
    system_prompt = {
        "role": "system",
        "content": f"""You are a knowledgeable assistant who has read and deeply understands this book. 
        Use the following summary as context for our discussion:
        {book_summary}
        
        Provide thoughtful, relevant responses based on the book's content and themes. 
        When appropriate, reference specific examples from the book to support your points."""
    }
    st.session_state.messages = [system_prompt]

def assistant_response(messages: List[Dict[str, str]], model: str) -> str:
    """Get response from OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stream=True
        )
        return response
    except Exception as e:
        st.error(f"Error getting response from OpenAI: {str(e)}")
        return None

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
            
            response = assistant_response(st.session_state.messages, st.session_state.openai_model)
            if response:
                try:
                    for chunk in response:
                        full_response += chunk.choices[0].delta.get("content", "")
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error processing response: {str(e)}")

def main():
    """Main application function."""
    st.title("John's Books")

    # Load data
    df = load_data()

    # Create two columns for the filters
    col1, col2 = st.columns([1, 2])

    with col1:
        # Category filter dropdown
        selected_category = st.selectbox(
            "Select Category",
            options=CATEGORIES,
            index=0,
            key="category_filter"
        )

    # Filter books based on selected category
    if selected_category == "All Categories":
        filtered_df = df
    else:
        filtered_df = df[df['Category'] == selected_category]

    with col2:
        # Book selection dropdown (filtered by category)
        selected_book = st.selectbox(
            "Select Book",
            options=filtered_df['Title'].tolist(),  # Changed from 'Book' to 'Title'
            index=None,
            placeholder="Choose a book..."
        )

    if selected_book:
        book_data = df[df['Title'] == selected_book].iloc[0]  # Changed from 'Book' to 'Title'

        # Create two tabs
        tab1, tab2 = st.tabs(["Summary", "Chat"])

        with tab1:
            st.markdown("### Book Summary")
            st.write(book_data['Summary'])

        with tab2:
            st.markdown("### Chat about the Book")
            if not st.session_state.messages:
                initialize_chat(book_data['Summary'])
            display_chat_interface()

if __name__ == "__main__":
    main()
