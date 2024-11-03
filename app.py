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
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None
if 'show_expander' not in st.session_state:
    st.session_state.show_expander = False

# Define categories
CATEGORIES = [
    "All Categories",
    "History",
    "Self-help",
    "Biography",
    "Science",
    "Non-fiction",
    "Fiction",
    "Business",
    "Health",
    "Memoir",
    "Thriller",
    "Classic",
    "Philosophy",
    "Fantasy",
    "Travel"
]

def load_data() -> pd.DataFrame:
    """Load book data from CSV."""
    try:
        # First try to load from the current directory
        if os.path.exists('data.csv'):
            return pd.read_csv('data.csv')
        
        # Then try to load from the app's directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(app_dir, 'data.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        
        st.error("data.csv not found. Please ensure the data file exists.")
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

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

def assistant_response(messages: List[Dict[str, str]], model: str):
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
                        chunk_content = chunk.choices[0].delta.get("content", "")
                        full_response += chunk_content
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
    
    if df.empty:
        st.warning("No data available. Please check if the data file exists and is properly formatted.")
        return

    # Create three columns for the filters
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        # Category filter dropdown
        selected_category = st.selectbox(
            "Select Category",
            options=CATEGORIES,
            index=0,
            key="category_filter"
        )

    with col2:
        # Author filter dropdown
        # Get unique authors and add "All Authors" option
        authors = ["All Authors"] + sorted(df['Author'].unique().tolist())
        selected_author = st.selectbox(
            "Select Author",
            options=authors,
            index=0,
            key="author_filter"
        )

    # Filter books based on selected category and author
    filtered_df = df.copy()
    if selected_category != "All Categories":
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    if selected_author != "All Authors":
        filtered_df = filtered_df[filtered_df['Author'] == selected_author]

    with col3:
        # Book selection dropdown (filtered by category and author)
        selected_book = st.selectbox(
            "Select Book",
            options=filtered_df['Title'].tolist(),
            index=0 if len(filtered_df) > 0 else None,
            placeholder="Choose a book..."
        )

    # Button to show the books list in an expander
    if st.button("Show Books List"):
        st.session_state.show_expander = not st.session_state.show_expander

    # Display expander with the books table
    if st.session_state.show_expander:
        with st.expander("Books List", expanded=True):
            st.dataframe(filtered_df[['Title', 'Category', 'Author', 'Summary']])

    if selected_book:
        # If the selected book has changed, reset the chat
        if st.session_state.selected_book != selected_book:
            st.session_state.selected_book = selected_book
            # Reset the chat messages
            st.session_state.messages = []
            # Get the book data
            book_data = df[df['Title'] == selected_book].iloc[0]
            # Initialize chat with new book summary
            initialize_chat(book_data['Summary'])
        else:
            # Get the book data
            book_data = df[df['Title'] == selected_book].iloc[0]

        # Create two tabs (removed Personalized Takeaway tab)
        tab1, tab2 = st.tabs(["Summary", "Chat"])

        # Display Summary tab
        with tab1:
            st.markdown("### Book Summary")
            st.write(book_data['Summary'])

        # Display Chat tab
        with tab2:
            st.markdown("### Chat about the Book")
            display_chat_interface()

if __name__ == "__main__":
    main()
