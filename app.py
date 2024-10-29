import streamlit as st
import openai
import os
from typing import List, Dict
import base64
from io import BytesIO
from docx import Document
import PyPDF2

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to call OpenAI API
def assistant_response(messages: List[Dict[str, str]], model: str = "gpt-4"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error getting response from OpenAI: {str(e)}")
        return None

# Function to extract text from various file types
def extract_text_from_file(file):
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_extension in ['doc', 'docx']:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = file.read().decode('utf-8')
    
    return text

# Function to create a downloadable link for a string
def get_download_link(string, filename, text):
    b64 = base64.b64encode(string.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Demand Letter Generator
def demand_letter_generator():
    st.header("Demand Letter Generator")
    
    incident_date = st.date_input("Incident Date")
    incident_location = st.text_input("Incident Location")
    incident_type = st.selectbox("Incident Type", ["Car Accident", "Slip and Fall", "Other"])
    injury_description = st.text_area("Injury Description")
    supporting_docs = st.file_uploader("Upload Supporting Documents", accept_multiple_files=True)
    compensation_amount = st.number_input("Compensation Amount", min_value=0)
    sender_info = st.text_area("Sender Information")
    recipient_info = st.text_area("Recipient Information")
    
    if st.button("Generate Demand Letter"):
        prompt = f"""
        Generate a professional demand letter for a personal injury case with the following details:
        Incident Date: {incident_date}
        Incident Location: {incident_location}
        Incident Type: {incident_type}
        Injury Description: {injury_description}
        Compensation Amount: ${compensation_amount}
        Sender Information: {sender_info}
        Recipient Information: {recipient_info}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert legal assistant specializing in personal injury cases."},
            {"role": "user", "content": prompt}
        ]
        
        response = assistant_response(messages)
        
        if response:
            st.markdown("### Generated Demand Letter")
            st.write(response)
            st.markdown(get_download_link(response, "demand_letter.txt", "Download Demand Letter"), unsafe_allow_html=True)

# Counteroffer Drafting Tool
def counteroffer_drafting_tool():
    st.header("Counteroffer Drafting Tool")
    
    offer_letter = st.file_uploader("Upload Offer Letter", type=["pdf", "docx"])
    offered_amount = st.number_input("Offered Amount", min_value=0)
    points_of_disagreement = st.text_area("Points of Disagreement")
    new_evidence = st.file_uploader("Upload New Evidence", accept_multiple_files=True)
    counteroffer_amount = st.number_input("Proposed Counteroffer Amount", min_value=0)
    
    if st.button("Generate Counteroffer"):
        offer_text = extract_text_from_file(offer_letter) if offer_letter else ""
        
        prompt = f"""
        Generate a counteroffer letter based on the following:
        Original Offer: {offer_text}
        Offered Amount: ${offered_amount}
        Points of Disagreement: {points_of_disagreement}
        Proposed Counteroffer Amount: ${counteroffer_amount}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert legal assistant specializing in settlement negotiations."},
            {"role": "user", "content": prompt}
        ]
        
        response = assistant_response(messages)
        
        if response:
            st.markdown("### Generated Counteroffer Letter")
            st.write(response)
            st.markdown(get_download_link(response, "counteroffer_letter.txt", "Download Counteroffer Letter"), unsafe_allow_html=True)

# Email Template Generator
def email_template_generator():
    st.header("Email Template Generator")
    
    recipient_type = st.selectbox("Recipient Type", ["Insurer", "Witness", "Client", "Other"])
    subject_line = st.text_input("Subject Line")
    message_purpose = st.text_area("Message Purpose")
    case_context = st.text_area("Case Context")
    tone = st.radio("Tone", ["Formal", "Friendly"])
    
    if st.button("Generate Email Template"):
        prompt = f"""
        Generate an email template with the following details:
        Recipient Type: {recipient_type}
        Subject Line: {subject_line}
        Message Purpose: {message_purpose}
        Case Context: {case_context}
        Tone: {tone}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert legal assistant specializing in professional communication."},
            {"role": "user", "content": prompt}
        ]
        
        response = assistant_response(messages)
        
        if response:
            st.markdown("### Generated Email Template")
            st.write(response)
            st.markdown(get_download_link(response, "email_template.txt", "Download Email Template"), unsafe_allow_html=True)

# Police Statement Assistant
def police_statement_assistant():
    st.header("Police Statement Assistant")
    
    incident_date = st.date_input("Incident Date")
    incident_time = st.time_input("Incident Time")
    incident_location = st.text_input("Incident Location")
    incident_description = st.text_area("Incident Description")
    
    timeline = st.text_area("Timeline of Events (One event per line)")
    evidence = st.file_uploader("Upload Evidence", accept_multiple_files=True)
    witness_info = st.text_area("Witness Information")
    
    if st.button("Generate Police Statement"):
        prompt = f"""
        Generate a clear and accurate police statement based on the following information:
        Incident Date: {incident_date}
        Incident Time: {incident_time}
        Incident Location: {incident_location}
        Incident Description: {incident_description}
        Timeline of Events:
        {timeline}
        Witness Information: {witness_info}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert legal assistant specializing in creating accurate police statements."},
            {"role": "user", "content": prompt}
        ]
        
        response = assistant_response(messages)
        
        if response:
            st.markdown("### Generated Police Statement")
            st.write(response)
            st.markdown(get_download_link(response, "police_statement.txt", "Download Police Statement"), unsafe_allow_html=True)

# Medical Record Summarizer
def medical_record_summarizer():
    st.header("Medical Record Summarizer")
    
    medical_records = st.file_uploader("Upload Medical Records", accept_multiple_files=True)
    focus_conditions = st.multiselect("Focus Conditions", ["Fractures", "Soft Tissue Injuries", "Head Injuries", "Spinal Injuries", "Other"])
    highlight_specific = st.checkbox("Highlight Specific Treatments or Medical Terms")
    
    if st.button("Summarize Medical Records"):
        all_text = ""
        for file in medical_records:
            all_text += extract_text_from_file(file) + "\n\n"
        
        prompt = f"""
        Summarize the following medical records, focusing on these conditions: {', '.join(focus_conditions)}
        {"Also highlight specific treatments and medical terms." if highlight_specific else ""}
        
        Medical Records:
        {all_text}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert medical professional specializing in summarizing complex medical records."},
            {"role": "user", "content": prompt}
        ]
        
        response = assistant_response(messages)
        
        if response:
            st.markdown("### Medical Records Summary")
            st.write(response)
            st.markdown(get_download_link(response, "medical_summary.txt", "Download Medical Summary"), unsafe_allow_html=True)

# AI Chatbot Legal Assistant
def ai_chatbot_legal_assistant():
    st.header("AI Chatbot Legal Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What can I help you with?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert legal assistant specializing in personal injury cases."},
                    *st.session_state.messages
                ],
                stream=True,
            )
            for chunk in response:
                chunk_content = chunk.choices[0].delta.get("content", "")
                full_response += chunk_content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Main App
def main():
    st.set_page_config(page_title="Legal Assistant Toolkit", page_icon="⚖️", layout="wide")
    
    st.title("Legal Assistant Toolkit")
    
    tools = {
        "Demand Letter Generator": {"icon": "📝", "description": "Generate professional demand letters for personal injury cases."},
        "Counteroffer Drafting Tool": {"icon": "💼", "description": "Draft strategic counteroffers to initial settlement proposals."},
        "Email Template Generator": {"icon": "📧", "description": "Create context-aware emails for various legal communications."},
        "Police Statement Assistant": {"icon": "👮", "description": "Guide users in creating clear and accurate police statements."},
        "Medical Record Summarizer": {"icon": "🏥", "description": "Simplify complex medical documents into concise summaries."},
        "AI Chatbot Legal Assistant": {"icon": "🤖", "description": "Get real-time assistance with legal queries and tasks."}
    }
    
    st.sidebar.title("Tools")
    tool_selection = st.sidebar.radio("Select a Tool:", list(tools.keys()))
    
    if tool_selection == "Demand Letter Generator":
        demand_letter_generator()
    elif tool_selection == "Counteroffer Drafting Tool":
        counteroffer_drafting_tool()
    elif tool_selection == "Email Template Generator":
        email_template_generator()
    elif tool_selection == "Police Statement Assistant":
        police_statement_assistant()
    elif tool_selection == "Medical Record Summarizer":
        medical_record_summarizer()
    elif tool_selection == "AI Chatbot Legal Assistant":
        ai_chatbot_legal_assistant()

if __name__ == "__main__":
    main()
