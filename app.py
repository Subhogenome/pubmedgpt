import os
import ssl
import time
import uuid
from Bio import Entrez
import streamlit as st
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context

# LangChain LLM setup
api = st.secrets["api"]
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api)

# Entrez setup
Entrez.email = "your_email@example.com"  # Add your email here

# Examples for PubMed query generation
examples = [
    {"input": "harmful effect of probiotics", "output": '"probiotics"[All Fields] AND "adverse effects"[All Fields] AND (side effect* OR complication*)'},
    {"input": "how is Lactobacillus related to human immunity", "output": '"Lactobacillus"[All Fields] AND ("immunity"[All Fields] OR TLR[All Fields] and IgA[All Fields] OR  cytokine[All Fields]) OR  "humans"[All Fields]'},
    {"input": "Human", "output": '"Human"[All Fields] OR "Homo Sapeins"[All Fields] OR "Humans"'}
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

prefix = (
    "You are an expert in converting English questions to Python PubMed query.\n"
    "You must create PubMed-compatible queries using MeSH terms, abstract, and title fields.\n"
    "Only return the query string as the answer, no explanation or extra text.\n"
    "Here are some examples:"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)

question_summary_prompt = PromptTemplate(
    input_variables=["question", "text"],
    template="""
You are a biomedical researcher who explains things on the basis of list of pubmed articles. Based on the question provided, read the following pubmed abstracts and generate a concise, relevant summary that directly answers or relates to the question.

QUESTION:
{question}

List of articles:
{text}

ANSWER-BASED SUMMARY:"""
)

question_summary_chain = LLMChain(
    llm=model,
    prompt=question_summary_prompt
)

def search_pubmed(term, retmax=10):
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']

def fetch_details(id_list):
    ids = ",".join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = handle.read()
    handle.close()
    return records

def batch_fetch_details(id_list, batch_size=10):
    all_records = []
    for start in range(0, len(id_list), batch_size):
        end = min(start + batch_size, len(id_list))
        batch_ids = id_list[start:end]
        records = fetch_details(batch_ids)
        all_records.append(records)
        time.sleep(1)
    return all_records


# Initialize chat session management
if "chat_ids" not in st.session_state:
    st.session_state.chat_ids = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "all_chats" not in st.session_state:
    # all_chats = {chat_id: [(role, message), ...], ...}
    st.session_state.all_chats = {}

if "last_question_tracker" not in st.session_state:
    # Stores the last question per chat for follow-up chaining
    st.session_state.last_question_tracker = {}

# Sidebar: chat session list and new chat button
with st.sidebar:
    st.title("🗂️ Chat Sessions")

    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chat_ids.append(new_id)
        st.session_state.current_chat_id = new_id
        st.session_state.all_chats[new_id] = []
        st.session_state.last_question_tracker[new_id] = ""

    for cid in st.session_state.chat_ids:
        messages = st.session_state.all_chats.get(cid, [])
        first_user_question = None
        for role, msg in messages:
            if role == "user":
                first_user_question = msg
                break
        if first_user_question:
            label = first_user_question.strip()
            if len(label) > 30:
                label = label[:27] + "..."
        else:
            label = "New Chat"
        if st.button(label, key=cid):
            st.session_state.current_chat_id = cid

# Main chat interface
if st.session_state.current_chat_id is None:
    st.info("Create or select a chat from the sidebar to start.")
else:
    chat_id = st.session_state.current_chat_id

    # Display chat history for current chat
    messages = st.session_state.all_chats.get(chat_id, [])
    for role, message in messages:
        with st.chat_message(role):
            st.write(message)

    # User input
    if question := st.chat_input("Ask your biomedical question..."):

        # Follow-up question chaining
        last_q = st.session_state.last_question_tracker.get(chat_id, "")
        if last_q:
            combined_question = f"{last_q} Follow-up: {question}"
        else:
            combined_question = question

        st.session_state.last_question_tracker[chat_id] = combined_question

        # Append user message & display
        st.session_state.all_chats[chat_id].append(("user", question))
        with st.chat_message("user"):
            st.write(question)

        # Generate PubMed query
        formatted_prompt = few_shot_prompt.format(input=combined_question)
        query_response = model.invoke(formatted_prompt)
        pubmed_query = query_response.content.strip()

        # Append and show PubMed query
        st.session_state.all_chats[chat_id].append(("assistant", f"🔍 PubMed Query:\n`{pubmed_query}`"))
        with st.chat_message("assistant"):
            st.write(f"🔍 PubMed Query:\n`{pubmed_query}`")

        # Fetch articles
        id_list = search_pubmed(pubmed_query, retmax=10)
        all_articles = batch_fetch_details(id_list)

        # Generate summary
        summary = question_summary_chain.run(question=combined_question, text=str(all_articles))

        # Append and show summary
        st.session_state.all_chats[chat_id].append(("assistant", f"🧠 Summary Based on Articles:\n{summary}"))
        with st.chat_message("assistant"):
            st.write(f"🧠 Summary Based on Articles:\n{summary}")
