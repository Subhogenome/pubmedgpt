import os
import uuid
import ssl
import time
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
Entrez.email = "your_email@example.com"  # Replace with a valid email

# ---------- Prompt Templates ----------
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
    "You are an expert in converting English questions to PubMed queries.\n"
    "Use MeSH terms, abstract and title fields. Only return the query string, no explanations.\n"
    "Examples:"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)

# Summary generation prompt
question_summary_prompt = PromptTemplate(
    input_variables=["question", "text"],
    template="""
You are a biomedical researcher who explains things based on PubMed abstracts. 
Given the question and article list, generate a concise and relevant summary.

QUESTION:
{question}

List of articles:
{text}

ANSWER-BASED SUMMARY:"""
)
question_summary_chain = LLMChain(llm=model, prompt=question_summary_prompt)

# ---------- PubMed Functions ----------
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

# ---------- Multi-Chat Session Management ----------
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}  # {chat_id: [(role, message)]}
if "chat_ids" not in st.session_state:
    st.session_state.chat_ids = []
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.chat_ids.append(new_id)
    st.session_state.current_chat_id = new_id
    st.session_state.all_chats[new_id] = []

if "last_question_tracker" not in st.session_state:
    st.session_state.last_question_tracker = {}  # Per chat

# Sidebar UI
with st.sidebar:
    st.title("🗂️ Chat Sessions")
    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chat_ids.append(new_id)
        st.session_state.current_chat_id = new_id
        st.session_state.all_chats[new_id] = []
        st.session_state.last_question_tracker[new_id] = ""

    for cid in st.session_state.chat_ids:
        label = f"Chat {cid[:6]}"
        if st.button(label):
            st.session_state.current_chat_id = cid

# Get current chat and its state
chat_id = st.session_state.current_chat_id
messages = st.session_state.all_chats[chat_id]
last_question = st.session_state.last_question_tracker.get(chat_id, "")

# Display previous messages
for role, message in messages:
    with st.chat_message(role):
        st.write(message)

# Chat input
if question := st.chat_input("Ask your biomedical question..."):
    # Combine follow-up
    if last_question:
        combined_question = f"{last_question} Follow-up: {question}"
    else:
        combined_question = question
    st.session_state.last_question_tracker[chat_id] = combined_question

    # Show user message
    messages.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    # Generate query
    formatted_prompt = few_shot_prompt.format(input=combined_question)
    query_response = model.invoke(formatted_prompt)
    pubmed_query = query_response.content.strip()

    # Show query
    messages.append(("assistant", f"🔍 PubMed Query:\n`{pubmed_query}`"))
    with st.chat_message("assistant"):
        st.write(f"🔍 PubMed Query:\n`{pubmed_query}`")

    # Fetch articles
    id_list = search_pubmed(pubmed_query, retmax=10)
    all_articles = batch_fetch_details(id_list)

    # Generate and show summary
    summary = question_summary_chain.run(question=combined_question, text=str(all_articles))
    messages.append(("assistant", f"🧠 Summary Based on Articles:\n{summary}"))
    with st.chat_message("assistant"):
        st.write(f"🧠 Summary Based on Articles:\n{summary}")
