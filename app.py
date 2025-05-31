import os
from Bio import Entrez
import streamlit as st
import ssl
import time
import re
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

# Question summarizer prompt
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

# PubMed fetch functions
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# User input
if question := st.chat_input("Ask your biomedical question..."):

    # Combine with last question if it's a follow-up
    if st.session_state.last_question:
        combined_question = f"{st.session_state.last_question} Follow-up: {question}"
    else:
        combined_question = question

    st.session_state.last_question = combined_question

    # Display and store user input
 #   st.chat_message("user").write(question)
#    st.session_state.messages.append(("user", question))
    st.session_state.messages.append(("user", question))

    # Generate PubMed query from the question
    formatted_prompt = few_shot_prompt.format(input=combined_question)
    query_response = model.invoke(formatted_prompt)
    pubmed_query = query_response.content.strip()
    
    # Display and store PubMed query
    st.chat_message("assistant").write(f"🔍 PubMed Query:\n`{pubmed_query}`")
    st.session_state.messages.append(("assistant", f"🔍 PubMed Query:\n`{pubmed_query}`"))

    # Fetch article records
    id_list = search_pubmed(pubmed_query, retmax=10)
    all_articles = batch_fetch_details(id_list)

    # Generate answer summary
    summary = question_summary_chain.run(question=combined_question, text=str(all_articles))

    # Display and store summary
    st.chat_message("assistant").write(f"🧠 Summary Based on Articles:\n{summary}")
    st.session_state.messages.append(("assistant", f"🧠 Summary Based on Articles:\n{summary}"))

# Display chat history
for role, message in st.session_state.messages[:-2]:  # Show all except last 2 (just added above)
    with st.chat_message(role):
        st.write(message)
