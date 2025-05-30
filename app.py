import os
from Bio import Entrez
from Bio import SeqIO
import streamlit as st
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
from langchain.prompts import PromptTemplate
from groq import Groq
import time
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
api = st.secrets["api"]
model =ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",api_key=api)

examples = [
    {"input": "harmful effect of probiotics", "output": '"probiotics"[All Fields] AND "adverse effects"[All Fields] AND (side effect* OR complication*)'},
    {"input": "how is Lactobacillus related to human immunity", "output": '"Lactobacillus"[All Fields] AND ("immunity"[All Fields] OR TLR[All Fields] and IgA[All Fields] OR  cytokine[All Fields]) OR  "humans"[All Fields]'},
    {"input": "Human", "output": '"Human"[All Fields] OR "Homo Sapeins"[All Fields] OR "Humans"'}
]




# How to format each example
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

# Corrected: Single string as prefix
prefix = (
    "You are an expert in converting English questions to Python PubMed query.\n"
    "You must create PubMed-compatible queries using MeSH terms, abstract, and title fields.\n"
    "Only return the query string as the answer, no explanation or extra text.\n"
    "Here are some examples:"
)


# Create the few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)
import re
def extract_titles_and_articles(records):
    # Use regular expression to find all titles and their corresponding articles
    pattern = re.compile(r'(TI  - .*?)(?=\nTI  -|\Z)', re.DOTALL)
    matches = pattern.findall(records)
    titles_and_articles = [(re.search(r'TI  - (.*?)\n', match).group(1), match) for match in matches]
    return titles_and_articles



def search_pubmed(term, retmax=500):
    # Search PubMed with the specified term and return up to retmax results
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
        time.sleep(1)  # Respect rate limits by pausing between requests
    return all_records




question_summary_prompt = PromptTemplate(
    input_variables=["question", "text"],
    template="""
You are a biomedical researcher who explains things on the basis of list of puibmed articles . Based on the question provided, read the following pubmed abstracts and generate a concise, relevant summary that directly answers or relates to the question.

QUESTION:
{question}

List of atricles:
{text}

ANSWER-BASED SUMMARY:"""
)




question_summary_chain = LLMChain(
    llm=model,
    prompt=question_summary_prompt
)





# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input from the user
if question := st.chat_input("Ask your biomedical question..."):
    
    # Display user message
    st.chat_message("user").write(question)
    st.session_state.messages.append(("user", question))

    # Get model's intermediate understanding (optional few-shot)
    formatted_prompt = few_shot_prompt.format(input=question)
    response = model.invoke(formatted_prompt)
    output_text = response.content.strip()

    # Fetch PubMed abstracts
    id_list = search_pubmed(output_text, retmax=10)
    all_articles = batch_fetch_details(id_list)

    # Generate the answer summary
    summary = question_summary_chain.run(question=question, text=str(all_articles))

    # Display assistant response
    st.chat_message("assistant").write(summary)
    st.session_state.messages.append(("assistant", summary))

# Show previous conversation history
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.write(message)
