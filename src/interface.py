import answering
import streamlit as st

st.write(
    """
# Krijg antwoorden van Rijksoverheid.nl
"""
)

query = st.text_input(
    "Wat wil je beantwoord hebben?", "Zit er BTW op zonnepanelen?"
)

if query:
    result = answering.get_answer_from_llm(
        query
    )

st.write("### Antwoord: \n", f"**{result['result']}**")

st.write("_Gebruikte bronnen_")

for source in set(list(map(lambda x: x["source"], result["source_documents"]))):
    st.code(source, language="markdown", line_numbers=False)