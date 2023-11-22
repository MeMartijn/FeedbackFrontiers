import answering
import streamlit as st

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.write(
    """
# Stel jouw vraag aan de overheid

Ik kan je helpen met het vinden van antwoorden over meer dan 350 onderwerpen en 11.207 stukken tekst waar de overheid bij betrokken is.
"""
)

query = st.text_input(
    "Wat is je vraag?",
    placeholder = "Zit er BTW op zonnepanelen?",
)
st.caption("Bijv. 'Wat doen waterschappen in godsnaam?' | 'Hoezo duurt het formeren van een nieuwe coalitie zo lang?'")

if query:
    result = answering.get_answer_from_llm(
        query
    )

    st.write("### ✅ Antwoord", )
    st.info(f"**{result['result']}**")

    st.balloons()

    st.divider()

    st.write("_Bronnen:_")

    for source in set(list(map(lambda x: x["source"], result["source_documents"]))):
        title = answering.get_webpage_title(source)
        st.write(f"[{title}]({source})", language="markdown", line_numbers=False)

    # st.write("_Mogelijke vervolgvragen_")

    # if query:
    #     new_questions = answering.get_new_questions_from_llm(
    #         query
    #     )

    # st.write("### Vervolgvragen: \n", f"**{new_questions['result']}**")

    # st.write("_Feedback_")

    # feedback = st.text_input(
    #     "Gaf dit antwoord op je vraag? Zo nee, welke informatie mist u?", ""
    # )

    # if feedback:
    #     reply = answering.post_feedback(
    #         query,
    #         feedback
    #     )
    #     print(reply)
    #     st.write(reply)