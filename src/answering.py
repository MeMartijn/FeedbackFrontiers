import json
from bs4 import BeautifulSoup
from markdownify import markdownify
from dotenv import find_dotenv, load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv(find_dotenv())

# Ready all models
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

def html_to_markdown(html_string):
    # Parse the HTML string using BeautifulSoup
    soup = BeautifulSoup(html_string, 'html.parser')

    # Convert the parsed HTML to Markdown using markdownify
    markdown_text = markdownify(str(soup))

    return markdown_text

def html_to_markdown(html_string):
    # Parse the HTML string using BeautifulSoup
    soup = BeautifulSoup(html_string, 'html.parser')

    # Convert the parsed HTML to Markdown using markdownify
    markdown_text = markdownify(str(soup))

    return markdown_text

def create_content_snippet(question):
    # Get all content pieces in the answer
    if "content" in question["answer"].keys():
        content_pieces = list(
            map(
                lambda content_piece: [html_to_markdown(paragraph) for paragraph in content_piece.values()],
                question["answer"]["content"])
            )
        concat_content = "\n".join([item for row in content_pieces for item in row])
    else:
        concat_content = ""

    return f"""Vraag: {question["question"]}
Antwoord:
{html_to_markdown(question["answer"]["introduction"]) if "introduction" in question["answer"].keys() else ""}
{concat_content}

{"Onderwerpen: " + ", ".join(question["answer"]["subjects"]) if "subjects" in question["answer"].keys() else ""}
{"Thema's: " + ", ".join(question["answer"]["themes"]) if "themes" in question["answer"].keys() else ""}
{"Autoriteit: " + question["answer"]["authority"] if "authority" in question["answer"].keys() else ""}"""

# Get vectordb
vectordb_persist_dir = "./../data/faiss_index"
if not os.path.exists(vectordb_persist_dir):
    # Load file
    with open("./../data/qa-data.json", 'r') as file:
        data = json.load(file)

    # Create contents
    for qa in data:
        qa["content"] = create_content_snippet(qa)

    # Create documents
    factors = [
        Document(
            page_content=qa_item["content"],
            metadata={
                "identifier": qa_item["id"],
                "source": qa_item["canonical"]
            },
        )
        for qa_item in data
    ]

    # Create vector store
    vectordb = FAISS.from_documents(
        documents=factors,
        embedding=embedding,
    )
    vectordb.save_local(vectordb_persist_dir)
else:
    # ChromaDB has been initialised before, recreate instance
    vectordb = FAISS.load_local(vectordb_persist_dir, embedding)

def get_answer_from_llm(query):    
    # Build prompt
    template = """Gedraag je als een helpvolle assistent voor mensen die op zoek zijn naar allerlei antwoorden op vragen die iets te maken hebben met de Rijksoverheid. Beantwoord deze vraag ALLEEN op basis van de gegeven bronnen, niet op basis van eigen kennis. Als je de vraag niet kan beantwoorden, verontschuldig je en zeg dat de webmaster op de hoogte is gebracht van het niet hebben van de gevraagde informatie.
    Bronnen: ```{context}```
    Vraag: ```{question}```
    Behulpzaam antwoord: """
    qa_chain_prompt = PromptTemplate.from_template(template)

    # Define search kwargs
    search_kwargs = {"k": 5}

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_chain_prompt},
    )

    # Get result
    result = qa_chain({"query": query})

    return {
        **result,
        "source_documents": [
            {
                "page_content": doc.page_content,
                "identifier": doc.metadata["identifier"],
                "source": doc.metadata["source"],
            }
            for doc in result["source_documents"]
        ]
    }