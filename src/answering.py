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
import requests

# Load environment variables
load_dotenv(find_dotenv())

# Ready all models
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define config
vectordb_persist_dir = "./../data/faiss_index_v2"

def html_to_markdown(html_string):
    # Parse the HTML string using BeautifulSoup
    soup = BeautifulSoup(html_string, 'html.parser')

    # Convert the parsed HTML to Markdown using markdownify
    markdown_text = markdownify(str(soup))

    return markdown_text

def filter_data_for_duplicates(data):
    # Set to keep track of unique IDs
    unique_ids = set()

    # List to store the filtered data
    filtered_data = []

    # Iterate through the data
    for item in data:
        # Check if the ID is already in the set
        if item['id'] not in unique_ids:
            # Add the ID to the set
            unique_ids.add(item['id'])
            # Add the item to the filtered data list
            filtered_data.append(item)
    
    return filtered_data

def extract_pieces_from_question(question):
    # Set empty list for all content pieces
    pieces = []

    # Set suffix to add to each content piece
    suffix = "\n" + (
        ('Onderwerpen: ' + '; '.join(question['answer']['subjects']) + "\n" if 'subjects' in question['answer'].keys() else '') +
        ('Thema\'s: ' + '; '.join(question['answer']['themes']) if 'themes' in question['answer'].keys() else '')
    ).strip()

    # Part 1: content pieces based on question
    if "question" in question["answer"].keys():
        if "introduction" in question["answer"].keys():
            # Content piece 1: just the Q and A
            pieces.append(f"Vraag: {question['answer']['question']}\nAntwoord:\n{html_to_markdown(question['answer']['introduction'])}".strip() + suffix)
        elif "content" not in question["answer"].keys():
            # Content piece 2: there is no content, just return the question for reference materials for the end user
            pieces.append(f"Vraag: {question['answer']['question']}".strip() + suffix)
        elif "content" in question["answer"].keys() and len(question["answer"]["content"]) == 1:
            # Content piece 3: question with the only content piece
            content = '\n'.join([html_to_markdown(paragraph_piece) for paragraph_piece in question['answer']['content'][0].values()])
            pieces.append(f"Vraag: {question['answer']['question']}\nAntwoord:\n{content}".strip() + suffix)
    
    # Part 2: content pieces based on content
    if "content" in question["answer"].keys() and len(question["answer"]["content"]) > 1:
        for content_piece in question["answer"]["content"]:
            content = '\n'.join([html_to_markdown(paragraph_piece) for paragraph_piece in content_piece.values()])
            pieces.append(content.strip() + suffix)

    return pieces

def generate_documents(data):
    content = []

    for qa_item in data:
        # Get all individual content pieces
        content_pieces = extract_pieces_from_question(qa_item)

        # Create documents for each piece
        document_pieces = [
            Document(
                page_content=piece,
                metadata={
                    "source": qa_item["canonical"]
                },
            )
            for piece in content_pieces
        ]

        # Merge documents into full content list
        content.extend(document_pieces)
    
    return content

with open('./../data/qa-data.json', 'r') as file:
    data = filter_data_for_duplicates(json.load(file))

if not os.path.exists(vectordb_persist_dir):
    # Generate all documents
    docs = generate_documents(data)

    # Create vector store
    vectordb = FAISS.from_documents(
        documents=docs,
        embedding=embedding,
    )
    vectordb.save_local(vectordb_persist_dir)
else:
    # ChromaDB has been initialised before, recreate instance
    vectordb = FAISS.load_local(vectordb_persist_dir, embedding)


def post_feedback(query, feedback): 
    search = vectordb.similarity_search(query + " " + feedback)
    return search
    # # Get result
    # result = qa_chain({"query": query})
    # return {
    #     True
    # }

def get_answer_from_llm(query):    
    # Build prompt
    template = """Gedraag je als een helpvolle assistent voor mensen die op zoek zijn naar allerlei antwoorden op vragen die iets te maken hebben met de Rijksoverheid. Beantwoord deze vraag ALLEEN op basis van de gegeven bronnen, niet op basis van eigen kennis. Als je de vraag niet kan beantwoorden, verontschuldig je en zeg dat de website van de Rijksoverheid geen informatie heeft over de gestelde vraag.
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
                "source": doc.metadata["source"],
            }
            for doc in result["source_documents"]
        ]
    }

def get_new_questions_from_llm(query):    
    # Build prompt
    template = """Gedraag je als een helpvolle assistent voor mensen die op zoek zijn naar allerlei antwoorden op vragen die iets te maken hebben met de Rijksoverheid. Geef geen antwoord op de vraag, maar Geef 3 nieuwe vragen die gerelateerd zijn aan deze vraag ALLEEN op basis van de gegeven bronnen, niet op basis van eigen kennis. Als je de vraag niet kan beantwoorden, verontschuldig je en zeg dat de webmaster op de hoogte is gebracht van het niet hebben van de gevraagde informatie. Houd je antwoord beknopt.
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
                "source": doc.metadata["source"],
            }
            for doc in result["source_documents"]
        ]
    }

def get_webpage_title(url):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the web page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the title tag and extract its text
            title_tag = soup.find('title')
            if title_tag is not None:
                return title_tag.text
            else:
                return url
        else:
            return url
    except Exception as e:
        return "url"