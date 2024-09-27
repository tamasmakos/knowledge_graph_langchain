import streamlit as st
from llm import llm, embeddings
from graph import graph

# Create the Neo4jVector
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

try:
    # Test the embedding function
    test_embedding = embeddings.embed_query("test")
    if not test_embedding:
        raise ValueError("Embedding function returned an empty result")

    neo4jvector = Neo4jVector.from_existing_index(
        embeddings,                              # (1)
        graph=graph,                             # (2)
        index_name="bibleVerses",                # (3)
        node_label="Verse",                      # (4)
        text_node_property="text",               # (5)
        embedding_node_property="embedding",     # (6)
    )
except Exception as e:
    st.error(f"Error initializing Neo4jVector: {str(e)}")
    neo4jvector = None

# Create the retriever
retriever = neo4jvector.as_retriever()

# Create the prompt
from langchain_core.prompts import ChatPromptTemplate

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create the chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

def get_verse_text(input):
    try:
        response = plot_retriever.invoke({"input": input})
        # Assuming the response contains an 'answer' key with the relevant text
        if 'answer' in response:
            return response['answer']
        else:
            return "I'm sorry, I couldn't find the specific verse text."
    except Exception as e:
        return f"An error occurred while retrieving the verse: {str(e)}"