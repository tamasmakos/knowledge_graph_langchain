import streamlit as st
from llm import llm
from graph import graph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about the Bible, specifically focusing on the Gospels (Matthew, Mark, Luke, and John).
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Example Cypher Statements:

1. To find a specific verse:
```
MATCH (v:Verse {{book: "Book Name", chapter: ChapterNumber, name: VerseNumber}})
RETURN v.text
```

2. To find entities mentioned in a specific chapter:
```
MATCH (c:Chapter {{book: "Book Name", number: ChapterNumber}})<-[:IS_IN_CHAPTER]-(v:Verse)-[:APPEARS_IN]->(e:Entity)
RETURN DISTINCT e.name, e.type
```

3. To find verses containing a specific entity:
MATCH (e:Entity {{name: "Entity Name"}})<-[:APPEARS_IN]-(v:Verse)
RETURN v.book, v.chapter, v.name, v.text
Schema:

Schema:
(Book) -[:IS_IN_BOOK]-> (Chapter) -[:IS_IN_CHAPTER]-> (Verse) -[:APPEARS_IN]-> (Entity)

Node Properties:
Book: name
Chapter: number, book
Verse: name (verse number), chapter, book, text, embeddings, caption
Entity: name, type, caption

Question:
{question}
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)


cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True,
    validate_cypher=True,
)