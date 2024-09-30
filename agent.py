from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain import hub
from utils import get_session_id
from tools.cypher import cypher_qa
from tools.vector import get_verse_text

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a bible expert providing information about the Gospels (Matthew, Mark, Luke, and John). Always use the Bible Verse Search tool first for any verse-related queries. Only use other tools if the Bible Verse Search doesn't provide enough information."),
        ("human", "{input}"),
    ]
)

# bible_chat = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="Bible Verse Search",  
        description="Use this tool to find specific verses or similar verses within the Gospels (Matthew, Mark, Luke, and John). This should be your first choice for any verse-related queries.",
        func=get_verse_text
    ),
    Tool.from_function(
        name="Bible information",
        description="Provide information about bible verses using Cypher queries. Only use this if Bible Verse Search doesn't give enough information.",
        func = cypher_qa
    ),
    # Tool.from_function(
    #     name="General Chat",
    #     description="For general bible discussion not covered by other tools. This should be your last resort.",
    #     func=bible_chat.invoke,
    # )
]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a bible expert providing information about the Gospels (Matthew, Mark, Luke, and John).
Always use the Bible Verse Search tool first for any verse-related queries.
If the Bible Verse Search tool doesn't find an exact match, analyze the results and provide the most relevant information.
Only use other tools if the Bible Verse Search doesn't provide any useful information.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to the Gospels.
Do not answer any questions using your pre-trained knowledge, only use the information provided by the tools.
TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:



```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```


When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:



```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']