from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph, END
import os

graph_builder = StateGraph(MessagesState)




# OCR imports
# from pdf2image import convert_from_path
# import pytesseract


# 1. Load PDF and extract pages
file_path = "./documents/AIML Road Map 2025.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)

# 3. Embed chunks using Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Create vector store from embedded docs
vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    # Use a clear system prompt, but do not prepend it to the messages (let the graph handle system prompts)
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([retrieve])


def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # If no tool messages, force retrieval
    if not tool_messages:
        last_human = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
        if last_human:
            retrieved, _ = retrieve(last_human.content)
            class ToolMsg:
                def __init__(self, content):
                    self.content = content
            tool_messages = [ToolMsg(retrieved)]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not getattr(message, 'tool_calls', None))
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

vector_store = InMemoryVectorStore(embeddings)



llm = ChatOllama(
    model = "granite3.2-vision",
    temperature = 0.8,
    num_predict = 256,
    # other params ...
) 



async def main():
    response = []
    async for chunk in llm.astream(messages):
        response.append(chunk.content)
    final_output = "".join(response)
    print(final_output)




graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)


graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)

graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)



graph = graph_builder.compile()
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    # Use the graph to process the user query, ensuring retrieval is used
    message = "What is the AIML Road Map 2025? What are my plans to start with"
    initial_state = MessagesState(messages=[
        HumanMessage(content=message)
    ])
    # Run the graph synchronously
    result = asyncio.run(graph.ainvoke(initial_state))
    # Extract and print the final response from the graph
    final_messages = result["messages"]
    for msg in final_messages:
        print(f"type: {getattr(msg, 'type', None)}, content: {getattr(msg, 'content', '')}")