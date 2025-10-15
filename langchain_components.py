# ----------------------------------------------------------------------------
# 2.1 LLM WRAPPERS
# ----------------------------------------------------------------------------
"""
Wrappers provide a unified interface to interact with different LLM providers.
They standardize how you call different models (OpenAI, Anthropic, HuggingFace, etc.)
"""

# Example: Different LLM Wrappers
from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import HuggingFaceHub

# OpenAI Chat Model (GPT-3.5, GPT-4)
chat_openai = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500
)

# OpenAI Completion Model (older models)
llm_openai = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7
)

# Anthropic Claude
chat_claude = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.7
)

# HuggingFace Models
llm_hf = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.7}
)

"""
Why Wrappers?
- Unified API: Same code works with different providers
- Easy switching: Change provider without rewriting code
- Built-in features: Rate limiting, retries, error handling
"""

# ----------------------------------------------------------------------------
# 2.2 PROMPT TEMPLATES
# ----------------------------------------------------------------------------
"""
Templates help you create reusable, parameterized prompts.
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Simple Prompt Template
simple_prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template="Write a marketing copy for {product} targeting {audience}."
)

# Using the template
formatted = simple_prompt.format(product="AI Chatbot", audience="small businesses")
print("Simple Prompt:", formatted)

# Chat Prompt Template (for chat models)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role} assistant."),
    ("human", "Help me with: {task}")
])

# Format chat messages
messages = chat_prompt.format_messages(
    role="coding",
    task="debug my Python code"
)

# Few-Shot Prompts (with examples)
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"}
]

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template="Word: {word}\nAntonym: {antonym}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of each word:",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"]
)

print("\nFew-Shot Prompt:", few_shot_prompt.format(input="big"))

# ----------------------------------------------------------------------------
# 2.3 CHAINS
# ----------------------------------------------------------------------------
"""
Chains link together multiple components to create complex workflows.
"""

from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.7)

# A. Simple LLM Chain
print("\n=== SIMPLE LLM CHAIN ===")
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a one-sentence summary about {topic}."
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="summary")

# B. Sequential Chain (multiple steps)
print("\n=== SEQUENTIAL CHAIN ===")
prompt2 = PromptTemplate(
    input_variables=["summary"],
    template="Translate this to Spanish: {summary}"
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="translation")

overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["topic"],
    output_variables=["summary", "translation"],
    verbose=True
)

# result = overall_chain({"topic": "Artificial Intelligence"})

# C. Transform Chain (custom processing)
def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    # Custom transformation
    return {"output": text.upper()}

transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["output"],
    transform=transform_func
)

# ----------------------------------------------------------------------------
# 2.4 MODERN APPROACH: LCEL (LangChain Expression Language)
# ----------------------------------------------------------------------------
"""
LCEL is the modern, recommended way to build chains.
More flexible and easier to understand than legacy chains.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda,RunnableBranch

print("\n=== LCEL EXAMPLES ===")

# Simple chain with LCEL
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm | StrOutputParser()

# Usage: result = chain.invoke({"topic": "programmers"})

# Multi-step LCEL chain
summarize_prompt = PromptTemplate.from_template("Summarize: {text}")
translate_prompt = PromptTemplate.from_template("Translate to French: {summary}")

multi_chain = (
    {"text": RunnablePassthrough()}        # Step 1: Accept input as {"text": ...}
    | summarize_prompt                     # Step 2: Fill in "Summarize: {text}" prompt
    | llm                                  # Step 3: Generate summary using LLM
    | StrOutputParser()                    # Step 4: Extract summary string
    | (lambda summary: {"summary": summary})  # Step 5: Wrap summary into dict
    | translate_prompt                     # Step 6: Fill in "Translate to French: {summary}" prompt
    | llm                                  # Step 7: Translate using LLM
    | StrOutputParser()                    # Step 8: Extract final translation
)


# Runnables ( RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch,  )
'''
RunnablePassthrough is a simple runnable that takes an input and returns it as-is, without any modification.
'''
passthrough = RunnablePassthrough()
output = passthrough.invoke({"message": "Hello from LangChain!"})
print(output)

'''
RunnableParallel is a runnable that allows for the parallel execution of multiple sub-runnables
'''
def add_one(x):
    return x + 1

def multiply_by_two(x):
    return x * 2

parallel_chain = RunnableParallel(
    original=RunnablePassthrough(),
    plus_one=RunnableLambda(add_one),
    times_two=RunnableLambda(multiply_by_two)
)

output = parallel_chain.invoke(5)
print(output)


'''
RunnableBranch This Runnable allows for conditional execution based on a given condition
'''
model = ChatOpenAI()
spanish_prompt = ChatPromptTemplate.from_template("Translate '{text}' to Spanish.")
english_prompt = ChatPromptTemplate.from_template("Translate '{text}' to English.")

translator = RunnableBranch(
    (lambda x: x["language"] == "spanish", spanish_prompt | model),
    (lambda x: x["language"] == "english", english_prompt | model),
)
# result = translator.invoke({"text": "hello", "language": "spanish"})

# ----------------------------------------------------------------------------
# 2.5 OUTPUT PARSERS
# ----------------------------------------------------------------------------
"""
Parse and structure LLM outputs into usable formats.
"""

from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    CommaSeparatedListOutputParser,
    PydanticOutputParser
)
from pydantic import BaseModel, Field

# List Parser
list_parser = CommaSeparatedListOutputParser()
format_instructions = list_parser.get_format_instructions()

list_prompt = PromptTemplate(
    template="List 5 {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

# Structured Parser
response_schemas = [
    ResponseSchema(name="name", description="Name of the person"),
    ResponseSchema(name="age", description="Age of the person"),
    ResponseSchema(name="occupation", description="Occupation")
]

structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Pydantic Parser (type-safe)
class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")

pydantic_parser = PydanticOutputParser(pydantic_object=Person)

# ----------------------------------------------------------------------------
# 2.6 MEMORY
# ----------------------------------------------------------------------------
"""
Memory allows chains to remember previous interactions.
Essential for chatbots and conversational AI.
"""

from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory
)

# Buffer Memory (stores all messages)
buffer_memory = ConversationBufferMemory()
buffer_memory.save_context(
    {"input": "Hi, I'm John"},
    {"output": "Hello John! How can I help?"}
)
print("\nBuffer Memory:", buffer_memory.load_memory_variables({}))

# Window Memory (keeps last K messages)
window_memory = ConversationBufferWindowMemory(k=2)

# Summary Memory (summarizes old conversations)
summary_memory = ConversationSummaryMemory(llm=llm)

# Token Buffer (limits by token count)
token_memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=100)

# Using memory in chains
from langchain.chains import ConversationChain

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True
)

# ----------------------------------------------------------------------------
# 2.7 AGENTS
# ----------------------------------------------------------------------------
"""
Agents can use tools and make decisions about which actions to take.
"""

from langchain.agents import AgentType, initialize_agent, Tool
from langchain.tools import tool

# Define custom tools
@tool
def get_weather(location: str) -> str:
    """Get weather for a location"""
    return f"Weather in {location}: Sunny, 25°C"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

tools = [
    Tool(
        name="Weather",
        func=get_weather,
        description="Get current weather for a location"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Calculate mathematical expressions"
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Usage: agent.run("What's the weather in Mumbai and what's 25 * 4?")

# ----------------------------------------------------------------------------
# 2.8 DOCUMENT LOADERS & RETRIEVERS (RAG)
# ----------------------------------------------------------------------------
"""
Load documents and retrieve relevant information.
Core of Retrieval-Augmented Generation (RAG).
"""

from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load documents
# loader = TextLoader("data.txt")
# documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
# splits = text_splitter.split_documents(documents)

# Create embeddings and vector store
# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(splits, embeddings)

# Create retriever
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG Chain
from langchain.chains import RetrievalQA

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff"
# )
