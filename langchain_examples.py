# ============================================================================
# 3. EXAMPLES
# ============================================================================


from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.runnables import RunnablePassthrough

# Eg. 1
def example_basic_chain():
    """Basic question-answering chain"""
    llm = ChatOpenAI(temperature=0)
    prompt = PromptTemplate.from_template("Answer this question: {question}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": "What is Python?"})

# Eg. 2
def example_multi_step():
    """Multi-step processing"""
    llm = ChatOpenAI(temperature=0.7)
    
    # Step 1: Generate story
    story_prompt = PromptTemplate.from_template(
        "Write a short story about {topic} in 50 words."
    )
    
    # Step 2: Extract moral
    moral_prompt = PromptTemplate.from_template(
        "Extract the moral from this story:\n{story}"
    )
    
    # Combine steps
    chain = (
        {"topic": RunnablePassthrough()}
        | story_prompt
        | llm
        | StrOutputParser()
        | (lambda story: {"story": story})
        | moral_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke("friendship")


# Eg. 3
def example_chatbot():
    """Conversational chatbot with memory"""
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory(return_messages=True)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Simulate conversation
    response1 = conversation.predict(input="Hi, I'm learning Python")
    response2 = conversation.predict(input="What did I just tell you?")
    
    return response1, response2
