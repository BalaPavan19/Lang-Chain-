# ============================================================================
# 3. EXAMPLES
# ============================================================================

from util import llm
# from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.runnables import RunnablePassthrough

# Eg. 1
def example_basic_chain():
    """Basic question-answering chain"""
    prompt = PromptTemplate.from_template("Answer this question: {question}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": "What is Python?"})
 


# Eg. 2
def example_multi_step():
    """Multi-step processing"""
    
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
    memory = ConversationBufferMemory(return_messages=True)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    response1 = conversation.predict(input="Hi, I'm learning Python")
    response2 = conversation.predict(input="What did you just tell me?")
    
    return response1, response2



"""
LANGCHAIN PRACTICAL EXAMPLES
======================================
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List



# ============================================================================
# EXAMPLE 1: Understanding LLM Wrappers
# ============================================================================

"""
What are LLM Wrappers?
- They wrap API calls to different LLM providers
- Provide consistent interface across providers
- Handle connection, authentication, rate limiting

Think of it like a universal remote for different LLMs!
"""

def example_llm_wrapper():
    # Direct invocation (simplest form)
    response = llm.invoke("What is 2+2?")
    print(f"Direct LLM call: {response.content}\n")
    
    # Batch invocation
    responses = llm.batch([
        "What is Python?",
        "What is JavaScript?"
    ])
    print("Batch results:")
    for i, resp in enumerate(responses, 1):
        print(f"{i}. {resp.content[:50]}...")
    print()
    
    # Stream invocation
    print("Streaming response:")
    for chunk in llm.stream("what is stream response in langchain and how it works"):
        print(chunk.content, end="", flush=False)
    print("\n")


# example_llm_wrapper()

# ============================================================================
# EXAMPLE 2: Prompt Templates - Building Blocks
# ============================================================================

"""
Why Prompt Templates?
- Reusability: Write once, use many times
- Maintainability: Change prompt in one place
- Variable substitution: Dynamic prompts
"""

def example_prompt_templates():
    # Basic template
    template = PromptTemplate(
        input_variables=["language", "task"],
        template="Write a {language} function to {task}"
    )
    
    # Reuse same template for different inputs
    prompts = [
        template.format(language="Python", task="sort a list"),
        template.format(language="JavaScript", task="reverse a string"),
        template.format(language="Java", task="find maximum in array")
    ]
    
    print("Generated Prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")
        print(llm.invoke(prompt).content)
    print()
    
    # Chat template (for conversational AI)
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} expert."),
        ("human", "{question}")
    ])
    
    messages = chat_template.format_messages(
        role="Python programming",
        question="How do I handle exceptions?"
    )
    
    print("Chat Messages:")
    for msg in messages:
        print(f"  {msg.type}: {msg.content}")
    print()


# example_prompt_templates()

# ============================================================================
# EXAMPLE 3: Simple Chain - Connecting LLM and Prompt
# ============================================================================


"""
Chains connect components together.
Most basic: Prompt Template → LLM → Output
"""

def example_simple_chain():
    # Old way (LLMChain - being deprecated)
    print("Method 1: LLMChain (Legacy)")
    prompt = PromptTemplate.from_template("Explain {concept} in simple terms.")
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"concept": "blockchain"})
    print(f"Result: {result['text'][:100]}...\n")
    
    # New way (LCEL - Recommended)
    print("Method 2: LCEL (Modern)")
    prompt = PromptTemplate.from_template("Explain {concept} in simple terms.")
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"concept": "machine learning"})
    print(f"Result: {result[:100]}...\n")


# example_simple_chain()

# ============================================================================
# EXAMPLE 4: Sequential Chain - Multi-Step Processing
# ============================================================================


"""
Sequential chains pass output from one step as input to next.
Use case: Generate → Translate → Summarize
"""

def example_sequential_chain():
    print("Building a Story Generator → Translator → Summarizer\n")
    
    # Step 1: Generate story
    story_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a 100-word story about {topic}."
    )
    story_chain = LLMChain(
        llm=llm,
        prompt=story_prompt,
        output_key="story"
    )
    
    # Step 2: Translate to Spanish
    translate_prompt = PromptTemplate(
        input_variables=["story"],
        template="Translate this story to Spanish:\n{story}"
    )
    translate_chain = LLMChain(
        llm=llm,
        prompt=translate_prompt,
        output_key="translation"
    )
    
    # Step 3: Summarize in one line
    summary_prompt = PromptTemplate(
        input_variables=["translation"],
        template="Summarize this in one sentence:\n{translation}"
    )
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        output_key="summary"
    )
    
    # Combine all chains
    overall_chain = SequentialChain(
        chains=[story_chain, translate_chain, summary_chain],
        input_variables=["topic"],
        output_variables=["story", "translation", "summary"],
        verbose=True
    )
    
    result = overall_chain({"topic": "a robot learning to paint"})
    
    print("\n--- Results ---")
    print(f"Original Story: {result['story'][:80]}...")
    print(f"Translation: {result['translation'][:80]}...")
    print(f"Summary: {result['summary']}")
    print()


# example_sequential_chain()

# ============================================================================
# EXAMPLE 5: LCEL Multi-Step (Modern Approach)
# ============================================================================


"""
LCEL (LangChain Expression Language) is more flexible and readable.
Uses pipe operator (|) to chain components.
"""

def example_lcel_chain():
    # Define prompts
    joke_prompt = PromptTemplate.from_template(
        "Tell a joke about {topic}"
    )
    
    explain_prompt = PromptTemplate.from_template(
        "Explain why this joke is funny:\n{joke}"
    )
    
    # Build chain using pipes
    chain = (
        {"topic": RunnablePassthrough()}
        | joke_prompt
        | llm
        | StrOutputParser()
        | (lambda joke: {"joke": joke})
        | explain_prompt
        | llm
        | StrOutputParser()
    )
    
    result = chain.invoke("programmers")
    print(f"Explanation: {result}\n")


# example_lcel_chain()

# ============================================================================
# EXAMPLE 6: Output Parsers - Structuring Responses
# ============================================================================


"""
Output parsers convert LLM text into structured formats.
Use cases: Extract JSON, lists, custom objects
"""

def example_output_parsers():
    # Parse as JSON
    class Recipe(BaseModel):
        name: str = Field(description="Recipe name")
        ingredients: List[str] = Field(description="List of ingredients")
        time_minutes: int = Field(description="Cooking time in minutes")
    
    parser = JsonOutputParser(pydantic_object=Recipe)
    
    prompt = PromptTemplate(
        template="Generate a recipe for {dish}.\n{format_instructions}",
        input_variables=["dish"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({"dish": "biryani"})
    
    print("Parsed Recipe (as Python dict):")
    print(f"Name: {result['name']}")
    print(f"Ingredients: {', '.join(result['ingredients'])}")
    print(f"Time: {result['time_minutes']} minutes\n")


# example_output_parsers()

# ============================================================================
# EXAMPLE 7: Memory - Building Chatbots
# ============================================================================


"""
Memory allows chains to remember previous messages.
Essential for multi-turn conversations.
"""

def example_memory():
    from langchain.chains import ConversationChain
    
    # Create conversation with memory
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Simulate conversation
    # print("Chatbot Conversation:\n")
    
    # resp1 = conversation.invoke(input="Hi! My name is Bala and I love cricket.")
    # print(f"Bot: {resp1}\n")
    
    # resp2 = conversation.invoke(input="What's my name?")
    # print(f"Bot: {resp2}\n")
    
    # resp3 = conversation.invoke(input="What sport do I like?")
    # print(f"Bot: {resp3}\n")
    
    # # Show memory contents
    # print("--- Memory Contents ---")
    # print(memory.load_memory_variables({}))
    # print()
    
    while True:
        you = input('You : ')
        print(conversation.invoke(input=you))

# example_memory()

# ============================================================================
# EXAMPLE 8: Complete Application - Recipe Generator
# ============================================================================


def recipe_generator_app():
    """
    Complete app: Takes ingredients, generates recipe, provides nutrition info
    """
    
    class RecipeOutput(BaseModel):
        recipe_name: str
        instructions: List[str]
        cooking_time: int
        difficulty: str
    
    # Step 1: Generate recipe
    recipe_prompt = PromptTemplate.from_template(
        "Create a recipe using these ingredients: {ingredients}. "
        "Format as JSON with recipe_name, instructions (list), "
        "cooking_time (minutes), and difficulty (easy/medium/hard)."
    )
    
    # Step 2: Add nutrition info
    nutrition_prompt = PromptTemplate.from_template(
        "For this recipe: {recipe_name}, estimate calories and key nutrients. "
        "Keep it brief (2-3 sentences)."
    )
    
    # Build pipeline
    recipe_chain = recipe_prompt | llm | JsonOutputParser(pydantic_object=RecipeOutput)
    
    nutrition_chain = (
        nutrition_prompt
        | llm
        | StrOutputParser()
    )
    
    # Execute
    ingredients = "chicken, tomatoes, garlic, olive oil"
    
    recipe = recipe_chain.invoke({"ingredients": ingredients})
    nutrition = nutrition_chain.invoke({"recipe_name": recipe['recipe_name']})
    
    print(f"Recipe: {recipe['recipe_name']}")
    print(f"Difficulty: {recipe['difficulty']}")
    print(f"Time: {recipe['cooking_time']} minutes")
    print(f"\nInstructions:")
    for i, step in enumerate(recipe['instructions'], 1):
        print(f"{i}. {step}")
    print(f"\nNutrition Info: {nutrition}")
    print()


# recipe_generator_app()


