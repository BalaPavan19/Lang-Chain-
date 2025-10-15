# ============================================================================
# 1. WHAT IS LANGCHAIN?
# ============================================================================
"""
LangChain is a framework for developing applications powered by language models.
It provides tools to chain together different components to build complex AI applications.
"""

''' 
Components :  Models, Prompt Management, Chains, LCEL(with Runnables), Output Parsing, Memory Management, Vector DB, Agents
'''

"""
LangChain helps you:
- Connect LLMs with external data sources
- Chain multiple LLM calls together
- Build conversational agents
- Create retrieval-augmented generation (RAG) systems
- Manage prompts and memory

Core Philosophy: Build composable, modular LLM applications
"""

from util import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

#LLM
llm = ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL_NAME, 
                    google_api_key=settings.GOOGLE_API_KEY
                    )

#Simple Prompt
prompt = "Suggest me a skill that is in demand?"
response = llm.invoke(prompt)
# print(" Suggested Skill:\n", response.content)


#Prompt Template
template = "Give me 3 career skills that are in high demand in {year} for {gender}."
prompt_template = PromptTemplate.from_template(template)


#Build a Chain with LCEL(Lang Chain Expression Language) 
chain = prompt_template | llm | StrOutputParser()
response = chain.invoke({"year": "2025", "gender":"Man"})
print("\n Career Skills in 2025:\n", response)
