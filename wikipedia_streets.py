!pip install langchain
!pip install wikipedia
!pip install langchain_openai


#This file takes in a street name and then uses langchain to query wikipedia to sumarize the street.

import langchain
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI  # Replace with your preferred LLM provider
import wikipedia
print("")
print("********************")
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever() 

OPENAI_API_KEY="Add Key Here"  # Replace with your OpenAI API key
llm = OpenAI(temperature=0.7, max_tokens=150,openai_api_key=OPENAI_API_KEY)  # Adjust temperature and max_tokens as needed

def summarize_street(street_name):
    """
    This agent takes a street name as input and returns a summary of the corresponding Wikipedia article.
    """
    # Use Wikipedia tool to get the article content
    wiki_article = retriever.invoke(street_name)

    # Check if article is found
    if not wiki_article:
        return f"Sorry, couldn't find a Wikipedia article for '{street_name} Street'."

    # Use LLM to summarize the article
    #summary_prompt = f"Please summarize the following Wikipedia article for '{street_name} Street':\n {wiki_article}"
    #summary_prompt="Tell me a joke"
    

    template = """Here is a street name: {question}, Tell me one important thing that happend on the street given this article:\n{wiki_article}"""
    prompt = PromptTemplate.from_template(template)
    print("The prompt template is")
    print(prompt)
    
    llm_chain=LLMChain(prompt=prompt,llm=llm)
    summary = llm_chain.run({"question":street_name,"wiki_article":wiki_article})
 
    return f"Here's a summary of what happend on: '{street_name} Street':\n {summary}"


