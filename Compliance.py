#give generated output to llm to check for any sensitive information and add footnotes

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load Groq LLM
llm = ChatGroq(temperature=0, model="llama3-70b-8192")

# Compliance Prompt
compliance_prompt = PromptTemplate(
    input_variables=["answer"],
    template="""
You are a compliance and policy safety checker.

Review the following answer for any sensitive, risky, or confidential content.
If anything risky is found, add a footnote warning at the end explaining the concern.

Answer:
{answer}

Return the reviewed output below (with footnote if needed):
"""
)

def check_compliance(answer):
    chain = compliance_prompt | llm
    response = chain.invoke({"answer": answer})
    return response.content.strip()
