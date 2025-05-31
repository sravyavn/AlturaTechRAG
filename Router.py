#tell llm to decide which of the sources is best to answer the question. 


from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load the model
llm = ChatGroq(temperature=0, model="llama3-70b-8192")

# Template to route query
template = PromptTemplate(
    input_variables=["question"],
    template="""
You are a smart routing agent. Your job is to choose the most relevant source document for answering the question.

Available sources:
1. HR_Handbook.pdf
2. Security_Protocol.pdf
3. Sales_Playbook.pdf
4. Engineering_SOP_Website

Return ONLY the filename or source string from above that best fits.

Question: {question}
Answer:
"""
)

def route_query(question):
    chain = template | llm
    response = chain.invoke({"question": question})
    
    # taking corresponding tbl name based on o/p
    routed_source = response.content.strip()
    source_mapping = {
        "HR_Handbook.pdf": "hr-handbook",
        "Security_Protocol.pdf": "security-protocol",
        "Sales_Playbook.pdf": "sales-playbook",
        "Engineering_SOP_Website": "engineering-sop"
    }
    return source_mapping.get(routed_source)
