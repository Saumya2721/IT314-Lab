from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import operator
import os
import dotenv

class RequirementState(TypedDict):
    system_description: str
    stakeholders: str
    goals: str
    techniques: str
    execution: str
    fr_nfr: str
    current_feedback: str
    status: str

dotenv.load_dotenv()  
api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=api_key)


# Functions for each step in the requirements elicitation process
def identify_stakeholders(state: RequirementState) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Requirements Analyst."),
        ("human", "Identify stakeholders for the following system: {system_description}. Consider any previous feedback: {current_feedback}")
    ])
    chain = prompt | model
    result = chain.invoke({"system_description": state["system_description"], "current_feedback": state.get("current_feedback", "None")})
    return {"stakeholders": result.content, "current_feedback": ""} # Clear feedback after applying

def identify_stakeholder_goals(state: RequirementState) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Requirements Analyst."),
        ("human", "Identify goals for the following stakeholders: {stakeholders} in the context of the system: {system_description}. Consider any previous feedback: {current_feedback}")
    ])
    chain = prompt | model
    result = chain.invoke({"stakeholders": state["stakeholders"], "system_description": state["system_description"], "current_feedback": state.get("current_feedback", "None")})
    return {"goals": result.content, "current_feedback": ""}

def select_elicitation_techniques(state: RequirementState) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Requirements Analyst."),
        ("human", """ 
        Analyze the Stakeholders and their Goals to select the most appropriate elicitation technique(s).
        
        Example 1:
        Stakeholders: Busy C-Level Executives.
        Goals: Define high-level business objectives and budget constraints.
        Output: 1. Executive Interviews: Best for brief, high-level strategic alignment respecting their limited time.
        
        Example 2:
        Stakeholders: 500+ Remote Insurance Agents.
        Goals: Gather daily workflow pain points and feature requests.
        Output: 1. Online Questionnaire: Scalable method to gather quantitative data from a large, distributed group. 
        2. Focus Group (Workshop): Follow-up with a small subset to dive deeper into common complaints.
        
        Now, process the following:
        Stakeholders: {stakeholders}
        Goals: {goals}
        Previous Feedback to address (if any): {current_feedback}""")
    ])
    chain = prompt | model
    
    result = chain.invoke({
        "stakeholders": state["stakeholders"], 
        "goals": state["goals"], 
        "current_feedback": state.get("current_feedback", "None")
    })
    
    return {"techniques": result.content, "current_feedback": ""}

def elicitation_execution(state: RequirementState) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior Requirements Analyst carrying out elicitation techniques."),
        ("human", """
        Example:
        Techniques: 1. Online Questionnaire for Agents. 2. Interview with Underwriters.
        Output: 
        **Execution Plan:**
        *   **Questionnaire (Target: Insurance Agents):** 
            1. On a scale of 1-5, how difficult is the current policy consolidation process?
            2. What are the top 3 missing features you need to compete with other companies?
        *   **Interview Draft (Target: Lead Underwriter):**
            1. "Can you walk me through the risk assessment parameters you currently use?"
            2. "What constraints must the new system enforce to ensure compliance?"
        
        Now, process the following:
        Techniques: {techniques}
        Stakeholders: {stakeholders}
        Goals: {goals}
        Previous Feedback to address (if any): {current_feedback}""")
    ])

    chain = prompt | model
    result = chain.invoke({
        "techniques": state["techniques"], 
        "stakeholders": state["stakeholders"], 
        "goals": state["goals"],
        "current_feedback": state.get("current_feedback", "None")
    })

    return {"execution": result.content, "current_feedback": ""}

def fr_nfr_identification(state: RequirementState) -> list[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Requirements Analyst."),
        ("human", """Based on the elicitation execution: {execution}
        The elicitation techniques used: {techniques}
        The stakeholders: {stakeholders}
        And their goals: {goals}
        Identify functional and non-functional requirements for the system: {system_description}.
        Previous Feedback to address (if any): {current_feedback}""")
    ])
    
    chain = prompt | model
    result = chain.invoke({
        "execution": state["execution"], 
        "techniques": state["techniques"],
        "stakeholders": state["stakeholders"],
        "goals": state["goals"],
        "system_description": state["system_description"],
        "current_feedback": state.get("current_feedback", "None")
    })
    
    return {"fr_nfr": result.content, "current_feedback": ""}

# Feedback loop to refine the requirements elicitation process

def review_stakeholders(state: RequirementState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict Requirements Engineering Manager."),
        ("human", """Review the stakeholders: {stakeholders} against the system: {system_description}. 
        Are all key groups represented?
        If yes, output ONLY 'approve'. 
        If no, output ONLY 'reject:' followed by specific, actionable feedback on who is missing and why.""")
    ])
    # 1. Add StrOutputParser() to the chain pipeline
    chain = prompt | model | StrOutputParser()
    
    # 2. Remove .content, just invoke and call .strip() directly on the returned string
    result = chain.invoke({
        "stakeholders": state["stakeholders"], 
        "system_description": state["system_description"]
    }).strip()
    
    if result.lower().startswith("approve"):
        return {"status": "approved", "current_feedback": ""}
    else:
        return {"status": "rejected", "current_feedback": result}

def review_goals(state: RequirementState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict Requirements Engineering Manager."),
        ("human", """Review the stakeholder goals: {goals} against the stakeholders: {stakeholders} and system: {system_description}. 
        Are all key objectives represented?
        If yes, output ONLY 'approve'. 
        If no, output ONLY 'reject:' followed by specific, actionable feedback on which goals are missing or unclear.""")
    ])
    # 1. Add StrOutputParser() to the chain pipeline
    chain = prompt | model | StrOutputParser()
    
    # 2. Remove .content, just invoke and call .strip() directly on the returned string
    result = chain.invoke({
        "goals": state["goals"], 
        "stakeholders": state["stakeholders"], 
        "system_description": state["system_description"]
    }).strip()
    
    if result.lower().startswith("approve"):
        return {"status": "approved", "current_feedback": ""}
    else:
        return {"status": "rejected", "current_feedback": result}

def review_techniques(state: RequirementState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict Requirements Engineering Manager."),
        ("human", """Review the elicitation techniques: {techniques} against the stakeholders: {stakeholders}, their goals: {goals}, and the system: {system_description}. 
        Are the techniques appropriate for the context?
        If yes, output ONLY 'approve'. 
        If no, output ONLY 'reject:' followed by specific, actionable feedback on which techniques are unsuitable or missing.""")
    ])
    # 1. Add StrOutputParser() to the chain pipeline
    chain = prompt | model | StrOutputParser()
    
    # 2. Remove .content, just invoke and call .strip() directly on the returned string
    result = chain.invoke({
        "techniques": state["techniques"], 
        "stakeholders": state["stakeholders"], 
        "goals": state["goals"], 
        "system_description": state["system_description"]
    }).strip()
    
    if result.lower().startswith("approve"):
        return {"status": "approved", "current_feedback": ""}
    else:
        return {"status": "rejected", "current_feedback": result}

def review_execution(state: RequirementState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict Requirements Engineering Manager."),
        ("human", """Review the elicitation execution: {execution} against the elicitation techniques: {techniques}, stakeholders: {stakeholders}, their goals: {goals}, and the system: {system_description}. 
        Was the execution thorough and effective?
        If yes, output ONLY 'approve'. 
        If no, output ONLY 'reject:' followed by specific, actionable feedback on what was missed or poorly executed.""")
    ])

    chain = prompt | model | StrOutputParser()
    result = chain.invoke({
        "execution": state["execution"], 
        "techniques": state["techniques"], 
        "stakeholders": state["stakeholders"], 
        "goals": state["goals"], 
        "system_description": state["system_description"]
    }).strip()

    if result.lower().startswith("approve"):
        return {"status": "approved", "current_feedback": ""}   
    else:
        return {"status": "rejected", "current_feedback": result}

def review_fr_nfr(state: RequirementState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict Requirements Engineering Manager."),
        ("human", """Review the identified functional and non-functional requirements: {fr_nfr} against the elicitation execution: {execution}, techniques: {techniques}, stakeholders: {stakeholders}, and goals: {goals}.
        Are the requirements complete and well-defined?
        If yes, output ONLY 'approve'.
        If no, output ONLY 'reject:' followed by specific, actionable feedback on what is missing or unclear.""")
    ])
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({
        "fr_nfr": state["fr_nfr"],
        "execution": state["execution"],
        "techniques": state["techniques"],
        "stakeholders": state["stakeholders"],
        "goals": state["goals"],
        "system_description": state["system_description"]
    }).strip()

    if result.lower().startswith("approve"):
        return {"status": "approved", "current_feedback": ""}
    else:
        return {"status": "rejected", "current_feedback": result}

def route_feedback(state: RequirementState) -> str:
    if state["status"] == "approved":
        return "next"
    return "retry"


# Build the StateGraph for the requirements elicitation workflow

workflow = StateGraph(RequirementState)

# Add Generation Nodes
workflow.add_node("identify_stakeholders", identify_stakeholders)
workflow.add_node("identify_stakeholder_goals", identify_stakeholder_goals)
workflow.add_node("select_elicitation_techniques", select_elicitation_techniques)
workflow.add_node("elicitation_execution", elicitation_execution)
workflow.add_node("fr_nfr_identification", fr_nfr_identification)

# Add Review Nodes
workflow.add_node("review_stakeholders", review_stakeholders)
workflow.add_node("review_goals", review_goals)
workflow.add_node("review_techniques", review_techniques)
workflow.add_node("review_execution", review_execution)
workflow.add_node("review_fr_nfr", review_fr_nfr)

# Set Entry Point
workflow.add_edge(START, "identify_stakeholders")

# Wire Generation to Review (Standard Edges)
workflow.add_edge("identify_stakeholders", "review_stakeholders")
workflow.add_edge("identify_stakeholder_goals", "review_goals")
workflow.add_edge("select_elicitation_techniques", "review_techniques")
workflow.add_edge("elicitation_execution", "review_execution")
workflow.add_edge("fr_nfr_identification", "review_fr_nfr")

# Wire Review to Routing (Conditional Edges)
workflow.add_conditional_edges("review_stakeholders", route_feedback, {
    "retry": "identify_stakeholders",       # Loop back
    "next": "identify_stakeholder_goals"    # Move forward
})

workflow.add_conditional_edges("review_goals", route_feedback, {
    "retry": "identify_stakeholder_goals", 
    "next": "select_elicitation_techniques"
})

workflow.add_conditional_edges("review_techniques", route_feedback, {
    "retry": "select_elicitation_techniques", 
    "next": "elicitation_execution"
})

workflow.add_conditional_edges("review_execution", route_feedback, {
    "retry": "elicitation_execution", 
    "next": "fr_nfr_identification"
})

workflow.add_conditional_edges("review_fr_nfr", route_feedback, {
    "retry": "fr_nfr_identification", 
    "next": END
})

pipeline = workflow.compile()

# Execution

if __name__ == "__main__":
    import json
    
    # The case study text provided in the lab instructions
    system_description = """LIC Market Driven System
LIC, an insurance company, wants to digitize a range of business processes and provide a
complete solution that addresses all aspects of the agent-insurer relationship. Consider
yourself as a part of the Requirement Analyst team at Retinodes Software Company, and
your job is to gather, analyze, negotiate, validate, and prioritize the set of requirements.
In this new requirement of the project, there are no existing systems that can be analyzed
for the development. Requirements have to be gathered, negotiated, validated, and
prioritized through multiple stakeholders. This is a complex process because different
stakeholders have different perspectives, requirements, expectations, and priorities.
Therefore, Retinodes wants to have a requirements engineering framework that can be used
in market-facing projects where requirements need to be gathered from multiple
stakeholders.
To start with, you need to identify the set of stakeholders associated with the system,
understand the domain information about the insurance market, and identify possible
system features. Stakeholders may include customers, insurance agents, LIC management,

policy administrators, underwriters, and other relevant personnel. The insurance domain
should also be studied to understand existing policies, customer demands, pricing, business
rules, and restrictions.
The first product LIC wants you to develop is a system for creating consolidated insurance
packages that can compete with packages provided by other insurance companies. Existing
insurance packages should be analyzed to understand their features, benefits, pricing, and
conditions. The requirements of customers and insurance agents should also be considered
while designing these packages.
Another product is based on customer priority. Based on the insurance policies available, the
customer can create his/her own package by selecting suitable policies and send a request
for review. The system has to automatically analyze the proposed package, identify possible
issues or restrictions, provide suggestions (if any), and finally give a competing price for the
package.
To understand the problem domain, existing insurance packages have to be analyzed and the
demands and restrictions from insurance policies and agents have to be understood
completely. Different stakeholders may have conflicting requirements, so the requirements
need to be negotiated, validated, and prioritized based on business importance and
feasibility.
The requirements and feasibility report generated by you will be further used by the
development team for implementation. The report should provide a clear understanding of
the stakeholders, domain information, system features, functional and non-functional
requirements, constraints, priorities, and feasibility of the proposed solution."""

    initial_state = {
        "system_description": system_description, 
        "current_feedback": "",
        "status": ""
    }

    print("Starting Requirements Elicitation Pipeline...")
    
    # invoke() runs the entire graph to completion and returns the final state
    final_state = pipeline.invoke(initial_state)

    print("Pipeline finished! Saving deliverables...")

    # Save to TXT[cite: 1]
    with open("Output.txt", "w", encoding="utf-8") as txt_file:
        for key in ["stakeholders", "goals", "techniques", "execution", "fr_nfr"]:
            txt_file.write(f"=== {key.upper()} ===\n{final_state.get(key, 'N/A')}\n\n")

    print("Successfully created Output.txt!")