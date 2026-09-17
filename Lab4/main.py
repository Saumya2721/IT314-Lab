from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
import dotenv

class AgileState(TypedDict):
    # --- Lab 3 Context ---
    system_description: str
    stakeholders: str
    goals: str
    fr_nfr: str
    
    # --- Lab 4 Artifacts ---
    user_stories: str
    sprint_plan: str
    selected_sprint: str
    prototype_code: str
    test_cases: str
    test_report: str
    
    # --- Control Variables ---
    status: str
    current_feedback: str
    iteration: int
    max_iter: int

dotenv.load_dotenv()  
api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)
parser = StrOutputParser()

## Define functions for each step in the process

def process_user_stories(state: AgileState) -> dict:
    if state["iteration"] >= state["max_iter"]:
        return {"status": "approved", "iteration": 0, "current_feedback": ""}

    # 1. Generate (Stage 6)
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Agile Product Owner."),
        ("human", """Based on the FR/NFRs: {fr_nfr}
        Stakeholders: {stakeholders}
        goals: {goals}
        system description: {system_description}
        Generate standard user stories (As a <role>, I want <goal>, so that <benefit>). Include the front and back of the card.
        Feedback to address: {current_feedback}""")
    ])
    stories = (gen_prompt | model | parser).invoke(state)

    # 2. Evaluate (Stage 7 - INVEST Criteria Check)[cite: 2]
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict Agile Coach."),
        ("human", """Evaluate the following user stories: {user_stories}
        Evaluate each generated user story strictly against the INVEST criteria (Independent, Negotiable, Valuable, Estimable, Small, Testable)[cite: 2].
        If they pass all criteria, output ONLY 'approve'. 
        If any story fails one or more criteria, output ONLY 'reject:' followed by the specific reason they failed[cite: 2].""") 
    ])

    evaluation = (eval_prompt | model | parser).invoke({**state, "user_stories": stories}).strip().lower()

    if evaluation.startswith("approve"):
        return {"user_stories": stories, "status": "approved", "iteration": 0, "current_feedback": ""}
    else:
        return {"user_stories": stories, "status": "rejected", "iteration": state["iteration"] + 1, "current_feedback": evaluation}


def process_sprint_grouping(state: AgileState) -> dict:
    if state["iteration"] >= state["max_iter"]:
        return {"status": "approved", "iteration": 0, "current_feedback": ""}

    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Scrum Master."),
        ("human", "Group these user stories into 2-3 logical Sprints based on priority: {user_stories}."
        "Feedback: {current_feedback}")
    ])
    sprints = (gen_prompt | model | parser).invoke(state)

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict Agile Coach."),
        ("human", "Review this Sprint plan: {sprints} against the user stories: {user_stories}. Output ONLY 'approve' if logical, or 'reject:' with reasons.")
    ])
    evaluation = (eval_prompt | model | parser).invoke({"sprints": sprints, "user_stories": state["user_stories"]}).strip().lower()

    if evaluation.startswith("approve"):
        return {"sprint_plan": sprints, "status": "approved", "iteration": 0, "current_feedback": ""}
    else:
        return {"sprint_plan": sprints, "status": "rejected", "iteration": state["iteration"] + 1, "current_feedback": evaluation}

# --- HUMAN IN THE LOOP (HITL) ---
def human_sprint_selection(state: AgileState) -> dict:
    print("\n" + "="*50)
    print(" OPINION REQUIRED")
    print("="*50)
    print(state["sprint_plan"])
    # Pause terminal execution to get user input
    selected = input("\n Enter the Sprint number you want to prototype (e.g., 'Sprint 1'): ")
    return {"selected_sprint": selected, "status": "approved"}

def process_prototype(state: AgileState) -> dict:
    if state["iteration"] >= state["max_iter"]:
        return {"status": "approved", "iteration": 0, "current_feedback": ""}

    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Frontend Developer."),
        ("human", """Generate a single, working HTML file (with embedded CSS/JS) that prototypes the user stories in: {selected_sprint}.
        Reference the overall goals: {goals} and the functional requirements (fr) and non-functional requirements (nfr) {fr_nfr}. Output ONLY valid HTML code. Feedback: {current_feedback}""")
    ])
    code = (gen_prompt | model | parser).invoke(state)

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict QA Lead."),
        ("human", "Review this HTML prototype code: {code}. Does it include HTML, CSS, and JS logic? Does it have any errors? Reference it against the user stories in: {selected_sprint} and the overall requirements: {fr_nfr}. Output ONLY 'approve' or 'reject:' with reasons.")
    ])
    evaluation = (eval_prompt | model | parser).invoke({"code": code, "selected_sprint": state["selected_sprint"], "fr_nfr": state["fr_nfr"]}).strip().lower()

    if evaluation.startswith("approve"):
        return {"prototype_code": code, "status": "approved", "iteration": 0, "current_feedback": ""}
    else:
        return {"prototype_code": code, "status": "rejected", "iteration": state["iteration"] + 1, "current_feedback": evaluation}

def process_test_cases(state: AgileState) -> dict:
    if state["iteration"] >= state["max_iter"]:
        return {"status": "approved", "iteration": 0, "current_feedback": ""}

    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a QA Tester."),
        ("human", """Based on the user stories in: {selected_sprint}, the functional requirements (fr) and non-functional requirements (nfr): {fr_nfr},generate structured test cases (ID, pre-conditions, steps, expected result)[cite: 2]. Feedback: {current_feedback}""")
    ])
    tests = (gen_prompt | model | parser).invoke(state)

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict QA Lead."),
        ("human", "Are these test cases complete and testable? {tests}. Output ONLY 'approve' or 'reject:'.")
    ])
    evaluation = (eval_prompt | model | parser).invoke({"tests": tests}).strip().lower()

    if evaluation.startswith("approve"):
        return {"test_cases": tests, "status": "approved", "iteration": 0, "current_feedback": ""}
    else:
        return {"test_cases": tests, "status": "rejected", "iteration": state["iteration"] + 1, "current_feedback": evaluation}

def process_automated_testing(state: AgileState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Automated Testing Simulator."),
        ("human", """Simulate the execution of these test cases: {test_cases} against this prototype code: {prototype_code}.
        Output a structured test report indicating Pass/Fail for each case and why[cite: 2].""")
    ])
    report = (prompt | model | parser).invoke(state)
    return {"test_report": report, "status": "approved"}

# Define the state graph
def route_feedback(state: AgileState) -> str:
    if state["status"] == "approved":
        return "next"
    return "retry"

workflow = StateGraph(AgileState)

workflow.add_node("User_Stories", process_user_stories)
workflow.add_node("Sprint_Grouping", process_sprint_grouping)
workflow.add_node("Sprint_Selection_HITL", human_sprint_selection)
workflow.add_node("Prototype", process_prototype)
workflow.add_node("Test_Cases", process_test_cases)
workflow.add_node("Automated_Testing", process_automated_testing)

workflow.add_edge(START, "User_Stories")

# Add Conditional Edges
workflow.add_conditional_edges("User_Stories", route_feedback, {"retry": "User_Stories", "next": "Sprint_Grouping"})
workflow.add_conditional_edges("Sprint_Grouping", route_feedback, {"retry": "Sprint_Grouping", "next": "Sprint_Selection_HITL"})

# Standard edges for guaranteed handoffs
workflow.add_edge("Sprint_Selection_HITL", "Prototype")

workflow.add_conditional_edges("Prototype", route_feedback, {"retry": "Prototype", "next": "Test_Cases"})
workflow.add_conditional_edges("Test_Cases", route_feedback, {"retry": "Test_Cases", "next": "Automated_Testing"})
workflow.add_edge("Automated_Testing", END)

app = workflow.compile()

if __name__ == "__main__":

    with open("Lab3_Output.txt", "r", encoding="utf-8") as f:
        content = f.read() # Read once into memory
    
        # Extract Stakeholders
        lab3_stakeholders = content.split("=== STAKEHOLDERS ===")[1].split("=== GOALS ===")[0].strip()
    
        # Extract Goals (stopping at Techniques to avoid capturing extra sections)
        lab3_goals = content.split("=== GOALS ===")[1].split("=== TECHNIQUES ===")[0].strip()
    
        # Extract FRs/NFRs
        lab3_fr_nfr = content.split("=== FR_NFR ===")[1].strip()

    with open("system_desc.txt", "r", encoding="utf-8") as f:
        system_description = f.read().strip()

    initial_state = {
        "system_description": system_description,
        "stakeholders": lab3_stakeholders,
        "goals": lab3_goals,
        "fr_nfr": lab3_fr_nfr,
        "status": "",
        "current_feedback": "",
        "iteration": 0,
        "max_iter": 1 # Will force approve after 1 retry
    }

    print("Starting Lab 4 Pipeline...")
    final_state = app.invoke(initial_state)

    # Save outputs
    with open("Lab4_Output.txt", "w", encoding="utf-8") as f:
        for key in ["user_stories", "sprint_plan", "selected_sprint", "test_cases", "test_report"]:
            f.write(f"=== {key.upper()} ===\n{final_state.get(key, 'N/A')}\n\n")
            
    with open("prototype.html", "w", encoding="utf-8") as f:
        f.write(final_state.get("prototype_code", ""))
        
    print("Successfully created Lab4_Output.txt and prototype.html!")