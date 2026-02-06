import os
import sys
import time
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    load_files_node, 
    compute_diffs_node, 
    analyze_changes_node, 
    generate_json_node, 
    generate_html_node, 
    feedback_node
)

# Load environment variables
load_dotenv()

def route_feedback(state: AgentState):
    feedback = state.get("user_feedback")
    
    # Handle dict (new format) or str (legacy/simple format)
    if isinstance(feedback, dict):
        action = feedback.get("action", "approve")
    else:
        # feedback is string or None
        action = str(feedback).lower() if feedback else "approve"
        
    if action == "approve":
        print("--- Workflow Completed ---")
        return END
    else:
        print(f"--- Rerouting for Re-analysis (Action: {action}) ---")
        return "analyze"

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("load", load_files_node)
    workflow.add_node("diff", compute_diffs_node)
    workflow.add_node("analyze", analyze_changes_node)
    workflow.add_node("gen_json", generate_json_node)
    workflow.add_node("gen_html", generate_html_node)
    workflow.add_node("feedback", feedback_node)
    
    # Add Edges
    workflow.set_entry_point("load")
    workflow.add_edge("load", "diff")
    workflow.add_edge("diff", "analyze")
    workflow.add_edge("analyze", "gen_json")
    workflow.add_edge("gen_json", "gen_html")
    workflow.add_edge("gen_html", "feedback")
    
    # Conditional Edge
    workflow.add_conditional_edges(
        "feedback",
        route_feedback,
        {
            END: END,
            "analyze": "analyze"
        }
    )
    
    return workflow.compile()

def main():
    print("Starting Requirement Evolution Agent...")
    
    domain = input("Enter domain name OR Git File URL (default: 'General'): ").strip() or "General"
    
    # Check for OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        key = input("Enter your OpenAI API Key: ").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
        else:
            print("No API Key provided. Analysis might fail.")
    
    initial_state = {
        "domain": domain,
        "file_paths": [], # Will be loaded from directory
        "versions": [],
        "diffs": [],
        "json_output": {},
        "html_path": "",
        "user_feedback": None,
        "iteration": 0,
        "start_time": time.time()
    }
    
    app = build_graph()
    
    # Run the graph
    # Depending on recursion limit, might need config
    config = {"recursion_limit": 50}
    
    for event in app.stream(initial_state, config=config):
        # Stream events if needed, primarily relies on print statements in nodes
        pass

if __name__ == "__main__":
    main()
