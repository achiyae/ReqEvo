import os
import time
import json
import pickle
from dotenv import load_dotenv
from typing import Dict, Any
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

def load_existing_state(name: str) -> Dict[str, Any]:
    """Loads state from an existing pickle file."""
    states_dir = os.path.join(os.getcwd(), "states")
    filename = f"{name}.pkl"
    filepath = os.path.join(states_dir, filename)
    
    if not os.path.exists(filepath):
        # Try if name already includes .pkl
        if name.endswith('.pkl'):
             filepath = os.path.join(states_dir, name)
        else:
             # Try search in dir
             files = [f for f in os.listdir(states_dir) if f.startswith(name)]
             if files:
                 filepath = os.path.join(states_dir, files[0])
             else:
                 raise FileNotFoundError(f"Could not find state for {name} in {states_dir}")

    with open(filepath, 'rb') as f:
        state = pickle.load(f)
        
    return state

def main():
    print("Starting Requirement Evolution Agent...")
    
    while True:
        print("\nSelect Mode:")
        print("1. [N]ew Analysis")
        print("2. [R]eview Existing")
        choice = input("Choice (1/2 or N/R) [N]: ").strip().lower()
        if choice in ('1', 'n', ''):
            choice = 'n'
            break
        elif choice in ('2', 'r'):
            choice = 'r'
            break
        else:
            print(f"Invalid choice logged: '{choice}'. Please select 1/N or 2/R.")

    if choice == 'r':
        while True:
            name = input("Enter the file name root (e.g. 'pep-0773'): ").strip()
            if not name:
                print("Invalid input logged: File name cannot be empty.")
                continue
            try:
                state = load_existing_state(name)
                print(f"Resuming analysis for: {state['domain']}")
                break
            except Exception as e:
                print(f"Error resuming state for '{name}': {e}")
                print("Please try again with a valid file name.")
        
        try:
            while True:
                # ... same while loop logic ...
                # Direct loop for review/edit
                state.update(generate_html_node(state))
                feedback_result = feedback_node(state)
                state.update(feedback_result)
                
                feedback = state.get('user_feedback')
                if isinstance(feedback, str) and feedback == 'approve':
                    break
                
                action = feedback.get('action') if isinstance(feedback, dict) else str(feedback)
                
                if action == 'finish':
                    print("\n--- Finalizing Analysis ---")
                    state['is_final'] = True
                    state.update(generate_html_node(state))
                    break
                elif action == 'retry' or action == 'approve':
                    if action == 'retry':
                        print("\n--- Rerunning Analysis based on feedback ---")
                        state.update(analyze_changes_node(state))
                        state.update(generate_json_node(state))
                    else:
                        break
                else:
                    break
                    
            print("Finished.")
            
        except Exception as e:
            print(f"Error resuming state: {e}")
            return
    else:
        while True:
            domain = input("Enter domain name OR Git File URL (default: 'General'): ").strip() or "General"
            if domain:
                break
            else:
                 print("Invalid input logged: Input cannot be empty. Please provide a domain or Git URL.")
        
        # Check for OPENAI_API_KEY
        while not os.environ.get("OPENAI_API_KEY"):
            key = input("Enter your OpenAI API Key: ").strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
                break
            else:
                print("No API Key provided. Analysis requires an OpenAI API Key.")

        initial_state = {
            "domain": domain,
            "versions": [],
            "diffs": [],
            "start_time": time.time(),
            "user_feedback": None,
            "is_final": False
        }

        # We can't use the compiled LangGraph directly if we need the custom loop with 'finish'
        # Let's adjust the build_graph or just use a manual loop for consistency
        print(f"Starting NEW analysis for: {domain}")
        
        state = initial_state
        # Nodes: load -> diff -> analyze -> gen_json
        state.update(load_files_node(state))
        state.update(compute_diffs_node(state))
        state.update(analyze_changes_node(state))
        state.update(generate_json_node(state))
        
        while True:
            state.update(generate_html_node(state))
            feedback_result = feedback_node(state)
            state.update(feedback_result)
            
            feedback = state.get('user_feedback')
            action = feedback.get('action') if isinstance(feedback, dict) else str(feedback)
            
            if action == 'finish':
                print("\n--- Finalizing Analysis ---")
                state['is_final'] = True
                state.update(generate_html_node(state))
                break
            elif action == 'retry':
                print("\n--- Rerunning Analysis based on feedback ---")
                state.update(analyze_changes_node(state))
                state.update(generate_json_node(state))
            else:
                break
            
        print("--- Workflow Completed ---")

if __name__ == "__main__":
    main()
