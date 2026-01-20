import os
import difflib
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from agent.state import AgentState, RequirementVersion, DiffEntry
from agent.utils import render_html_report, open_in_browser

# --- Analysis Models ---
class ChangeAnalysis(BaseModel):
    reason_type: str = Field(description="One of: Contradiction, Mistake, Inclusion, Summarization/shortening, Deletion, Clarification, Demonstration, Meaning, Other")
    reason_text: str = Field(description="Explanation for the change")

# --- Nodes ---

def load_files_node(state: AgentState) -> Dict[str, Any]:
    """Loads requirement files from the specified paths."""
    print("--- Loading Files ---")
    files = state.get('file_paths', [])
    versions: List[RequirementVersion] = []
    
    # If no files in state, try to find them in 'requirements' dir
    if not files:
        req_dir = os.path.join(os.getcwd(), 'requirements')
        if os.path.exists(req_dir):
            files = [os.path.join(req_dir, f) for f in sorted(os.listdir(req_dir)) if f.endswith('.txt')]
    
    for idx, file_path in enumerate(files):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            versions.append({
                "version_id": idx + 1,
                "content": content,
                "filename": os.path.basename(file_path)
            })
            
    return {"versions": versions, "file_paths": files}

def compute_diffs_node(state: AgentState) -> Dict[str, Any]:
    """Computes granular diffs between sequential versions."""
    print("--- Computing Diffs ---")
    versions = state['versions']
    diffs: List[DiffEntry] = []
    
    # Sort by version_id
    sorted_versions = sorted(versions, key=lambda v: v['version_id'])
    
    global_diff_id = 1
    
    for i in range(len(sorted_versions) - 1):
        old_v = sorted_versions[i]
        new_v = sorted_versions[i+1]
        
        # Split into non-empty lines (assuming each line is a requirement)
        old_lines = [line for line in old_v['content'].splitlines() if line.strip()]
        new_lines = [line for line in new_v['content'].splitlines() if line.strip()]
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            
            # Extract chunks
            old_chunk = old_lines[i1:i2]
            new_chunk = new_lines[j1:j2]
            
            # Heuristic: Break down 'replace' blocks into 1-to-1 pairs if meaningful
            # Or just 'insert'/'delete' line by line
            
            # Calculate how many sub-diffs to generate
            # For strict granular analysis "per requirement", let's behave as follows:
            # - If 1 line replaces 1 line -> 1 diff
            # - If N lines replace N lines -> N distinct diffs
            # - If N lines replace M lines (and N!=M) -> Harder. 
            #   Let's just align them 1-to-1 as much as possible, and the rest as insert/delete?
            #   Or keep it as a block?
            #   Simplicity for now: For 'replace', split min(N,M) pairs, then add inserts/deletes? 
            #   Actually, difflib usually handles N!=M by having separate insert/delete/replace blocks 
            #   or a replace block. 
            #   Let's stick to a simple loop over max(len) to force granularity.
            
            max_len = max(len(old_chunk), len(new_chunk))
            
            for k in range(max_len):
                sub_old = old_chunk[k] if k < len(old_chunk) else None
                sub_new = new_chunk[k] if k < len(new_chunk) else None
                
                # Construct display diff for this specific item
                diff_lines = []
                if sub_old:
                    diff_lines.append(f"- {sub_old}")
                if sub_new:
                    diff_lines.append(f"+ {sub_new}")
                
                diff_text = "\n".join(diff_lines)
                
                diffs.append({
                    "diff_id": global_diff_id,
                    "old_version_id": old_v['version_id'],
                    "new_version_id": new_v['version_id'],
                    "diff_text": diff_text,
                    "reason_type": "Pending Analysis",
                    "reason_text": "Pending...",
                    "old_content_snippet": sub_old if sub_old else "",
                    "new_content_snippet": sub_new if sub_new else ""
                })
                global_diff_id += 1
        
    return {"diffs": diffs}

def analyze_changes_node(state: AgentState) -> Dict[str, Any]:
    """Uses LLM to analyze the reasons for changes."""
    print("--- Analyzing Changes ---")
    
    # Check for API Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found. Skipping analysis.")
        return {}

    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    parser = JsonOutputParser(pydantic_object=ChangeAnalysis)
    
    prompt_messages = [
        ("system", "You are an expert business analyst specializing in requirement evolution."),
        ("user", """Analyze the changes between these two requirement document versions.
        
        Old Version:
        {old_text}
        
        New Version:
        {new_text}
        
        Diff:
        {diff_text}
        
        Identify the PRIMARY reason for the changes. If there are multiple distinct changes, 
        summarize the dominant one or the most critical one.
        
        Possible reason types:
        - Contradiction: The change was introduced to fix a contradiction between two requirements.
        - Mistake: The change fixes a mistake in one requirements or more.
        - Inclusion: The requirement was change to be more inclusive.
        - Summarization/shortening: The change is made to summarize and shorten a lengthy requirement.
        - Deletion: The requirement was redundant and therefore removed.
        - Clarification of a requirement: The change was made to help understanding the requirement.
        - Demonstration (example, visualization): An example was required to assist in understanding the requirement.
        - Meaning: The requirement required a change to its meaning/intention.
        - Other. 
        
        {feedback_section}
        
        {format_instructions}
        """)
    ]
    
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    chain = prompt | llm | parser
    
    versions = {v['version_id']: v['content'] for v in state['versions']}
    updated_diffs = []
    
    # Check for feedback
    feedback = state.get('user_feedback')
    feedback_section = ""
    if feedback and feedback != "approve":
        feedback_section = f"IMPORTANT: The user rejected a previous analysis with the following feedback/correction:\n'{feedback}'\nPlease adjust your analysis to respect this feedback."
    
    for diff in state['diffs']:
        # Skip if already analyzed (unless re-running)
        # But here we just re-analyze or analyze pending
        
        old_text = versions.get(diff['old_version_id'], "")
        new_text = versions.get(diff['new_version_id'], "")
        
        try:
            result = chain.invoke({
                "old_text": old_text,
                "new_text": new_text,
                "diff_text": diff['diff_text'],
                "feedback_section": feedback_section,
                "format_instructions": parser.get_format_instructions()
            })
            
            diff['reason_type'] = result['reason_type']
            diff['reason_text'] = result['reason_text']
            
        except Exception as e:
            print(f"Error analyzing diff {diff['diff_id']}: {e}")
            diff['reason_type'] = "Error"
            diff['reason_text'] = f"Analysis failed: {str(e)}"
            
        updated_diffs.append(diff)
        
    # Clear feedback after using it so we don't get stuck in a loop if we proceed
    # Actually, in the graph flow, if we loop back, we might want to keep it ONE time.
    # But usually, if we re-analyze, the next human review will reset it or provide new feedback.
    # We will clear it in the 'feedback' node reset or just assume state update handles it. 
    # In LangGraph, we return the DIFFS update. We probably shouldn't clear 'user_feedback' here 
    # explicitly unless we return it as None. 
    # Let's return it as None to "consume" the feedback.
    return {"diffs": updated_diffs, "user_feedback": None}

def generate_json_node(state: AgentState) -> Dict[str, Any]:
    print("--- Generating JSON ---")
    
    # Construct the final JSON structure requested
    # { "domain": "name", "number of versions": 0, "diffs": [ ... ] }
    
    versions_map = {v['version_id']: v for v in state['versions']}
    
    json_diffs = []
    for d in state['diffs']:
        old_v = versions_map.get(d['old_version_id'])
        new_v = versions_map.get(d['new_version_id'])
        
        json_diffs.append({
            "diff_id": d['diff_id'],
            "reason type": d['reason_type'],
            "reason text": d['reason_text'],
            "old_version": {
                "version id": d['old_version_id'],
                "requirement id": 0, # Placeholder as per request usually 0 if unstructured
                "content": old_v['content'] if old_v else ""
            },
            "new_version": {
                "version id": d['new_version_id'],
                "requirement id": 0,
                "content": new_v['content'] if new_v else ""
            },
            "diff": d['diff_text']
        })
        
    json_output = {
        "domain": state.get("domain", "Unknown Domain"),
        "number of versions": len(state['versions']),
        "diffs": json_diffs
    }
    
    # Save to file
    with open("output.json", "w") as f:
        json.dump(json_output, f, indent=2)
        
    return {"json_output": json_output}

def generate_html_node(state: AgentState) -> Dict[str, Any]:
    print("--- Generating HTML ---")
    html_path = render_html_report(
        domain=state.get("domain", "Unknown"),
        num_versions=len(state['versions']),
        diffs=state['diffs']
    )
    
    # Auto-open
    open_in_browser(html_path)
    
    return {"html_path": html_path}

def feedback_node(state: AgentState) -> Dict[str, Any]:
    print("--- Waiting for User Feedback ---")
    # In a real LangGraph server, this would be an interrupt.
    # Here we are running locally, so we might simulate or just rely on the user 
    # checking the HTML. The 'user' here is the collaborative user I'm pair programming with
    # or the end-user of the script.
    
    # The requirement says: "Allow user to make changes and go back based on the user input to 4."
    # We will check the 'user_feedback' key in state. 
    # If this key is set to 'retry', we go back.
    # But how do we get the input? 
    # We can prompt via input() in the terminal if running interactively.
    
    print("Check the opened HTML file.")
    user_input = input("Enter 'approve' to finish, or enter correction Instructions to re-analyze: ")
    
    if user_input.strip().lower() == 'approve' or user_input.strip() == '':
        return {"user_feedback": "approve"}
    else:
        # We treat any other input as correction instructions that should guide the analysis?
        # For simplicity, we just loop back. 
        # A more advanced version would add this feedback to the LLM prompt.
        return {"user_feedback": user_input}

