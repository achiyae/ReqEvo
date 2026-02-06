import os
import difflib
import json
import time
from typing import List, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from agent.state import AgentState, RequirementVersion, DiffEntry
from agent.utils import render_html_report, open_in_browser, REASON_DEFINITIONS
from agent.git_utils import parse_github_url, fetch_file_history, compute_git_diff

# --- Analysis Models ---
class ChangeAnalysis(BaseModel):
    reason_type: str = Field(description=f"One of: {', '.join(REASON_DEFINITIONS.keys())}")
    reason_text: str = Field(description="Explanation for the change")

# --- Nodes ---

def load_files_node(state: AgentState) -> Dict[str, Any]:
    """Loads requirement files from the specified paths OR Git URL."""
    print("--- Loading Files ---")
    
    # Check if 'domain' acts as a placeholder for URL or if we check a specific state key
    # For now, let's look at the 'domain' field or a 'file_paths' entry.
    # The prompt said we'll ask for domain OR git url.
    
    domain_or_url = state.get('domain', '')
    versions: List[RequirementVersion] = []
    
    # Heuristic: Is it a GitHub URL?
    if 'github.com' in domain_or_url:
        print(f"Detected GitHub URL: {domain_or_url}")
        try:
            git_info = parse_github_url(domain_or_url)
            versions = fetch_file_history(
                git_info['repo_url'], 
                git_info['file_path'], 
                git_info['branch']
            )
            # Update domain name to be the file name
            return {
                "versions": versions, 
                "domain": git_info['file_path'],
                "file_paths": [domain_or_url]
            }
        except Exception as e:
            print(f"Error fetching from Git: {e}")
            return {"versions": []}

    # Fallback to local files
    files = state.get('file_paths', [])
    
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
                "filename": os.path.basename(file_path),
                "commit_hash": None,
                "date": None,
                "author": None
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
        
        # Check if we should use Git Diff (if files are saved locally AND have hash)
        if old_v.get('commit_hash') and new_v.get('commit_hash') and old_v.get('filename') and new_v.get('filename'):
            diff_output = compute_git_diff(old_v['filename'], new_v['filename'])
            
            # Simple Hunk Parser
            hunks = []
            current_hunk = []
            in_hunk = False
            
            for line in diff_output.splitlines():
                if line.startswith('@@'):
                    if current_hunk:
                        hunks.append("\n".join(current_hunk))
                        current_hunk = []
                    in_hunk = True
                    current_hunk.append(line)
                elif in_hunk:
                    current_hunk.append(line)
                    
            if current_hunk:
                hunks.append("\n".join(current_hunk))
                
            for hunk in hunks:
                diffs.append({
                    "diff_id": global_diff_id,
                    "old_version_id": old_v['version_id'],
                    "new_version_id": new_v['version_id'],
                    "diff_text": hunk,
                    "reason_type": "Pending Analysis",
                    "reason_text": "Pending...",
                    "old_content_snippet": "", 
                    "new_content_snippet": "",
                    "old_commit_hash": old_v.get('commit_hash'),
                    "old_date": old_v.get('date'),
                    "new_commit_hash": new_v.get('commit_hash'),
                    "new_date": new_v.get('date')
                })
                global_diff_id += 1
                
        else:
            # Fallback to difflib logic for plain text files
            old_lines = [line for line in old_v['content'].splitlines() if line.strip()]
            new_lines = [line for line in new_v['content'].splitlines() if line.strip()]
            
            matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    continue
                
                old_chunk = old_lines[i1:i2]
                new_chunk = new_lines[j1:j2]
                
                max_len = max(len(old_chunk), len(new_chunk))
                
                for k in range(max_len):
                    sub_old = old_chunk[k] if k < len(old_chunk) else None
                    sub_new = new_chunk[k] if k < len(new_chunk) else None
                    
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
                        "new_content_snippet": sub_new if sub_new else "",
                        "old_commit_hash": old_v.get('commit_hash'),
                        "old_date": old_v.get('date'),
                        "new_commit_hash": new_v.get('commit_hash'),
                        "new_date": new_v.get('date')
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

    llm = ChatOpenAI(temperature=0, model="gpt-5-nano", api_key=os.environ.get("OPENAI_API_KEY"))
    parser = JsonOutputParser(pydantic_object=ChangeAnalysis)
    
    # Build Reason Types list for prompt
    reasons_prompt_list = "\n".join([f"- {k}: {v}" for k, v in REASON_DEFINITIONS.items()])
    
    prompt_messages = [
        ("system", "You are an expert business analyst specializing in requirement evolution."),
        ("user", f"""Analyze the changes between these two requirement document versions.
        
        Old Version:
        {{old_text}}
        
        New Version:
        {{new_text}}
        
        Diff:
        {{diff_text}}
        
        Identify the PRIMARY reason for the changes. If there are multiple distinct changes, 
        summarize the dominant one or the most critical one.
        
        Possible reason types:
        {reasons_prompt_list}
        
        {{feedback_section}}
        
        {{format_instructions}}
        """)
    ]
    
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    chain = prompt | llm | parser
    
    versions = {v['version_id']: v['content'] for v in state['versions']}
    updated_diffs = []
    
    # Check for feedback
    feedback = state.get('user_feedback')
    feedback_section = ""
    # legacy global feedback support
    if feedback and isinstance(feedback, str) and feedback != "approve":
         feedback_section = f"IMPORTANT: The user rejected a previous analysis with the following feedback/correction:\n'{feedback}'\nPlease adjust your analysis to respect this feedback."
    
    for diff in state['diffs']:
        # Logic for determining if we should analyze or skip
        should_analyze = False
        specific_reason = None
        specific_explanation = None
        
        # 1. Existing Feedback Check
        if isinstance(feedback, dict):
             # Check for structured feedback
             specific_reason = feedback.get(f"reason_{diff['diff_id']}")
             specific_explanation = feedback.get(f"explanation_{diff['diff_id']}")
             
             if specific_reason or specific_explanation:
                 should_analyze = True
        
        # 2. Logic
        if diff.get('reason_type') == "Pending Analysis":
            should_analyze = True
        elif feedback == 'retry':
            # Legacy global retry
            should_analyze = True
            
        if not should_analyze:
            updated_diffs.append(diff)
            continue

        print(f"Analyzing diff {diff['diff_id']} (Reason: {specific_reason}, Exp: {specific_explanation})...")
            
        old_text = versions.get(diff['old_version_id'], "")
        new_text = versions.get(diff['new_version_id'], "")
        
        current_feedback_section = ""
        if specific_reason or specific_explanation:
            current_feedback_section = "IMPORTANT: The user rejected the previous analysis.\n"
            if specific_reason:
                current_feedback_section += f"- The user SPECIFIED the reason type must be: '{specific_reason}'.\n"
            if specific_explanation:
                current_feedback_section += f"- User Explanation/Context: '{specific_explanation}'.\n"
            current_feedback_section += "Please adjust your analysis to strictly reflect this feedback."
        elif feedback_section: 
             # Fallback to global feedback
             current_feedback_section = feedback_section

        try:
            result = chain.invoke({
                "old_text": old_text,
                "new_text": new_text,
                "diff_text": diff['diff_text'],
                "feedback_section": current_feedback_section,
                "format_instructions": parser.get_format_instructions()
            })
            
            diff['reason_type'] = result['reason_type']
            diff['reason_text'] = result['reason_text']
            # Replaced/cleared previous correction if any, effectively
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
    
    # Calculate execution time
    start_time = state.get('start_time')
    if start_time:
        elapsed = time.time() - start_time
        print(f"--- Analysis Completed in {elapsed:.2f} seconds ---")
        
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
    
    # Generate filename: output_{basename}.json
    domain = state.get("domain", "Unknown Domain")
    base_name = os.path.basename(domain)
    name_root, _ = os.path.splitext(base_name)
    safe_name = "".join(c for c in name_root if c.isalnum() or c in ('-', '_')).strip()
    if not safe_name:
        safe_name = "analysis"
        
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        
    output_filename = os.path.join(outputs_dir, f"output_{safe_name}.json")
    
    # Save to file
    with open(output_filename, "w") as f:
        json.dump(json_output, f, indent=2)
        
    return {"json_output": json_output}

def generate_html_node(state: AgentState) -> Dict[str, Any]:
    print("--- Generating HTML ---")
    
    domain = state.get("domain", "Unknown")
    
    # Generate filename: report_{basename}.html
    # Remove extension if present
    base_name = os.path.basename(domain)
    name_root, _ = os.path.splitext(base_name)
    
    # Basic sanitization
    safe_name = "".join(c for c in name_root if c.isalnum() or c in ('-', '_')).strip()
    if not safe_name:
        safe_name = "analysis"
        
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(os.getcwd(), "reports")
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        
    output_filename = os.path.join(reports_dir, f"report_{safe_name}.html")
    
    html_path = render_html_report(
        domain=domain,
        num_versions=len(state['versions']),
        diffs=state['diffs'],
        reason_types=list(REASON_DEFINITIONS.keys()),
        output_path=output_filename
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
    
    
    print("Check the opened HTML file. Waiting for feedback via web UI...")
    
    feedback_data = {}
    event = threading.Event()
    
    class FeedbackHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/submit':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                feedback_data.update(data)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'received'}).encode('utf-8'))
                
                # Signal completion
                event.set()
                
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
        def log_message(self, format, *args):
            return # Silence logs

    server_address = ('localhost', 8000)
    httpd = HTTPServer(server_address, FeedbackHandler)
    
    # Run server in a separate thread so we can wait on the event
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    print(f"Feedback server running at http://localhost:8000")
    
    try:
        # Loop until event is set
        while not event.is_set():
            # Wait for 1 second at a time to allow signal handling for Ctrl+C
            event.wait(1.0)
    except KeyboardInterrupt:
        print("\nUser interrupted. Shutting down server...")
        # Optionally exit or just cleanup
        httpd.shutdown()
        # Re-raise so LangGraph or main can handle it? 
        # Or return END? 
        return {"user_feedback": "approve"} # Treat interruption as done/approve? Or exit

    httpd.shutdown()
    
    action = feedback_data.get('action', 'approve')
    print(f"User action: {action}")
    
    if action == 'approve':
        return {"user_feedback": "approve"}
    else:
        # Pass the whole data dict as feedback
        return {"user_feedback": feedback_data}

