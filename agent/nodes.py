import os
import difflib
import json
import time
import pickle
import threading
from typing import List, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from agent.state import AgentState, RequirementVersion, DiffEntry
from agent.utils import render_html_report, open_in_browser, REASON_DEFINITIONS
from agent.git_utils import (
    parse_github_url, 
    fetch_file_history, 
    compute_git_diff, 
    save_extended_cache_metadata, 
    get_local_git_remote,
    get_local_git_author,
    get_local_git_date,
    enrich_version_with_headers
)

# --- Analysis Models ---
class ChangeAnalysis(BaseModel):
    reason_type: str = Field(description=f"One of: {', '.join(REASON_DEFINITIONS.keys())}")
    reason_text: str = Field(description="Explanation for the change")

# --- Nodes ---

def load_files_node(state: AgentState) -> Dict[str, Any]:
    """Loads requirement files from the specified paths OR Git URL."""
    print("--- Loading Files ---")
    
    domain_or_url = state.get('domain', '')
    versions: List[RequirementVersion] = []
    files = state.get('file_paths', [])
    domain = domain_or_url
    
    # 1. Fetch versions (either from GitHub or fallback to local files)
    if 'github.com' in domain_or_url:
        print(f"Detected GitHub URL: {domain_or_url}")
        try:
            git_info = parse_github_url(domain_or_url)
            versions = fetch_file_history(
                git_info['repo_url'], 
                git_info['file_path'], 
                git_info['branch']
            )
            domain = git_info['file_path']
            files = [domain_or_url]
        except Exception as e:
            print(f"Error fetching from Git: {e}")
            versions = []
    else:
        # Fallback to local files
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
                    "authors": []
                })

    # 2. Enrich versions and state with metadata
    # Count lines per version
    for idx, v in enumerate(versions):
        v['num_lines'] = len(v['content'].splitlines()) if v.get('content') else 0
        
        # If local files, try to extract git info
        if not v.get('commit_hash') and files:
            if idx < len(files):
                file_path = files[idx]
                if os.path.exists(file_path):
                    author = get_local_git_author(file_path)
                    date = get_local_git_date(file_path)
                    if author and author not in ("Cached", "Unknown"):
                        v['authors'] = [author]
                    if date:
                        v['date'] = date
        # Enrich with document headers (PEP headers)
        v = enrich_version_with_headers(v)
        # Remove per-version headers to avoid redundancy; they'll be stored at document level later
        if 'headers' in v:
            v.pop('headers')
    
    # Determine document-level headers from the latest version (if any)
    document_headers = versions[-1].get('headers') if versions else {}
    
    # Check if there is cached metadata
    cached_jobs = {}
    cached_pop = None
    if 'github.com' in domain_or_url:
        try:
            git_info = parse_github_url(domain_or_url)
            safe_filename = git_info['file_path'].replace('/', '_').replace('\\', '_')
            versions_dir = os.path.join(os.getcwd(), 'versions', safe_filename)
            metadata_path = os.path.join(versions_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                cached_jobs = meta.get('authors_jobs', {})
                cached_pop = meta.get('document_popularity')
        except Exception:
            pass

    # Extract unique authors
    authors_set = set()
    for v in versions:
        v_auths = v.get('authors')
        if v_auths:
            for auth in v_auths:
                if auth and auth not in ("Cached", "Unknown"):
                    authors_set.add(auth)
    authors = list(authors_set)
    
    # Resolve author jobs
    from agent.metadata_fetcher import fetch_author_jobs, fetch_document_popularity
    
    # Use cached if possible
    authors_jobs = {}
    missing_authors = []
    for author in authors:
        if author in cached_jobs:
            authors_jobs[author] = cached_jobs[author]
        else:
            missing_authors.append(author)
            
    if missing_authors:
        fetched_jobs = fetch_author_jobs(missing_authors, domain)
        authors_jobs.update(fetched_jobs)
    else:
        authors_jobs.update(cached_jobs)
        
    # Resolve document popularity
    if cached_pop is not None:
        doc_pop = cached_pop
    else:
        doc_pop = fetch_document_popularity(domain_or_url)
        
    # Apply jobs to versions
    for v in versions:
        v_auths = v.get('authors', [])
        if v_auths:
            jobs_list = [authors_jobs.get(auth, "Unknown") for auth in v_auths]
            if len(set(jobs_list)) == 1:
                v['author_job'] = jobs_list[0]
            else:
                v['author_job'] = ", ".join(f"{auth}: {job}" for auth, job in zip(v_auths, jobs_list))
        else:
            v['author_job'] = None
            
    # Save back to cache if git
    if 'github.com' in domain_or_url:
        try:
            git_info = parse_github_url(domain_or_url)
            safe_filename = git_info['file_path'].replace('/', '_').replace('\\', '_')
            versions_dir = os.path.join(os.getcwd(), 'versions', safe_filename)
            save_extended_cache_metadata(versions_dir, authors_jobs, doc_pop)
        except Exception:
            pass
            
    return {
        "versions": versions,
        "domain": domain,
        "file_paths": files,
        "total_authors": len(authors),
        "authors_jobs": authors_jobs,
        "document_popularity": doc_pop,
        "document_headers": document_headers,
        "document_authors": authors
    }

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
                    # Skip the @@ line
                elif in_hunk:
                    # Skip git metadata lines
                    if line.startswith('---') or line.startswith('+++') or line.startswith('diff') or line.startswith('index') or line.startswith('new file') or line.startswith('deleted file'):
                        continue
                    current_hunk.append(line)
                    
            if current_hunk:
                hunks.append("\n".join(current_hunk))
                
            for hunk in hunks:
                diffs.append({
                    "diff_id": global_diff_id,
                    "old_version_id": old_v['version_id'],
                    "new_version_id": new_v['version_id'],
                    "diff_text": hunk.strip(),
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
                        "diff_text": diff_text.strip(),
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
        ("system", """You are an expert business analyst specializing in requirement evolution. 
        Your goal is to accurately classify changes between document versions.
        
        ### CLASSIFICATION GUIDELINES & LESSONS:
        1. **Meaning vs. New**: If a change discusses the same requirement but extends it or implements it differently, classify it as 'Meaning'. Reserve 'New' ONLY for entirely fresh requirements that have no predecessor in the previous version.
        2. **Summarization/shortening vs. New**: If a part of a requirement (like a redundant note, header, or parenthetical) is removed, it is 'Summarization/shortening', NOT 'New'.
        3. **Typo vs. New**: Case changes, spelling corrections (e.g., 'Nuget' to 'NuGet'), and grammar fixes are 'Typo', NOT 'New'.
        4. **Mistake vs. Deletion**: If a statement is corrected (e.g., changing a rule or a platform support statement), it is a 'Mistake' correction. 'Deletion' is only for removing an ENTIRE requirement block that is now redundant.
        5. **Meaning vs. Mistake**: If a rule changes (e.g., a command name changes from 'run' to 'exec'), it represents a change in intent/behavior and should be classified as 'Meaning', NOT 'Mistake'.
        6. **Prioritize Significant Changes**: If a diff contains multiple parts (e.g., a minor 'Clarification' and a 'Meaning' change), always classify based on the most significant or "strongest" change. For example, 'Meaning' is more important than 'Clarification'.
        7. **Summarization/shortening vs. Clarification**: If a change involves combining multiple lines, rows, or bullet points into one while keeping the wording and meaning nearly identical, classify it as 'Summarization/shortening', NOT 'Clarification'.
        """),
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
                current_feedback_section += "CRITICAL: You MUST output this exact reason type, even if you disagree or if it contradicts the definitions. Do not argue. Your task is to explain why this chosen type is correct.\n"
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
    """Generate JSON output adhering to the new schema.

    The output contains:
    1. Top‑level document metadata (authors, headers, etc.)
    2. A ``versions`` dictionary keyed by version ID with minimal metadata
    3. A ``diffs`` list that references version IDs but does not duplicate
       full document information.
    """
    # Build a dictionary of versions keyed by version_id with minimal metadata
    versions_dict = {
        v['version_id']: {
            "version_id": v['version_id'],
            "commit_hash": v.get('commit_hash'),
            "date": v.get('date'),
            "num_lines": v.get('num_lines'),
            "content": v.get('content')
        }
        for v in state['versions']
    }

    # Build diffs list without redundant version information
    json_diffs = []
    for d in state['diffs']:
        old_v = versions_dict.get(d['old_version_id'])
        new_v = versions_dict.get(d['new_version_id'])
        json_diffs.append({
            "diff_id": d['diff_id'],
            "reason_type": d.get('reason_type'),
            "reason_text": d.get('reason_text'),
            "old_version": {
                "version_id": d['old_version_id'],
                "content": old_v.get('content') if old_v else "",
                "commit_hash": d.get('old_commit_hash'),
                "date": d.get('old_date'),
                "num_lines": old_v.get('num_lines') if old_v else None,
            },
            "new_version": {
                "version_id": d['new_version_id'],
                "content": new_v.get('content') if new_v else "",
                "commit_hash": d.get('new_commit_hash'),
                "date": d.get('new_date'),
                "num_lines": new_v.get('num_lines') if new_v else None,
            },
            "diff": d['diff_text'],
            "old_content_snippet": d.get('old_content_snippet', ''),
            "new_content_snippet": d.get('new_content_snippet', ''),
        })

    json_output = {
        "domain": state.get("domain", "Unknown Domain"),
        "number_of_versions": len(state['versions']),
        "document_metadata": {
            "num_lines_latest": state['versions'][-1].get('num_lines') if state['versions'] else 0,
            "total_authors": state.get("total_authors", 0),
            "document_popularity": state.get("document_popularity"),
            "authors_jobs": state.get("authors_jobs", {}),
            "headers": state.get("document_headers", {}),
            "authors": state.get("document_authors", []),
        },
        "versions": versions_dict,
        "diffs": json_diffs,
    }

    # Determine a safe base filename from the domain
    domain = state.get("domain", "Unknown Domain")
    base_name = os.path.basename(domain)
    name_root, _ = os.path.splitext(base_name)
    safe_name = "".join(c for c in name_root if c.isalnum() or c in ('-', '_')).strip()
    if not safe_name:
        safe_name = "analysis"

    # Write JSON output
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_filename = os.path.join(outputs_dir, f"output_{safe_name}.json")
    with open(output_filename, "w") as f:
        json.dump(json_output, f, indent=2)

    # Persist full state as pickle for later reuse
    states_dir = os.path.join(os.getcwd(), "states")
    os.makedirs(states_dir, exist_ok=True)
    pickle_filename = os.path.join(states_dir, f"{safe_name}.pkl")
    with open(pickle_filename, "wb") as f:
        pickle.dump(dict(state), f)

    return {"json_output": json_output}


def generate_html_node(state: AgentState) -> Dict[str, Any]:
    print("--- Generating HTML ---")
    
    domain = state.get("domain", "Unknown")
    is_final = state.get("is_final", False)
    
    # Generate filename: report_{basename}.html
    base_name = os.path.basename(domain)
    name_root, _ = os.path.splitext(base_name)
    safe_name = "".join(c for c in name_root if c.isalnum() or c in ('-', '_')).strip()
    if not safe_name:
        safe_name = "analysis"
        
    if is_final:
        reports_dir = os.path.join(os.getcwd(), "final_reports")
        output_filename = os.path.join(reports_dir, f"final_report_{safe_name}.html")
    else:
        reports_dir = os.path.join(os.getcwd(), "reports")
        output_filename = os.path.join(reports_dir, f"report_{safe_name}.html")
        
    versions_map = {v['version_id']: v for v in state['versions']}
    enriched_diffs = []
    for d in state['diffs']:
        d_enriched = dict(d)
        old_v = versions_map.get(d['old_version_id'])
        new_v = versions_map.get(d['new_version_id'])
        if old_v:
            d_enriched['old_author'] = ", ".join(old_v.get('authors', [])) if old_v.get('authors') else None
            d_enriched['old_author_job'] = old_v.get('author_job')
            d_enriched['old_num_lines'] = old_v.get('num_lines')
        if new_v:
            d_enriched['new_author'] = ", ".join(new_v.get('authors', [])) if new_v.get('authors') else None
            d_enriched['new_author_job'] = new_v.get('author_job')
            d_enriched['new_num_lines'] = new_v.get('num_lines')
        enriched_diffs.append(d_enriched)

    document_metadata = {
        "num_lines_latest": state['versions'][-1].get('num_lines') if state['versions'] else 0,
        "total_authors": state.get("total_authors", 0),
        "document_popularity": state.get("document_popularity"),
        "authors_jobs": state.get("authors_jobs", {}),
        "headers": state.get("document_headers", {}),
        "document_authors": state.get("document_authors", [])
    }
        
    html_path = render_html_report(
        domain=domain,
        num_versions=len(state['versions']),
        diffs=enriched_diffs,
        reason_types=list(REASON_DEFINITIONS.keys()),
        output_path=output_filename,
        is_final=is_final,
        document_metadata=document_metadata
    )
    
    # Auto-open
    open_in_browser(html_path)
    
    return {"html_path": html_path}

def feedback_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- Waiting for User Feedback via HTML ---")
    print("Check the opened HTML file. Feedback server running at http://localhost:8000")
    
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
    
    # Run server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    try:
        # Loop until event is set
        while not event.is_set():
            event.wait(1.0)
    except KeyboardInterrupt:
        print("\nUser interrupted. Shutting down server...")
        httpd.shutdown()
        return {"user_feedback": "approve"}

    httpd.shutdown()
    
    action = feedback_data.get('action', 'approve')
    print(f"User action: {action}")
    
    # Return feedback data directly (including action type like 'finish')
    return {"user_feedback": feedback_data}
