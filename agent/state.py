from typing import List, Dict, Any, TypedDict, Optional

class RequirementVersion(TypedDict):
    version_id: int
    content: str
    filename: str
    commit_hash: Optional[str]
    date: Optional[str]
    num_lines: Optional[int]

class DiffEntry(TypedDict):
    diff_id: int
    old_version_id: int
    new_version_id: int
    diff_text: str
    reason_type: str
    reason_text: str
    old_content_snippet: str
    new_content_snippet: str
    old_commit_hash: Optional[str]
    old_date: Optional[str]
    new_commit_hash: Optional[str]
    new_date: Optional[str]

class AgentState(TypedDict):
    domain: str
    file_paths: List[str]
    versions: List[RequirementVersion]
    diffs: List[DiffEntry]
    json_output: Dict[str, Any]
    html_path: str
    user_feedback: Optional[Any] # None, 'approve' (str), or correction dict
    iteration: int
    start_time: float
    is_final: bool
    total_authors: Optional[int]
    authors_jobs: Optional[Dict[str, str]]
    document_popularity: Optional[Dict[str, Any]]
    document_headers: Optional[Dict[str, str]]   # headers from the latest version
    document_authors: Optional[List[str]]         # unique authors across all versions
