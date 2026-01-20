from typing import List, Dict, Any, TypedDict, Optional

class RequirementVersion(TypedDict):
    version_id: int
    content: str
    filename: str

class DiffEntry(TypedDict):
    diff_id: int
    old_version_id: int
    new_version_id: int
    diff_text: str
    reason_type: str
    reason_text: str
    old_content_snippet: str
    new_content_snippet: str

class AgentState(TypedDict):
    domain: str
    file_paths: List[str]
    versions: List[RequirementVersion]
    diffs: List[DiffEntry]
    json_output: Dict[str, Any]
    html_path: str
    user_feedback: Optional[str] # None, 'approve', or correction instructions
    iteration: int
