import os
import json
from typing import List, Dict, Any, Optional


def get_clean_doc_name(domain: str) -> str:
    """Extracts a clean, human-readable document name from the domain path/URL."""
    base = os.path.basename(domain)
    name_root, _ = os.path.splitext(base)
    # If name_root is like pep-0773, turn to PEP 773
    if name_root.lower().startswith('pep-'):
        try:
            num = int(name_root.split('-')[1])
            return f"PEP {num}"
        except Exception:
            return name_root.upper()
    return name_root


def fetch_document_popularity(domain: str) -> Optional[Dict[str, Any]]:
    """
    Queries the LLM to find if there are document-specific popularity, rating,
    star, citation, or ranking metrics for this document. Returns the metric and value if found.
    """
    doc_name = get_clean_doc_name(domain)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
        
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
        
        prompt = f"""For the requirement document/standard/protocol '{doc_name}', find if there are any document-specific popularity metrics, ratings, stars, or rankings (for example: PEP popularity surveys, RFC citation indices, community ranking/vote count, or Gist stars).
        
        If a rating, star count, or ranking exists specifically for this document (not the whole repository), return the value and the metric name.
        Otherwise, return null.
        
        Return the result as a valid JSON object in the format:
        {{
            "exists": true,
            "metric_name": "PEP Popularity Rank" or "Gist Stars" etc.,
            "value": "Top 10" or "45 stars" etc.
        }}
        or if it does not exist:
        {{
            "exists": false
        }}
        Do not include markdown formatting or backticks in the response. Return raw JSON only."""
        
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                content = "\n".join(lines[1:-1])
        
        data = json.loads(content)
        if data.get("exists") and data.get("metric_name") and data.get("value"):
            return {
                "metric_name": data["metric_name"],
                "value": data["value"]
            }
        return None
    except Exception as e:
        print(f"Error fetching document popularity: {e}")
        return None


def fetch_author_jobs(authors: List[str], domain: str) -> Dict[str, str]:
    """
    Queries the LLM to identify the job titles, professional roles, or
    affiliations for the given authors within the context of the document domain.
    """
    if not authors:
        return {}
    unique_authors = list(set(a for a in authors if a and a not in ("Cached", "Unknown")))
    if not unique_authors:
        return {}
        
    doc_name = get_clean_doc_name(domain)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {author: "Contributor" for author in unique_authors}
        
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
        
        authors_str = ", ".join(unique_authors)
        prompt = f"""For the project or document '{doc_name}', identify the job title, professional role, or affiliation for each of the following contributors/authors:
{authors_str}

Return a valid JSON object mapping each author's name to their job title or affiliation (e.g. "Core Developer", "Software Engineer", "Professor at University X", "Author"). Keep it concise.
Return ONLY raw JSON. Do not wrap in markdown or backticks."""

        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                content = "\n".join(lines[1:-1])
        
        jobs = json.loads(content)
        # Ensure all requested authors have an entry
        for author in unique_authors:
            if author not in jobs:
                jobs[author] = "Contributor"
        return jobs
    except Exception as e:
        print(f"Error fetching author jobs: {e}")
        return {author: "Contributor" for author in unique_authors}



def extract_reason_type(diff):
    """Extract reason type from multiple possible structures."""
    if not isinstance(diff, dict):
        return "Unknown"

    # New structure support
    if "reason_type" in diff:
        return diff.get("reason_type", "Unknown")

    # Legacy structure support
    if "reason type" in diff:
        return diff.get("reason type", "Unknown")

    return "Unknown"


def extract_version_id(diff):
    """Extract version id from multiple possible structures."""
    if not isinstance(diff, dict):
        return None

    # New structure
    new_version = diff.get("new_version")
    if isinstance(new_version, dict):
        if "version_id" in new_version:
            return new_version.get("version_id")
        if "version id" in new_version:
            return new_version.get("version id")

    # Direct field fallback
    if "version_id" in diff:
        return diff.get("version_id")

    return None


def get_diffs(data):
    """
    Support both old and new dataset structures.

    Old structure:
        data['diffs']

    New structure:
        diffs may be stored per-version.
    """

    # Old structure
    if "diffs" in data and isinstance(data["diffs"], list):
        return data["diffs"]

    # New structure: aggregate diffs from versions
    aggregated_diffs = []

    versions = data.get("versions", {})
    if isinstance(versions, dict):
        for version_key, version_data in versions.items():
            if not isinstance(version_data, dict):
                continue

            version_diffs = version_data.get("diffs", [])

            if isinstance(version_diffs, list):
                for diff in version_diffs:
                    if not isinstance(diff, dict):
                        continue

                    # Ensure version metadata exists
                    if "new_version" not in diff:
                        diff["new_version"] = {
                            "version_id": version_data.get(
                                "version_id",
                                version_key
                            )
                        }

                    aggregated_diffs.append(diff)

    return aggregated_diffs
