import os
import json
import difflib
from typing import List, Dict, Any, Optional, Tuple


def find_line_range(
    snippet: str,
    content: str,
    diff_text: str = "",
    diff_mode: str = "new",
) -> Optional[List[int]]:
    """Return a 1-based [start, end] line range for *snippet* inside *content*.

    Strategy (in order):
    1. If *snippet* is non-empty, try an exact multi-line substring search.
    2. If not found (or *snippet* is empty), extract candidate text from
       *diff_text* (lines prefixed with '-' for old, '+' for new).
    3. Fuzzy-search the candidate text as a sliding window over *content* lines,
       picking the window with the highest SequenceMatcher ratio.

    Returns ``None`` (null) if failed to find the lines in all attempts.
    """
    # Pre-rstrip lines to speed up exact search
    lines = [l.rstrip() for l in content.splitlines()] if content else []
    if not lines:
        return None

    def _exact_search(text: str) -> Tuple[int, int] | None:
        """Locate *text* block inside *lines*; return 1-based (start, end) or None."""
        needle_stripped = [l.rstrip() for l in text.strip().splitlines()]
        if not needle_stripped:
            return None
        n = len(needle_stripped)
        for i in range(len(lines) - n + 1):
            if lines[i:i + n] == needle_stripped:
                return (i + 1, i + n)
        return None

    def _fuzzy_search(text: str) -> Tuple[int, int] | None:
        """Find the best-matching window in *lines* for *text*.

        Returns None if no match is found (e.g. best_ratio is 0.0 or lower).
        """
        needle_lines = [l.rstrip() for l in text.strip().splitlines()]
        if not needle_lines:
            return None
        n = max(1, len(needle_lines))
        needle_joined = "\n".join(needle_lines)

        needle_words = set(needle_joined.lower().split())
        if not needle_words:
            return None

        # Precompute words per line
        line_words = [set(l.lower().split()) for l in lines]
        match_count = [len(w & needle_words) for w in line_words]

        # Calculate sliding window sums of size n
        num_windows = max(1, len(lines) - n + 1)
        window_sums = []
        if len(match_count) >= n:
            current_sum = sum(match_count[:n])
            window_sums.append((current_sum, 0))
            for i in range(1, num_windows):
                current_sum = current_sum - match_count[i - 1] + match_count[i + n - 1]
                window_sums.append((current_sum, i))
        else:
            # Document is shorter than the needle
            window_sums.append((sum(match_count), 0))

        # Find max sum
        max_sum = max(item[0] for item in window_sums)
        if max_sum == 0:
            return None

        # Sort windows by sum descending and take the top candidates (close to max_sum or top 5)
        window_sums.sort(key=lambda x: x[0], reverse=True)
        candidates = []
        for s, idx in window_sums:
            if len(candidates) < 10 and s >= max_sum * 0.8:
                candidates.append(idx)
            else:
                break

        best_ratio = -1.0
        best_start = 0
        for i in candidates:
            window = lines[i:i + n]
            window_joined = "\n".join(window)
            ratio = difflib.SequenceMatcher(
                None, window_joined, needle_joined, autojunk=False
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        if best_ratio <= 0.0:
            return None
        return (best_start + 1, best_start + n)

    def _extract_from_diff(mode: str) -> str:
        """Reconstruct the old or new version snippet from the unified-diff text, including context."""
        prefix = "-" if mode == "old" else "+"
        other_prefix = "+" if mode == "old" else "-"
        extracted = []
        for line in (diff_text or "").splitlines():
            if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("@@"):
                continue
            if line.startswith(other_prefix):
                continue
            if line.startswith(prefix):
                extracted.append(line[1:])
            elif line.startswith(" "):
                extracted.append(line[1:])
            elif not line.startswith("\\"):
                extracted.append(line)
        return "\n".join(extracted)

    # --- Main logic ---
    candidate = snippet.strip() if snippet else ""

    if candidate:
        result = _exact_search(candidate)
        if result:
            return list(result)
        # Exact failed — fall through to fuzzy with the snippet
        fuzzy_res = _fuzzy_search(candidate)
        if fuzzy_res:
            return list(fuzzy_res)

    # Try to reconstruct from diff text
    from_diff = _extract_from_diff(diff_mode)
    if from_diff.strip():
        result = _exact_search(from_diff)
        if result:
            return list(result)
        fuzzy_res = _fuzzy_search(from_diff)
        if fuzzy_res:
            return list(fuzzy_res)

    # Nothing worked — return None
    return None


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
