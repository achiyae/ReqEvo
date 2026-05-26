import os
import shutil
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime
from agent.state import RequirementVersion
import stat
import json
import re

def handle_remove_readonly(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If the error is due to an access error (read only file),
    it attempts to add write permission and then retries.
    If the error is for another reason it re-raises the error.
    """
    # Clear the readonly bit and reattempt the removal
    os.chmod(path, stat.S_IWRITE)
    try:
        func(path)
    except Exception:
        pass


def parse_github_url(url: str) -> Dict[str, str]:
    """
    Parses a GitHub URL to extract repo URL and file path.
    Example: https://github.com/python/peps/blob/main/pep-0008.txt
    Returns: {'repo_url': 'https://github.com/python/peps.git', 'file_path': 'pep-0008.txt'}
    """
    # Simple heuristic
    if 'github.com' not in url:
        raise ValueError("Currently only GitHub URLs are supported.")
    
    parts = url.split('/')
    if len(parts) < 7:
        raise ValueError("Invalid GitHub URL format.")
    
    user = parts[3]
    repo = parts[4]
    # parts[5] is usually 'blob' or 'tree'
    # parts[6] is branch
    # parts[7:] is path
    
    branch = parts[6]
    file_path = "/".join(parts[7:])
    
    repo_url = f"https://github.com/{user}/{repo}.git"
    
    return {
        "repo_url": repo_url,
        "branch": branch,
        "file_path": file_path
    }

def run_git_command(command: List[str], cwd: str) -> str:
    result = subprocess.run(
        command, 
        cwd=cwd, 
        capture_output=True, 
        text=True, 
        check=True,
        encoding='utf-8',
        errors='replace'
    )
    return result.stdout.strip()

def parse_document_headers(content: str) -> Dict[str, str]:
    headers = {}
    if not content:
        return headers
    
    # Strip any leading BOM or whitespace
    lines = content.lstrip().splitlines()
    current_key = None
    
    for line in lines:
        if not line.strip():
            # First empty line ends header section
            break
            
        # Continuation line (indented)
        if (line.startswith(' ') or line.startswith('\t')) and current_key:
            headers[current_key] = (headers[current_key] + " " + line.strip()).strip()
            continue
            
        # Check for Key: Value format
        match = re.match(r'^([A-Za-z0-9_-]+):\s*(.*)$', line)
        if match:
            current_key = match.group(1).strip()
            headers[current_key] = match.group(2).strip()
        else:
            # First non-matching non-indented line ends headers
            break
            
    return headers

def clean_author_name(author_str: str) -> List[str]:
    if not author_str:
        return []
    # Split by comma
    raw_authors = author_str.split(',')
    cleaned = []
    for ra in raw_authors:
        # Remove email inside angle brackets
        name = re.sub(r'<[^>]+>', '', ra).strip()
        # Clean any extra quotes or brackets
        name = name.strip('"\' ')
        if name:
            cleaned.append(name)
    return cleaned

def enrich_version_with_headers(v: Dict[str, Any]) -> Dict[str, Any]:
    content = v.get('content', '')
    headers = parse_document_headers(content)
    v['headers'] = headers
    
    # Extract authors
    if 'Author' in headers:
        v_authors = clean_author_name(headers['Author'])
        v['authors'] = v_authors
    else:
        if 'authors' not in v:
            v['authors'] = []
            
    # Extract date
    if 'Created' in headers:
        v['date'] = headers['Created']
        
    return v

def fetch_file_history(repo_url: str, file_path: str, branch: str = 'main') -> List[RequirementVersion]:
    """
    Clones repo to temp dir, retrieves history of specific file, 
    AND saves each version to a local 'versions/{filename}' directory.
    """
    temp_dir = tempfile.mkdtemp()
    print(f"--- Cloning {repo_url} to {temp_dir} ---")
    
    versions: List[RequirementVersion] = []
    
    # Create local versions directory
    safe_filename = file_path.replace('/', '_').replace('\\', '_')
    versions_dir = os.path.join(os.getcwd(), 'versions', safe_filename)
    
    # CHECK: If versions directory exists and is populated, can we skip cloning?
    if os.path.exists(versions_dir) and os.path.isdir(versions_dir):
        metadata_path = os.path.join(versions_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                for v_meta in meta.get('versions', []):
                    filename = os.path.basename(v_meta['filename'])
                    file_path_local = os.path.join(versions_dir, filename)
                    with open(file_path_local, 'r', encoding='utf-8') as f_content:
                        content = f_content.read()
                    
                    versions.append(enrich_version_with_headers({
                        "version_id": v_meta['version_id'],
                        "content": content,
                        "filename": file_path_local,
                        "commit_hash": v_meta.get('commit_hash'),
                        "date": v_meta.get('date'),
                        "num_lines": v_meta.get('num_lines'),
                        "author_job": v_meta.get('author_job'),
                        "headers": v_meta.get('headers'),
                        "authors": v_meta.get('authors') or []
                    }))
                print(f"Found existing versions with metadata in {versions_dir}, skipping clone.")
                return versions
            except Exception as e:
                print(f"Error loading metadata.json cache in {versions_dir}: {e}. Reverting to standard parsing.")
                versions = []

        existing_files = sorted([f for f in os.listdir(versions_dir) if f.startswith('v')], key=lambda x: int(x.split('_')[0][1:]))
        if existing_files:
            print(f"Found existing versions in {versions_dir}, skipping clone.")
            # Reconstruct versions from files
            for filename in existing_files:
                # Filename format: v{id}_{hash}.{ext}
                file_path_local = os.path.join(versions_dir, filename)
                try:
                    parts = filename.split('_')
                    version_id_str = parts[0] # v1
                    version_id = int(version_id_str[1:])
                    
                    commit_hash = parts[1].split('.')[0] # hash part
                    
                    with open(file_path_local, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    versions.append(enrich_version_with_headers({
                        "version_id": version_id,
                        "content": content,
                        "filename": file_path_local,
                        "commit_hash": commit_hash,
                        "date": "Cached", # lost
                        "authors": [], # lost
                        "num_lines": len(content.splitlines()),
                        "author_job": None
                    }))
                except Exception as e:
                    print(f"Skipping malformed file {filename}: {e}")
            
            return versions

    # Handle existing directory cleanup robustly if NOT valid
    if os.path.exists(versions_dir):
        if os.path.isfile(versions_dir):
            os.remove(versions_dir)
            
    if not os.path.exists(versions_dir):
        os.makedirs(versions_dir)

    try:
        run_git_command(['git', 'clone', '-b', branch, repo_url, '.'], temp_dir)
        
        log_cmd = [
            'git', 'log', 
            '--pretty=format:%H|%an|%ad', 
            '--date=iso', 
            '--reverse',
            '--', file_path
        ]
        
        log_output = run_git_command(log_cmd, temp_dir)
        
        if not log_output:
            print(f"No history found for {file_path}")
            return []
            
        commits = log_output.splitlines()
        print(f"Found {len(commits)} revisions for {file_path}")
        
        for idx, line in enumerate(commits):
            parts = line.split('|')
            if len(parts) < 3:
                continue
            
            commit_hash = parts[0]
            author = parts[1]
            date = parts[2]
            
            # Get content
            show_cmd = ['git', 'show', f"{commit_hash}:{file_path}"]
            try:
                content = run_git_command(show_cmd, temp_dir)
                
                # Save to local file
                short_hash = commit_hash[:7]
                # Ext might be .txt or .rst
                ext = os.path.splitext(file_path)[1] or '.txt'
                local_filename = f"v{idx+1}_{short_hash}{ext}"
                local_path = os.path.join(versions_dir, local_filename)
                
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                versions.append(enrich_version_with_headers({
                    "version_id": idx + 1,
                    "content": content,
                    "filename": local_path, # Point to the saved file
                    "commit_hash": commit_hash,
                    "date": date,
                    "authors": [author] if author and author not in ("Cached", "Unknown") else [],
                    "num_lines": len(content.splitlines()),
                    "author_job": None
                }))
            except subprocess.CalledProcessError:
                pass
                
        # Save metadata.json
        metadata_path = os.path.join(versions_dir, 'metadata.json')
        meta_to_save = {
            "versions": []
        }
        for v in versions:
            meta_to_save["versions"].append({
                "version_id": v["version_id"],
                "filename": os.path.basename(v["filename"]),
                "commit_hash": v.get("commit_hash"),
                "date": v.get("date"),
                "num_lines": v.get("num_lines"),
                "author_job": v.get("author_job"),
                "headers": v.get("headers"),
                "authors": v.get("authors")
            })
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta_to_save, f, indent=2)
            
    finally:
        shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
        
    return versions

def save_extended_cache_metadata(versions_dir: str, authors_jobs: Dict[str, str], document_popularity: Optional[Dict[str, Any]]):
    """Saves additional resolved metadata (jobs, document popularity) back to cached metadata.json."""
    metadata_path = os.path.join(versions_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # Update versions list to populate author jobs if missing
            for v in meta.get('versions', []):
                v_auths = v.get('authors', [])
                if v_auths:
                    jobs_list = [authors_jobs.get(auth, "Unknown") for auth in v_auths]
                    if len(set(jobs_list)) == 1:
                        v['author_job'] = jobs_list[0]
                    else:
                        v['author_job'] = ", ".join(f"{auth}: {job}" for auth, job in zip(v_auths, jobs_list))
            
            meta['authors_jobs'] = authors_jobs
            meta['document_popularity'] = document_popularity
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"Error updating cache metadata: {e}")

def get_local_git_remote() -> Optional[str]:
    """Gets the origin remote URL of the current workspace."""
    try:
        import subprocess
        res = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip() or None
    except Exception:
        return None

def get_local_git_author(file_path: str) -> Optional[str]:
    """Gets the author of the last commit that touched the file."""
    try:
        import subprocess
        res = subprocess.run(
            ['git', 'log', '-1', '--pretty=format:%an', '--', file_path],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip() or None
    except Exception:
        return None

def get_local_git_date(file_path: str) -> Optional[str]:
    """Gets the date of the last commit that touched the file."""
    try:
        import subprocess
        res = subprocess.run(
            ['git', 'log', '-1', '--pretty=format:%ad', '--date=iso', '--', file_path],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip() or None
    except Exception:
        return None

def get_git_diff_hunks(repo_path: str, old_hash: str, new_hash: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Uses git diff to get structured changes.
    Since we are working with a temp repo that is deleted, this function faces a challenge:
    The repo is gone by the time we want to compute diffs in the 'compute_diffs_node'.
    
    Solution:
    1. We need to keep the repo alive? 
       OR 
    2. We can run diff BETWEEN the locally saved files using `git diff --no-index`?
    
    The user asked to use "diff feature that git provides". `git diff --no-index file1 file2` works even without a repo.
    """
    pass # Replaced by actual implementation below

def compute_git_diff(old_file: str, new_file: str) -> str:
    """
    Uses git diff --no-index to compare two files.
    """
    cmd = ['git', 'diff', '--no-index', '--unified=3', old_file, new_file]
    # git diff returns exit code 1 if differences found, 0 if none.
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    return result.stdout

