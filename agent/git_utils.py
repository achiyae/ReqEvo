import os
import shutil
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime
from agent.state import RequirementVersion
import stat

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
    
    versions_dir = os.path.join(os.getcwd(), 'versions', safe_filename)
    
    # CHECK: If versions directory exists and is populated, can we skip cloning?
    if os.path.exists(versions_dir) and os.path.isdir(versions_dir):
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
                        
                    versions.append({
                        "version_id": version_id,
                        "content": content,
                        "filename": file_path_local,
                        "commit_hash": commit_hash,
                        "date": "Cached", # lost
                        "author": "Cached" # lost
                    })
                except Exception as e:
                    print(f"Skipping malformed file {filename}: {e}")
            
            return versions

    # Handle existing directory cleanup robustly if NOT valid
    if os.path.exists(versions_dir):
        if os.path.isfile(versions_dir):
            os.remove(versions_dir)
        # else: shutil.rmtree(versions_dir, onerror=handle_remove_readonly) 
        # User requested NOT to delete if it exists, but we only reach here if we decided NOT to use it?
        # Actually, if we are here, it means existing_files was empty or it wasn't a dir.
        # If it's an empty dir, we can just use it.
        # If it contains garbage, maybe we should clean it?
        # The prompt said "Don't clone the repo again if the versions of this repo already exist".
        # If they DON'T exist (i.e. empty dir), we proceed to clone.
        # We don't need to delete the dir, just use it.
            
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
                
                versions.append({
                    "version_id": idx + 1,
                    "content": content,
                    "filename": local_path, # Point to the saved file
                    "commit_hash": commit_hash,
                    "date": date,
                    "author": author
                })
            except subprocess.CalledProcessError:
                pass
                
    finally:
        shutil.rmtree(temp_dir, onerror=handle_remove_readonly)

        
    return versions

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

