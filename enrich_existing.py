import os
import pickle
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prevent opening browser tabs during batch processing
import agent.nodes
import agent.utils
agent.nodes.open_in_browser = lambda path: None
agent.utils.open_in_browser = lambda path: None

from agent.git_utils import save_extended_cache_metadata, enrich_version_with_headers
from agent.metadata_fetcher import fetch_author_jobs, fetch_document_popularity
from agent.nodes import generate_json_node, generate_html_node

def main():
    states_dir = os.path.join(os.getcwd(), "states")
    if not os.path.exists(states_dir):
        print("States directory not found.")
        return
        
    pickle_files = [f for f in os.listdir(states_dir) if f.endswith(".pkl")]
    print(f"Found {len(pickle_files)} existing states to enrich.")
    
    for filename in pickle_files:
        filepath = os.path.join(states_dir, filename)
        print(f"\n--- Enriching {filename} ---")
        
        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        domain = state.get("domain", "")
        print(f"Domain/Document: {domain}")
        
        # 1. Count lines per version and retrieve local author metadata if missing
        versions = state.get("versions", [])
        for v in versions:
            v['num_lines'] = len(v['content'].splitlines()) if v.get('content') else 0
            
            # Migrate old single author to authors list if authors is missing
            old_author = v.get('author')
            if old_author and old_author not in ("Cached", "Unknown") and not v.get('authors'):
                v['authors'] = [old_author]
                
            v = enrich_version_with_headers(v)
            
            # Clean up author field from state
            if 'author' in v:
                del v['author']
                
            if v.get('date') == "Cached":
                v['date'] = None
        
        # 2. Try to load cached jobs and popularity from metadata.json
        cached_jobs = {}
        cached_pop = None
        
        # Construct versions directory path based on domain/file path
        safe_filename = domain.replace('/', '_').replace('\\', '_')
        versions_dir = os.path.join(os.getcwd(), 'versions', safe_filename)
        metadata_path = os.path.join(versions_dir, 'metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                cached_jobs = meta.get('authors_jobs', {})
                cached_pop = meta.get('document_popularity')
                print("Found cached metadata.")
            except Exception as e:
                print(f"Error reading cache metadata: {e}")
                
        # 3. Extract unique authors
        authors_set = set()
        for v in versions:
            v_auths = v.get('authors')
            if v_auths:
                for auth in v_auths:
                    if auth and auth not in ("Cached", "Unknown"):
                        authors_set.add(auth)
        authors = list(authors_set)
        print(f"Unique authors found: {authors}")
        
        # 4. Resolve author jobs
        authors_jobs = {}
        missing_authors = []
        for author in authors:
            if author in cached_jobs:
                authors_jobs[author] = cached_jobs[author]
            else:
                missing_authors.append(author)
                
        if missing_authors:
            print(f"Fetching job roles for: {missing_authors}")
            fetched_jobs = fetch_author_jobs(missing_authors, domain)
            authors_jobs.update(fetched_jobs)
        else:
            authors_jobs.update(cached_jobs)
            
        # 5. Resolve document-specific popularity
        if cached_pop is not None:
            doc_pop = cached_pop
        else:
            print(f"Fetching document-specific popularity/stars for: {domain}")
            doc_pop = fetch_document_popularity(domain)
            
        print(f"Author Jobs: {authors_jobs}")
        print(f"Document Popularity: {doc_pop}")
        
        # 6. Apply metadata back to versions
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
                
        # 7. Update state dict
        state['versions'] = versions
        state['total_authors'] = len(authors)
        state['authors_jobs'] = authors_jobs
        state['document_popularity'] = doc_pop
        
        # 8. Save updated pickle
        try:
            with open(filepath, "wb") as f:
                pickle.dump(state, f)
            print("Pickle state updated.")
        except Exception as e:
            print(f"Error writing pickle {filename}: {e}")
            
        # 9. Save back to metadata.json cache
        if os.path.exists(versions_dir):
            try:
                meta_to_save = {"versions": []}
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
                meta_to_save['authors_jobs'] = authors_jobs
                meta_to_save['document_popularity'] = doc_pop
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_to_save, f, indent=2)
            except Exception as e:
                print(f"Error updating metadata.json: {e}")
            
        # 10. Regenerate JSON output file
        print("Regenerating JSON output...")
        generate_json_node(state)
        
        # 11. Regenerate HTML report
        print("Regenerating HTML report...")
        generate_html_node(state)
        
    print("\n--- Enrichment Complete! ---")

if __name__ == "__main__":
    main()
