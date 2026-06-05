import os
import shutil

from pathlib import Path


def setup_new_reviewer(reviewer: str):
    """Initialize directory structure for a new reviewer by copying from the AI subdir."""
    base_dirs = ["final_reports", "outputs", "plots", "reports", "states"]
    dataset_reviewers_dir = os.path.join(os.getcwd(), "dataset_reviewers")
    ai_dir = os.path.join(dataset_reviewers_dir, "AI")
    reviewer_dir = os.path.join(dataset_reviewers_dir, reviewer)

    os.makedirs(reviewer_dir, exist_ok=True)

    for d in base_dirs:
        dst_dir = os.path.join(reviewer_dir, d)
        src_dir = os.path.join(ai_dir, d)

        if not os.path.exists(dst_dir):
            if os.path.exists(src_dir):
                print(f"Copying AI/{d} -> {reviewer}/{d} ...")
                shutil.copytree(src_dir, dst_dir)
            else:
                print(f"Creating empty directory: {reviewer}/{d}")
                os.makedirs(dst_dir, exist_ok=True)
        else:
            print(f"Directory already exists, skipping: {reviewer}/{d}")
    
    print(f"Reviewer '{reviewer}' initialized successfully.")


def get_existing_reviewers():
    dataset_reviewers_dir = os.path.join(os.getcwd(), "dataset_reviewers")
    return [p.name for p in Path(dataset_reviewers_dir).iterdir() if p.is_dir()]
