import os

from pathlib import Path


def setup_new_reviewer(reviewer: str):
    """Initialize directory structure for a new reviewer."""
    dataset_reviewers_dir = os.path.join(os.getcwd(), "dataset_reviewers")
    reviewer_dir = os.path.join(dataset_reviewers_dir, reviewer)
    os.makedirs(reviewer_dir, exist_ok=True)
    print(f"Reviewer '{reviewer}' initialized successfully.")


def get_existing_reviewers():
    dataset_reviewers_dir = os.path.join(os.getcwd(), "dataset_reviewers")
    return [p.name for p in Path(dataset_reviewers_dir).iterdir() if p.is_dir()]
