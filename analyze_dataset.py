import os
import sys
import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, MaxNLocator

from agent.setup_reviewers import get_existing_reviewers
from agent.metadata_fetcher import extract_reason_type, extract_version_id, get_diffs
from agent.tracking_utils import (
    CATEGORIES,
    CATEGORY_MAPPING,
    CATEGORY_COLORS,
    get_vid_sort_key,
    get_new_vid as _get_new_vid,
    get_new_range as _get_new_range,
    build_chains,
    augment_chains_with_new_sentinel,
    chains_to_trees,
)


def generate_plots(plots_dir, reason_types, doc_names, doc_versions, doc_version_changes, doc_version_reasons_list, reason_color=None):
    os.makedirs(plots_dir, exist_ok=True)

    labels = [item[0] for item in reason_types.most_common()]
    values = [item[1] for item in reason_types.most_common()]

    # Plot 1: Reason types histogram
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='salmon')
    plt.xlabel('Reason Type')
    plt.ylabel('Frequency')
    plt.title('Histogram of Reason Types')
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'reason_types_histogram.png'))
    print(f"Saved histogram plot to '{os.path.join(plots_dir, 'reason_types_histogram.png')}'.")

    # Plot 1b: Reason categories histogram
    category_counts = Counter()
    for reason, count in reason_types.items():
        cat = CATEGORY_MAPPING.get(reason.lower(), reason)
        category_counts[cat] += count

    cat_labels = [item[0] for item in category_counts.most_common()]
    cat_values = [item[1] for item in category_counts.most_common()]
    cat_colors = [CATEGORY_COLORS.get(cat, "#7f8c8d") for cat in cat_labels]

    plt.figure(figsize=(10, 6))
    plt.bar(cat_labels, cat_values, color=cat_colors)
    plt.xlabel('Reason Category')
    plt.ylabel('Frequency')
    plt.title('Histogram of Reason Categories')
    plt.xticks(rotation=45, ha='right')
    if cat_values:
        max_val = max(cat_values)
        if max_val > 100:
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'reason_categories_histogram.png'))
    print(f"Saved categories histogram plot to '{os.path.join(plots_dir, 'reason_categories_histogram.png')}'.")


    # Plot 2: Versions per document
    plt.figure(figsize=(12, 6))
    plt.bar(doc_names, doc_versions, color='lightgreen')
    plt.xlabel('Document')
    plt.ylabel('Number of Versions')
    plt.title('Number of Versions per Document')
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'versions_per_document.png'))
    print(f"Saved versions per document plot to '{os.path.join(plots_dir, 'versions_per_document.png')}'.")

    # Plot 3: Amount of changes in each document version
    plt.figure(figsize=(12, 6))

    total_width = 0.8
    max_versions = (
        max([len(counts) for counts in doc_version_changes])
        if doc_version_changes else 1
    )

    bar_width = total_width / max(1, max_versions)

    for i, counts in enumerate(doc_version_changes):
        if not counts:
            continue

        num_counts = len(counts)
        start_x = i - (num_counts - 1) * bar_width / 2

        for j, count in enumerate(counts):
            x_pos = start_x + j * bar_width
            plt.bar(
                x_pos,
                count,
                width=bar_width * 0.8,
                color='darkblue',
                alpha=0.8,
                zorder=2
            )

    plt.xticks(range(len(doc_names)), doc_names, rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(MultipleLocator(5))
    plt.xlabel('Document')
    plt.ylabel('Amount of Changes')
    plt.title('Amount of Changes in each Document Version')
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plots_dir,
            'amount_of_changes_in_each_document_version.png'
        )
    )

    print(
        f"Saved amount of changes per version plot to "
        f"'{os.path.join(plots_dir, 'amount_of_changes_in_each_document_version.png')}'."
    )

    # Plot 4: Normalized stacked bar chart
    all_reasons = [item[0] for item in reason_types.most_common()]

    if len(all_reasons) >= 1:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Use the globally-consistent reason_color if provided; build a local
        # fallback only for reason types not already in the shared map.
        if reason_color is None:
            cmap = plt.get_cmap('tab20')
            reason_color = {r: cmap(i % 20) for i, r in enumerate(sorted(all_reasons))}

        n_docs = len(doc_names)
        total_width = 0.8

        max_versions = (
            max([len(v_list) for v_list in doc_version_reasons_list])
            if doc_version_reasons_list else 1
        )

        bar_width = total_width / max(1, max_versions)

        for i in range(n_docs):
            v_list = doc_version_reasons_list[i]
            num_v = len(v_list)

            if num_v == 0:
                continue

            start_x = i - (num_v - 1) * bar_width / 2

            for v_idx, v_counts in enumerate(v_list):
                x_pos = start_x + v_idx * bar_width
                bottom = 0.0
                total_v_diffs = sum(v_counts.values())

                for reason in all_reasons:
                    count = v_counts.get(reason, 0)

                    if count > 0:
                        percentage = (
                            (count / total_v_diffs) * 100
                            if total_v_diffs > 0 else 0
                        )

                        ax.bar(
                            x_pos,
                            percentage,
                            width=bar_width * 0.9,
                            bottom=bottom,
                            color=reason_color[reason]
                        )

                        bottom += percentage

                # Add total number of changes on top of the bar
                if total_v_diffs > 0:
                    ax.text(x_pos, 101, str(total_v_diffs), ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(x_pos, 1, '0', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(range(n_docs))
        ax.set_xticklabels(doc_names, rotation=45, ha='right')
        ax.set_xlabel('Document')
        ax.set_ylabel('Percentage of Change Reason Types (%)')
        ax.set_ylim(0, 110)
        ax.set_title(
            'Percentage of Changes Reason Types '
            'per Version in Each Document'
        )

        legend_handles = [
            mpatches.Patch(color=reason_color[reason], label=reason)
            for reason in all_reasons
        ]
        legend_handles.append(
            mpatches.Patch(color='none', label='Numbers on top: Total changes amount')
        )

        ax.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        plt.tight_layout()

        fig.savefig(
            os.path.join(plots_dir, 'reason_types_per_document.png')
        )

        print(f"Saved stacked bar plot to '{os.path.join(plots_dir, 'reason_types_per_document.png')}'.")

        # Plot 5: Normalized stacked bar chart for joint categories
        fig5, ax5 = plt.subplots(figsize=(14, 8))


        for i in range(n_docs):
            v_list = doc_version_reasons_list[i]
            num_v = len(v_list)

            if num_v == 0:
                continue

            start_x = i - (num_v - 1) * bar_width / 2

            for v_idx, v_counts in enumerate(v_list):
                x_pos = start_x + v_idx * bar_width
                bottom = 0.0

                # Map v_counts to joint categories
                cat_counts = Counter()
                for reason, count in v_counts.items():
                    cat = CATEGORY_MAPPING.get(reason.lower(), reason)
                    cat_counts[cat] += count

                total_v_diffs = sum(cat_counts.values())

                for cat in CATEGORIES:
                    count = cat_counts.get(cat, 0)

                    if count > 0:
                        percentage = (
                            (count / total_v_diffs) * 100
                            if total_v_diffs > 0 else 0
                        )

                        ax5.bar(
                            x_pos,
                            percentage,
                            width=bar_width * 0.9,
                            bottom=bottom,
                            color=CATEGORY_COLORS.get(cat, "#7f8c8d")
                        )

                        bottom += percentage

                # Add total number of changes on top of the bar
                if total_v_diffs > 0:
                    ax5.text(x_pos, 101, str(total_v_diffs), ha='center', va='bottom', fontsize=8)
                else:
                    ax5.text(x_pos, 1, '0', ha='center', va='bottom', fontsize=8)

        ax5.set_xticks(range(n_docs))
        ax5.set_xticklabels(doc_names, rotation=45, ha='right')
        ax5.set_xlabel('Document')
        ax5.set_ylabel('Percentage of Change Reason Categories (%)')
        ax5.set_ylim(0, 110)
        ax5.set_title(
            'Percentage of Changes Reason Categories '
            'per Version in Each Document'
        )

        legend_handles5 = [
            mpatches.Patch(color=CATEGORY_COLORS[cat], label=cat)
            for cat in CATEGORIES
        ]
        legend_handles5.append(
            mpatches.Patch(color='none', label='Numbers on top: Total changes amount')
        )

        ax5.legend(
            handles=legend_handles5,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        plt.tight_layout()

        fig5.savefig(
            os.path.join(plots_dir, 'reason_categories_per_document.png')
        )

        print(f"Saved stacked bar plot to '{os.path.join(plots_dir, 'reason_categories_per_document.png')}'.")

        # Plot 6: Stacked bar chart for joint categories displaying exact amounts
        fig6, ax6 = plt.subplots(figsize=(14, 8))

        # Calculate the maximum changes in any version to scale y-limit nicely
        max_v_changes = 0
        for i in range(n_docs):
            for v_counts in doc_version_reasons_list[i]:
                cat_counts = Counter()
                for reason, count in v_counts.items():
                    cat = CATEGORY_MAPPING.get(reason.lower(), reason)
                    cat_counts[cat] += count
                max_v_changes = max(max_v_changes, sum(cat_counts.values()))

        y_limit = max(1, max_v_changes) * 1.1
        label_offset = max(1, max_v_changes) * 0.01

        for i in range(n_docs):
            v_list = doc_version_reasons_list[i]
            num_v = len(v_list)

            if num_v == 0:
                continue

            start_x = i - (num_v - 1) * bar_width / 2

            for v_idx, v_counts in enumerate(v_list):
                x_pos = start_x + v_idx * bar_width
                bottom = 0.0

                # Map v_counts to joint categories
                cat_counts = Counter()
                for reason, count in v_counts.items():
                    cat = CATEGORY_MAPPING.get(reason.lower(), reason)
                    cat_counts[cat] += count

                total_v_diffs = sum(cat_counts.values())

                for cat in CATEGORIES:
                    count = cat_counts.get(cat, 0)

                    if count > 0:
                        ax6.bar(
                            x_pos,
                            count,
                            width=bar_width * 0.9,
                            bottom=bottom,
                            color=CATEGORY_COLORS.get(cat, "#7f8c8d")
                        )

                        bottom += count

                # Add total number of changes on top of the bar
                ax6.text(
                    x_pos,
                    total_v_diffs + label_offset,
                    str(total_v_diffs),
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        ax6.set_xticks(range(n_docs))
        ax6.set_xticklabels(doc_names, rotation=45, ha='right')
        ax6.set_xlabel('Document')
        ax6.set_ylabel('Amount of Changes per Category')
        ax6.set_ylim(0, y_limit)
        ax6.set_title(
            'Amount of Changes per Category '
            'per Version in Each Document'
        )

        legend_handles6 = [
            mpatches.Patch(color=CATEGORY_COLORS[cat], label=cat)
            for cat in CATEGORIES
        ]
        legend_handles6.append(
            mpatches.Patch(color='none', label='Numbers on top: Total changes amount')
        )

        ax6.legend(
            handles=legend_handles6,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        plt.tight_layout()

        fig6.savefig(
            os.path.join(plots_dir, 'amount_of_changes_reason_categories_per_document.png')
        )

        print(f"Saved stacked bar plot to '{os.path.join(plots_dir, 'amount_of_changes_reason_categories_per_document.png')}'.")

    plt.close('all')


def generate_requirements_tracking_plot(plots_dir, doc_name, diffs, versions_sorted_ids, versions_data=None, reason_color=None):
    """Generate a horizontal tracking chart for requirements that changed in multiple versions.

    Each row represents a requirement chain (the same requirement evolving across versions).
    Trees with common prefixes share rows up to the split point, fanning out on new rows below.
    Each cell is coloured by the reason_type of that version's change.
    """
    if not diffs:
        return

    chains = build_chains(diffs)
    multi_chains = [c for c in chains if len(c) >= 2]
    if not multi_chains:
        return

    multi_chains = augment_chains_with_new_sentinel(multi_chains, versions_data, versions_sorted_ids)

    # Convert flat chains to tree structure (merging common prefixes)
    roots = chains_to_trees(multi_chains)
    roots = [r for r in roots if r.get("children")]
    if not roots:
        return

    # Collect all reason types present in this document's chains.
    all_reason_types = sorted({
        extract_reason_type(d)
        for chain in multi_chains for d in chain
    })
    # Use the globally-consistent reason_color map if provided; build a local
    # fallback only for reason types not already in the shared map.
    if reason_color is None:
        cmap = plt.get_cmap("tab20")
        reason_color = {r: cmap(i % 20) for i, r in enumerate(all_reason_types)}

    # Map version IDs to x-axis positions
    vid_to_x = {str(vid): idx for idx, vid in enumerate(versions_sorted_ids)}

    # Assign row indices to tree nodes (common prefixes share row, branches go on rows below)
    node_row = {}

    def assign_rows(node, current_row):
        node_row[id(node)] = current_row
        children = node.get("children", [])
        if not children:
            return current_row
        max_row = current_row
        for i, child in enumerate(children):
            if i == 0:
                child_max = assign_rows(child, current_row)
                max_row = max(max_row, child_max)
            else:
                next_row = max_row + 1
                child_max = assign_rows(child, next_row)
                max_row = max(max_row, child_max)
        return max_row

    current_row = 0
    for root in roots:
        max_r = assign_rows(root, current_row)
        current_row = max_r + 1

    n_rows = current_row
    fig_height = max(4, n_rows * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(max(10, len(versions_sorted_ids) * 1.2), fig_height))

    bar_height = 0.6

    # Draw connecting lines and node boxes
    def draw_node(node):
        diff = node["diff"]
        row_idx = node_row[id(node)]
        vid = str(_get_new_vid(diff))
        x = vid_to_x.get(vid)

        children = node.get("children", [])

        for i, child in enumerate(children):
            child_diff = child["diff"]
            child_row = node_row[id(child)]
            child_vid = str(_get_new_vid(child_diff))
            child_x = vid_to_x.get(child_vid)

            if x is not None and child_x is not None:
                if child_row == row_idx:
                    # Same row: horizontal dashed line
                    ax.plot([x, child_x], [row_idx, row_idx], color="#95a5a6", linestyle="--", linewidth=1.0, zorder=1)
                else:
                    # Branching row: vertical/elbow dashed line from split node to branch start
                    ax.plot([x, x, child_x], [row_idx, child_row, child_row], color="#95a5a6", linestyle="--", linewidth=1.0, zorder=1)

            draw_node(child)

        if x is not None:
            reason = extract_reason_type(diff)
            color = reason_color.get(reason, "#7f8c8d")
            ax.broken_barh(
                [(x - 0.4, 0.8)],
                (row_idx - bar_height / 2, bar_height),
                facecolors=color,
                edgecolors="white",
                linewidth=0.5,
                zorder=2,
            )
            new_lr = _get_new_range(diff)
            if new_lr and len(new_lr) >= 2:
                bar_label = f"{new_lr[0]}-{new_lr[1]}"
            else:
                bar_label = ""
            if bar_label:
                ax.text(
                    x,
                    row_idx,
                    bar_label,
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                    fontweight="bold",
                    clip_on=True,
                    zorder=3,
                )

    for root in roots:
        draw_node(root)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([""] * n_rows)  # no tick labels; title describes the axis
    ax.tick_params(axis="y", length=0)  # also hide tick marks
    ax.set_xticks(range(len(versions_sorted_ids)))
    ax.set_xticklabels([str(v) for v in versions_sorted_ids], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Version")
    ax.set_ylabel("Text Diffs (recognized by Git)")
    ax.set_title(f"Requirements Tracking — {doc_name}")
    ax.set_xlim(-1, len(versions_sorted_ids))
    ax.set_ylim(-0.8, n_rows - 0.2)
    ax.invert_yaxis()

    legend_handles = [
        mpatches.Patch(color=reason_color.get(r, "#7f8c8d"), label=r)
        for r in all_reason_types
    ]
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )

    plt.tight_layout()

    tracking_dir = os.path.join(plots_dir, "requirements_tracking")
    os.makedirs(tracking_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in doc_name)
    out_path = os.path.join(tracking_dir, f"{safe_name}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved requirements tracking plot to '{out_path}'.")


def main():
    existing_reviewers = get_existing_reviewers()

    if len(sys.argv) < 2:
        print("Error: Reviewer name command line argument is required.")
        print("Usage: python analyze_dataset.py <reviewer_name>")
        print(f"Available reviewers: {', '.join(existing_reviewers)}")
        sys.exit(1)

    reviewer_arg = sys.argv[1].lower()
    reviewer = None
    for r in existing_reviewers:
        if r.lower() == reviewer_arg:
            reviewer = r
            break

    if not reviewer:
        print(f"Error: Reviewer '{sys.argv[1]}' not found. Available reviewers: {', '.join(existing_reviewers)}")
        sys.exit(1)

    print(f"Analyzing dataset for reviewer: {reviewer}...")

    outputs_dir = os.path.join("dataset_reviewers", reviewer, "outputs")

    reason_types = Counter()
    total_versions = 0
    total_version_changes = 0
    total_documents = 0
    total_changes = 0

    doc_names = []
    doc_versions = []
    doc_avg_changes = []
    doc_version_changes = []
    doc_total_diffs = []
    doc_reason_counts = []
    doc_version_reasons_list = []
    doc_metadata_list = []
    _pending_tracking = []  # deferred (domain, diffs, versions) tuples

    if not os.path.exists(outputs_dir):
        print(f"Directory '{outputs_dir}' not found.")
        return

    # Ensure the plots directory exists
    plots_dir = os.path.join("dataset_reviewers", reviewer, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for filename in os.listdir(outputs_dir):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(outputs_dir, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Support both naming conventions
        num_versions = (
            data.get("number_of_versions")
            or data.get("number of versions")
            or 0
        )

        diffs = get_diffs(data)

        # Filter non-substantive changes
        filtered_diffs = []
        for diff in diffs:
            reason = extract_reason_type(diff)
            if reason not in ("Non-substantive", "Typo"):
                filtered_diffs.append(diff)

        diffs = filtered_diffs

        domain_raw = data.get("domain", filename)
        domain = os.path.splitext(os.path.basename(domain_raw))[0]

        metadata = data.get("document_metadata", {})

        doc_metadata_list.append({
            "domain": domain,
            "num_lines": metadata.get("num_lines_latest", "N/A"),
            "total_authors": metadata.get("total_authors", "N/A"),
            "popularity": metadata.get("document_popularity"),
            "authors_jobs": metadata.get("authors_jobs", {})
        })

        version_counts = Counter()
        doc_reason_types = Counter()
        doc_version_reason_types = {}

        for diff in diffs:
            vid = extract_version_id(diff)

            if vid is not None:
                version_counts[vid] += 1

                if vid not in doc_version_reason_types:
                    doc_version_reason_types[vid] = Counter()

            reason = extract_reason_type(diff)

            reason_types[reason] += 1
            doc_reason_types[reason] += 1

            if vid is not None:
                doc_version_reason_types[vid][reason] += 1

        # Always count the first version (minimum version ID in the document's versions dictionary, fallback to "1")
        versions_dict = data.get("versions", {})
        if versions_dict:
            first_version_id = min(versions_dict.keys(), key=get_vid_sort_key)
        else:
            first_version_id = "1"

        # Check if the first version is already represented in doc_version_reason_types
        # Normalize to float/string sort key to avoid type mismatch (e.g. 1 vs "1")
        first_vid_val = get_vid_sort_key(first_version_id)
        has_first_version = any(get_vid_sort_key(k) == first_vid_val for k in doc_version_reason_types.keys())

        if not has_first_version:
            doc_version_reason_types[first_version_id] = Counter()

        doc_versions_sorted = sorted(
            list(doc_version_reason_types.keys()),
            key=get_vid_sort_key
        )

        non_empty_versions = len(doc_versions_sorted)

        # Excluding the first version (which has 0 changes) from the version changes list
        first_vid_val = get_vid_sort_key(first_version_id)
        doc_versions_for_changes = [vid for vid in doc_versions_sorted if get_vid_sort_key(vid) != first_vid_val]

        total_documents += 1
        total_versions += non_empty_versions
        total_version_changes += len(doc_versions_for_changes)
        total_changes += len(diffs)

        doc_names.append(domain)
        doc_versions.append(non_empty_versions)

        if len(doc_versions_for_changes) > 0:
            doc_avg_changes.append(len(diffs) / len(doc_versions_for_changes))
        else:
            doc_avg_changes.append(0)

        doc_version_changes.append([
            version_counts[vid] for vid in doc_versions_for_changes
        ])
        doc_total_diffs.append(len(diffs))
        doc_reason_counts.append(doc_reason_types)

        doc_version_reasons_list.append([
            doc_version_reason_types[vid]
            for vid in doc_versions_for_changes
        ])
        # Generate requirements tracking plot for this document.
        # reason_color_global is built after all docs are processed; we pass
        # None here and backfill via a deferred list below.
        _pending_tracking.append((domain, diffs, doc_versions_sorted, data.get("versions", {})))

    print("=== Dataset Analysis ===")
    print(f"Total documents analyzed: {total_documents}")

    if total_documents > 0:
        avg_versions = total_versions / total_documents
        print(
            f"2. Average number of versions per document: "
            f"{avg_versions:.2f}"
        )
    else:
        print("2. Average number of versions per document: N/A")

    if total_version_changes > 0:
        avg_changes_per_version = total_changes / total_version_changes
        print(
            f"3. Average number of changes per version: "
            f"{avg_changes_per_version:.2f}"
        )
    else:
        print("3. Average number of changes per version: N/A")

    print("\n=== Document Metadata Summary ===")

    for doc in doc_metadata_list:
        print(f"\nDocument: {doc['domain']}")
        print(f"  Length (Latest Version): {doc['num_lines']} lines")
        print(f"  Total Unique Authors: {doc['total_authors']}")

        pop = doc['popularity']
        if pop and isinstance(pop, dict):
            print(
                f"  Popularity: "
                f"{pop.get('value')} ({pop.get('metric_name')})"
            )
        else:
            print("  Popularity: N/A")

        if doc['authors_jobs']:
            print("  Authors & Roles:")
            for author, job in doc['authors_jobs'].items():
                print(f"    - {author}: {job}")
        else:
            print("  Authors & Roles: N/A")

    print("\n1. Histogram of Reason Types:")

    if not reason_types:
        print("No reason types found.")
    else:
        max_label_len = max(len(str(r)) for r in reason_types.keys())
        total_reasons = sum(reason_types.values())

        for reason, count in reason_types.most_common():
            bar = "#" * max(1, int(count / total_reasons * 50))
            print(f"{str(reason).ljust(max_label_len)} | {count:4d} {bar}")

    print("\n1b. Histogram of Reason Categories:")

    if not reason_types:
        print("No reason categories found.")
    else:
        category_counts = Counter()
        for reason, count in reason_types.items():
            cat = CATEGORY_MAPPING.get(reason.lower(), reason)
            category_counts[cat] += count

        max_cat_label_len = max(len(str(c)) for c in category_counts.keys())
        total_cats = sum(category_counts.values())

        for cat, count in category_counts.most_common():
            bar = "#" * max(1, int(count / total_cats * 50))
            print(f"{str(cat).ljust(max_cat_label_len)} | {count:4d} {bar}")




    try:
        plots_dir = os.path.join("dataset_reviewers", reviewer, "plots")

        # Build a single, globally-consistent reason_type color map so that
        # all plots (bar charts and tracking charts) use the same color for
        # the same reason type.
        _all_reason_types_global = sorted(reason_types.keys())
        _cmap = plt.get_cmap('tab20')
        reason_color_global = {
            r: _cmap(i % 20)
            for i, r in enumerate(_all_reason_types_global)
        }

        # Now generate the deferred requirements-tracking plots with the
        # globally-consistent color map.
        for _domain, _diffs, _versions, _versions_data in _pending_tracking:
            generate_requirements_tracking_plot(
                plots_dir,
                _domain,
                _diffs,
                _versions,
                versions_data=_versions_data,
                reason_color=reason_color_global,
            )

        # 1. Generate normal plots
        print("\nGenerating standard plots...")
        generate_plots(
            plots_dir,
            reason_types,
            doc_names,
            doc_versions,
            doc_version_changes,
            doc_version_reasons_list,
            reason_color=reason_color_global,
        )

        # 2. Filter INTERESTING documents
        MIN_INTERESTING_VERSIONS = 3
        filtered_indices = [i for i, changes in enumerate(doc_version_changes) if len(changes) >= MIN_INTERESTING_VERSIONS]

        if len(filtered_indices) > 0:
            print(f"\nGenerating plots for interesting documents (with at least {MIN_INTERESTING_VERSIONS} version changes)...")
            filtered_doc_names = [doc_names[i] for i in filtered_indices]
            filtered_doc_versions = [doc_versions[i] for i in filtered_indices]
            filtered_doc_version_changes = [doc_version_changes[i] for i in filtered_indices]
            filtered_doc_version_reasons_list = [doc_version_reasons_list[i] for i in filtered_indices]

            filtered_reason_types = Counter()
            for i in filtered_indices:
                filtered_reason_types.update(doc_reason_counts[i])

            interesting_docs_plots_dir = os.path.join(plots_dir, "interesting_docs_plots")

            generate_plots(
                interesting_docs_plots_dir,
                filtered_reason_types,
                filtered_doc_names,
                filtered_doc_versions,
                filtered_doc_version_changes,
                filtered_doc_version_reasons_list,
                reason_color=reason_color_global,
            )
        else:
            print(f"\nNo documents found with at least {MIN_INTERESTING_VERSIONS} version changes. Skipping only interesting docs plots.")

    except ImportError:
        print(
            "\nmatplotlib is not installed. "
            "Skipping graphical plot generation."
        )


if __name__ == "__main__":
    main()
