from agent.metadata_fetcher import extract_version_id
from agent.metadata_fetcher import extract_reason_type
from agent.metadata_fetcher import get_diffs
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, MaxNLocator


def get_vid_sort_key(k):
    try:
        return float(k)
    except Exception:
        return str(k)


def main():
    outputs_dir = "outputs"

    reason_types = Counter()
    total_versions = 0
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

    if not os.path.exists(outputs_dir):
        print(f"Directory '{outputs_dir}' not found.")
        return

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

        total_documents += 1
        total_versions += non_empty_versions
        total_changes += len(diffs)

        doc_names.append(domain)
        doc_versions.append(non_empty_versions)

        if non_empty_versions > 0:
            doc_avg_changes.append(len(diffs) / non_empty_versions)
        else:
            doc_avg_changes.append(0)

        doc_version_changes.append([
            version_counts[vid] for vid in doc_versions_sorted
        ])
        doc_total_diffs.append(len(diffs))
        doc_reason_counts.append(doc_reason_types)

        doc_version_reasons_list.append([
            doc_version_reason_types[vid]
            for vid in doc_versions_sorted
        ])

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

    if total_versions > 0:
        avg_changes_per_version = total_changes / total_versions
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

    try:
        plots_dir = "plots"
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
        print("\nSaved histogram plot to 'reason_types_histogram.png'.")

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
        print("Saved versions per document plot to 'versions_per_document.png'.")

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
            "Saved amount of changes per version plot to "
            "'amount_of_changes_in_each_document_version.png'."
        )

        # Plot 4: Normalized stacked bar chart
        all_reasons = [item[0] for item in reason_types.most_common()]

        if len(all_reasons) >= 1:
            fig, ax = plt.subplots(figsize=(14, 8))

            cmap = plt.get_cmap('tab20')

            reason_color = {
                reason: cmap(i % 20)
                for i, reason in enumerate(all_reasons)
            }

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

            print("Saved stacked bar plot to 'reason_types_per_document.png'.")

            # Plot 5: Normalized stacked bar chart for joint categories
            fig5, ax5 = plt.subplots(figsize=(14, 8))

            categories = ["Shortening", "Clarification", "Fix", "New"]
            category_colors = {
                "Shortening": "#f39c12",    # Warm orange/amber
                "Clarification": "#3498db", # Bright sky blue
                "Fix": "#e74c3c",           # Muted red/coral
                "New": "#2ecc71"            # Emerald green
            }
            category_mapping = {
                "deletion": "Shortening",
                "summarization/shortening": "Shortening",
                "generalization": "Shortening",
                "clarification": "Clarification",
                "demonstration": "Clarification",
                "meaning": "Fix",
                "mistake": "Fix",
                "contradiction": "Fix",
                "new": "New"
            }

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
                        cat = category_mapping.get(reason.lower(), reason)
                        cat_counts[cat] += count

                    total_v_diffs = sum(cat_counts.values())

                    for cat in categories:
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
                                color=category_colors.get(cat, "#7f8c8d")
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
                mpatches.Patch(color=category_colors[cat], label=cat)
                for cat in categories
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

            print("Saved stacked bar plot to 'reason_categories_per_document.png'.")

            # Plot 6: Stacked bar chart for joint categories displaying exact amounts
            fig6, ax6 = plt.subplots(figsize=(14, 8))

            # Calculate the maximum changes in any version to scale y-limit nicely
            max_v_changes = 0
            for i in range(n_docs):
                for v_counts in doc_version_reasons_list[i]:
                    cat_counts = Counter()
                    for reason, count in v_counts.items():
                        cat = category_mapping.get(reason.lower(), reason)
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
                        cat = category_mapping.get(reason.lower(), reason)
                        cat_counts[cat] += count

                    total_v_diffs = sum(cat_counts.values())

                    for cat in categories:
                        count = cat_counts.get(cat, 0)

                        if count > 0:
                            ax6.bar(
                                x_pos,
                                count,
                                width=bar_width * 0.9,
                                bottom=bottom,
                                color=category_colors.get(cat, "#7f8c8d")
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
                mpatches.Patch(color=category_colors[cat], label=cat)
                for cat in categories
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

            print("Saved stacked bar plot to 'amount_of_changes_reason_categories_per_document.png'.")

    except ImportError:
        print(
            "\nmatplotlib is not installed. "
            "Skipping graphical plot generation."
        )


if __name__ == "__main__":
    main()
