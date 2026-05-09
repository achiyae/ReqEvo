import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator


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
            
        num_versions = data.get("number of versions", 0)
        diffs = data.get("diffs", [])
        domain_raw = data.get("domain", filename)
        domain = os.path.splitext(os.path.basename(domain_raw))[0]
        
        total_documents += 1
        total_versions += num_versions
        total_changes += len(diffs)
        
        doc_names.append(domain)
        doc_versions.append(num_versions)
        doc_avg_changes.append(len(diffs) / num_versions if num_versions > 0 else 0)
        
        version_counts = Counter()
        doc_reason_types = Counter()
        doc_version_reason_types = {}
        for diff in diffs:
            vid = diff.get('new_version', {}).get('version id', None)
            if vid is not None:
                version_counts[vid] += 1
                if vid not in doc_version_reason_types:
                    doc_version_reason_types[vid] = Counter()
            reason = diff.get("reason type", "Unknown")
            reason_types[reason] += 1
            doc_reason_types[reason] += 1
            if vid is not None:
                doc_version_reason_types[vid][reason] += 1
            
        doc_version_changes.append(list(version_counts.values()))
        doc_total_diffs.append(len(diffs))
        doc_reason_counts.append(doc_reason_types)
        
        def get_vid_sort_key(k):
            try: return float(k)
            except: return str(k)
        doc_versions_sorted = sorted(list(doc_version_reason_types.keys()), key=get_vid_sort_key)
        doc_version_reasons_list.append([doc_version_reason_types[vid] for vid in doc_versions_sorted])

    print("=== Dataset Analysis ===")
    print(f"Total documents analyzed: {total_documents}")
    
    if total_documents > 0:
        avg_versions = total_versions / total_documents
        print(f"2. Average number of versions per document: {avg_versions:.2f}")
    else:
        print("2. Average number of versions per document: N/A (no documents found)")

    if total_versions > 0:
        avg_changes_per_version = total_changes / total_versions
        print(f"3. Average number of changes per version: {avg_changes_per_version:.2f}")
    else:
        print("3. Average number of changes per version: N/A (no versions found)")
        
    print("\n1. Histogram of Reason Types:")
    if not reason_types:
        print("No reason types found.")
    else:
        max_label_len = max(len(str(r)) for r in reason_types.keys()) if reason_types else 0
        total_reasons = sum(reason_types.values())
        for reason, count in reason_types.most_common():
            bar = "#" * max(1, int(count / total_reasons * 50))
            print(f"{str(reason).ljust(max_label_len)} | {count:4d} {bar}")
            
    # Optional: Plotting using matplotlib if installed
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
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'versions_per_document.png'))
        print("Saved versions per document plot to 'versions_per_document.png'.")

        # Plot 3: Amount of changes in each document version
        plt.figure(figsize=(12, 6))
        
        # Bars for each version
        total_width = 0.8
        max_versions = max([len(counts) for counts in doc_version_changes]) if doc_version_changes else 1
        bar_width = total_width / max(1, max_versions)
        
        for i, counts in enumerate(doc_version_changes):
            if not counts:
                continue
            num_counts = len(counts)
            start_x = i - (num_counts - 1) * bar_width / 2
            for j, count in enumerate(counts):
                x_pos = start_x + j * bar_width
                plt.bar(x_pos, count, width=bar_width*0.8, color='darkblue', alpha=0.8, zorder=2)

        plt.xticks(range(len(doc_names)), doc_names, rotation=45, ha='right')
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.xlabel('Document')
        plt.ylabel('Amount of Changes')
        plt.title('Amount of Changes in each Document Version')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'amount_of_changes_in_each_document_version.png'))
        print("Saved amount of changes per version plot to 'amount_of_changes_in_each_document_version.png'.")

        # Plot 4: Normalized stacked bar chart of reason types per version per document 
        # all_reasons is sorted most-popular → rarest (from reason_types.most_common())
        all_reasons = [item[0] for item in reason_types.most_common()]
        if len(all_reasons) >= 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            cmap = plt.get_cmap('tab20')
            # Fixed color per reason, indexed by global popularity rank
            reason_color = {reason: cmap(i % 20) for i, reason in enumerate(all_reasons)}
            
            n_docs = len(doc_names)
            total_width = 0.8
            max_versions = max([len(v_list) for v_list in doc_version_reasons_list]) if doc_version_reasons_list else 1
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
                    # Stack in fixed order: most popular at bottom, rarest at top
                    for reason in all_reasons:
                        count = v_counts.get(reason, 0)
                        if count > 0:
                            percentage = (count / total_v_diffs) * 100 if total_v_diffs > 0 else 0
                            ax.bar(x_pos, percentage, width=bar_width*0.9, bottom=bottom,
                                   color=reason_color[reason])
                            bottom += percentage
            
            ax.set_xticks(range(n_docs))
            ax.set_xticklabels(doc_names, rotation=45, ha='right')
            ax.set_xlabel('Document')
            ax.set_ylabel('Percentage of Change Reason Types (%)')
            ax.set_title('Percentage of Changes Reason Types per Version in Each Document')
            
            # Build legend in global popularity order (most popular first)
            legend_handles = [
                mpatches.Patch(color=reason_color[reason], label=reason)
                for reason in all_reasons
            ]
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, 'reason_types_per_document.png'))
            print("Saved stacked bar plot to 'reason_types_per_document.png'.")

    except ImportError:
        print("\nmatplotlib is not installed. Skipping graphical plot generation.")

if __name__ == "__main__":
    main()
