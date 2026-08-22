"""
agent/tracking_utils.py
========================
Shared helper functions and constants for tracking requirements, building
requirement chains, matching versions across diffs, and classification.
"""

from agent.metadata_fetcher import extract_reason_type

# ---------------------------------------------------------------------------
# Categories & Mapping Constants
# ---------------------------------------------------------------------------

CATEGORIES = ["Shortening", "Clarification", "Fix", "New"]

CATEGORY_MAPPING = {
    "deletion": "Shortening",
    "summarization/shortening": "Shortening",
    "generalization": "Shortening",
    "clarification": "Clarification",
    "demonstration": "Clarification",
    "meaning": "Fix",
    "mistake": "Fix",
    "contradiction": "Fix",
    "new": "New",
    "shortening": "Shortening",
    "fix": "Fix",
}

CATEGORY_COLORS = {
    "Shortening": "#f39c12",    # Warm orange/amber
    "Clarification": "#3498db", # Bright sky blue
    "Fix": "#e74c3c",           # Muted red/coral
    "New": "#2ecc71"            # Emerald green
}

NON_SUBSTANTIVE = {"Non-substantive", "Typo"}


# ---------------------------------------------------------------------------
# Range & Version Extraction Helpers
# ---------------------------------------------------------------------------

def get_vid_sort_key(k):
    """Sort key helper for version IDs (numeric float if possible, else string)."""
    try:
        return float(k)
    except Exception:
        return str(k)


def ranges_overlap(r1, r2):
    """Return True if two [start, end] ranges overlap (inclusive)."""
    if not r1 or not r2:
        return False
    return r1[0] <= r2[1] and r2[0] <= r1[1]


def get_old_vid(diff):
    """Extract old_version version_id from a diff dictionary."""
    ov = diff.get("old_version")
    if isinstance(ov, dict):
        return ov.get("version_id")
    return diff.get("old_version_id")


def get_new_vid(diff):
    """Extract new_version version_id from a diff dictionary."""
    nv = diff.get("new_version")
    if isinstance(nv, dict):
        return nv.get("version_id")
    return diff.get("new_version_id")


def get_old_range(diff):
    """Extract old_version line_range from a diff dictionary."""
    ov = diff.get("old_version")
    if isinstance(ov, dict):
        return ov.get("line_range")
    return None


def get_new_range(diff):
    """Extract new_version line_range from a diff dictionary."""
    nv = diff.get("new_version")
    if isinstance(nv, dict):
        return nv.get("line_range")
    return None


def extract_snippet(version_dict):
    """Extract requirement text from a version dict using its line_range."""
    if not isinstance(version_dict, dict):
        return None
    content = version_dict.get("content")
    line_range = version_dict.get("line_range")
    if not content or not line_range or len(line_range) < 2:
        return None
    lines = content.split("\n")
    start = line_range[0] - 1
    end = line_range[1]
    if start < 0 or end > len(lines):
        return None
    return "\n".join(lines[start:end]).strip()


def extract_diff_content(diff_text, mode):
    """Extract added (+) or deleted (-) lines from unified diff text."""
    if not diff_text:
        return []
    prefix = '-' if mode == 'old' else '+'
    meta_prefix = '---' if mode == 'old' else '+++'
    lines = []
    for line in diff_text.splitlines():
        if line.startswith(prefix) and not line.startswith(meta_prefix):
            stripped = line[1:].strip()
            if stripped:
                lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# Chain Construction & Matching Helpers
# ---------------------------------------------------------------------------

def build_chains(diffs):
    """
    Build requirement chains from a list of diffs.

    A diff is appended to an existing chain when the text of the requirement
    modified (deleted/changed) by the current diff matches the text produced
    (added/changed) by the chain's last diff.

    If no match is found at the last position (k=1) across all chains, the
    search falls back to one-before-last (k=2), two-before-last (k=3), etc.

    When matching at k=1: the diff is appended directly to the matched chain
    (linear continuation).
    When matching at k>1: the matched chain has already grown past the target,
    so the diff starts a *new branch* chain = chain[:len(chain)-k+1] + [diff],
    representing the requirement evolving in two different directions (split).

    Falls back to snippet text matching and range-based heuristics if needed.
    Returns a list of chains (each chain is a list of diffs).
    """
    def _sort_key(d):
        return (get_vid_sort_key(str(get_new_vid(d))), d.get("diff_id", 0))

    def _is_match(diff, target, old_lines, old_snippet):
        """Return True if diff's old content matches target's new content."""
        target_new_lines = extract_diff_content(target.get("diff", ""), "new")

        # At least 20% non-trivial line overlap
        if old_lines and target_new_lines:
            old_set = set(old_lines)
            new_set = set(target_new_lines)
            common = old_set & new_set
            ratio = len(common) / min(len(old_set), len(new_set))
            if ratio >= 0.20 and any(len(line) > 5 for line in common):
                return True

        return False

    sorted_diffs = sorted(diffs, key=_sort_key)
    chains = []

    for diff in sorted_diffs:
        raw_reason = extract_reason_type(diff)
        cat = CATEGORY_MAPPING.get(raw_reason.lower(), raw_reason)
        if cat == "New":
            # A 'New' diff introduces a new requirement; it can never continue an existing requirement.
            chains.append([diff])
            continue

        diff_text = diff.get("diff", "")
        old_lines = extract_diff_content(diff_text, "old")
        old_snippet = extract_snippet(diff.get("old_version"))

        matched = False
        if chains:
            max_k = max(len(c) for c in chains)
            new_branches = []  # branch chains created for k>1 splits
            for k in range(1, max_k + 1):
                for chain in chains:
                    if len(chain) < k:
                        continue
                    target = chain[-k]
                    if get_vid_sort_key(str(get_old_vid(diff))) < get_vid_sort_key(str(get_new_vid(target))):
                        continue

                    if _is_match(diff, target, old_lines, old_snippet):
                        if k == 1:
                            # Linear continuation: append in place
                            chain.append(diff)
                        else:
                            # Split: branch off from the match point
                            branch = chain[:len(chain) - k + 1] + [diff]
                            new_branches.append(branch)
                        matched = True
                        break

                if matched:
                    break

            chains.extend(new_branches)

        if not matched:
            chains.append([diff])

    return chains


def find_range_in_v1(first_diff, versions_data):
    """
    Return the [start, end] line range of the requirement in version 1.
    Looks for the old_version snippet of first_diff inside version 1 content
    by finding the sliding window with >= 60% line overlap.
    """
    MIN_LINES_OVERLAP_RATIO = 0.6

    if not versions_data:
        return None
    v1_entry = versions_data.get("1")
    if v1_entry is None:
        try:
            v1_key = min(versions_data.keys(), key=get_vid_sort_key)
            v1_entry = versions_data[v1_key]
        except Exception:
            return None
    v1_content = v1_entry.get("content", "") if isinstance(v1_entry, dict) else ""
    if not v1_content:
        return None

    old_v = first_diff.get("old_version", {})
    old_lr = old_v.get("line_range") if isinstance(old_v, dict) else None
    old_content = old_v.get("content") if isinstance(old_v, dict) else None

    snippet = None
    if old_content and old_lr and len(old_lr) >= 2:
        lines = old_content.split("\n")
        if 0 <= old_lr[0] - 1 < len(lines) and old_lr[1] <= len(lines):
            snippet = "\n".join(lines[old_lr[0] - 1 : old_lr[1]]).strip()

    if not snippet:
        return None

    v1_lines = v1_content.split("\n")
    snippet_lines = snippet.splitlines()
    n = len(snippet_lines)
    if n == 0 or len(v1_lines) < n * MIN_LINES_OVERLAP_RATIO:
        return None

    snippet_stripped = [l.strip() for l in snippet_lines]
    best_i, best_ratio = None, 0.0
    for i in range(len(v1_lines) - n + 1):
        window = [l.strip() for l in v1_lines[i:i + n]]
        matches = sum(w == nw for w, nw in zip(window, snippet_stripped))
        ratio = matches / n
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i

    if best_i is not None and best_ratio >= MIN_LINES_OVERLAP_RATIO:
        return [best_i + 1, best_i + n]
    return None


def augment_chains_with_new_sentinel(multi_chains, versions_data, versions_sorted_ids):
    """
    For chains whose first diff is NOT categorized as "New", prepend a
    synthetic sentinel diff representing its initial appearance in version 1.
    Chains sharing the same v1 origin or first diff share the same sentinel object.
    """
    v1_id = versions_sorted_ids[0] if versions_sorted_ids else "1"
    v1_entry = None
    if versions_data:
        v1_entry = versions_data.get(str(v1_id)) or versions_data.get(v1_id)
        if v1_entry is None:
            try:
                v1_key = min(versions_data.keys(), key=get_vid_sort_key)
                v1_entry = versions_data[v1_key]
            except Exception:
                v1_entry = None

    v1_content = v1_entry.get("content") if isinstance(v1_entry, dict) else None
    v1_commit_hash = v1_entry.get("commit_hash") if isinstance(v1_entry, dict) else None
    v1_date = v1_entry.get("date") if isinstance(v1_entry, dict) else None

    sentinel_cache = {}
    augmented = []
    for chain in multi_chains:
        if not chain:
            continue
        first_diff = chain[0]
        raw_reason = extract_reason_type(first_diff)
        first_cat = CATEGORY_MAPPING.get(raw_reason.lower(), raw_reason)
        if first_cat == "New":
            augmented.append(chain)
        else:
            v1_range = find_range_in_v1(first_diff, versions_data)
            if v1_range:
                sentinel_key = f"sentinel_v1_{v1_range[0]}_{v1_range[1]}"
            else:
                first_did = first_diff.get("diff_id")
                if first_did:
                    sentinel_key = f"sentinel_diff_{first_did}"
                else:
                    sentinel_key = f"sentinel_obj_{id(first_diff)}"

            if sentinel_key not in sentinel_cache:
                sentinel = {
                    "_synthetic": True,
                    "_sentinel_key": sentinel_key,
                    "reason_type": "New",
                    "diff_id": sentinel_key,
                    "new_version": {
                        "version_id": v1_id,
                        "line_range": v1_range,
                        "content": v1_content,
                        "commit_hash": v1_commit_hash,
                        "date": v1_date,
                    },
                }
                sentinel_cache[sentinel_key] = sentinel
            else:
                sentinel = sentinel_cache[sentinel_key]

            augmented.append([sentinel] + chain)
    return augmented


# ---------------------------------------------------------------------------
# Chain Tree Representation
# ---------------------------------------------------------------------------

def chains_to_trees(chains):
    """
    Convert a list of flat chains into a list of tree root nodes.

    Chains sharing a common ancestry (same diff Python object or sentinel appearing
    in multiple chains due to branching) are merged into a single tree with
    branching nodes representing requirement evolution splits.

    Each node is: {"diff": diff_dict, "children": [node, ...]}

    Returns a list of root nodes (nodes not reachable as a child of any other).
    """
    if not chains:
        return []

    node_registry = {}    # key -> node dict
    child_keys = set()    # keys that are children of at least one other node
    key_cache = {}        # id(diff) -> key  (stable per Python object)
    sentinel_counter = [0]

    def _get_key(diff):
        obj_id = id(diff)
        if obj_id not in key_cache:
            did = diff.get("diff_id", "")
            if did == "" or did is None:
                if diff.get("_synthetic") and diff.get("_sentinel_key"):
                    key_cache[obj_id] = str(diff["_sentinel_key"])
                else:
                    sentinel_counter[0] += 1
                    key_cache[obj_id] = f"__sentinel_{sentinel_counter[0]}__"
            else:
                key_cache[obj_id] = str(did)
        return key_cache[obj_id]

    for chain in chains:
        parent_key = None
        for diff in chain:
            key = _get_key(diff)
            if key not in node_registry:
                node_registry[key] = {"diff": diff, "children": []}
            node = node_registry[key]
            if parent_key is not None:
                parent_node = node_registry[parent_key]
                if node not in parent_node["children"]:
                    parent_node["children"].append(node)
                child_keys.add(key)
            parent_key = key

    roots = [n for k, n in node_registry.items() if k not in child_keys]
    return roots


def format_chain_tree(roots):
    """
    Return a horizontal Unicode-art tree of chain roots.

    The common prefix of split chains is rendered once. The first child of
    any split node continues on the same line (solid ─── connector). Additional
    children fan out below, connected by dashed lines (╌╌╌). A vertical bar │
    is drawn between branch rows to keep later siblings visually tied to the
    split node.

    Example::

        [New] ─── #3(v1→v2) ─── #7(v2→v3) ─── #14(v3→v4)
                           └╌╌╌╌ #9(v2→v5)
    """
    if not roots:
        return ""

    canvas = {}  # (row, col) -> single character

    def _paint(row, col, text):
        for i, ch in enumerate(text):
            canvas[(row, col + i)] = ch

    def _label(diff):
        if diff.get("_synthetic"):
            return "[New]"
        did = diff.get("diff_id", "?")
        ov  = get_old_vid(diff)
        nv  = get_new_vid(diff)
        return f"#{did}(v{ov}→v{nv})"

    SOLID  = " ─── "        # solid connector (first child, same row)
    CONN_W = len(SOLID)     # 5 characters

    def _render(node, row, col):
        """Paint node and its whole subtree at (row, col). Returns max row used."""
        label = _label(node["diff"])
        _paint(row, col, label)
        children = node["children"]
        if not children:
            return row

        next_col = col + len(label)

        # First child: solid connector, same row
        _paint(row, next_col, SOLID)
        max_row = _render(children[0], row, next_col + CONN_W)

        # Remaining children: dashed branch connectors on successive rows
        branch_col = next_col
        branch_row = max_row + 1
        for i, child in enumerate(children[1:]):
            is_last = (i == len(children) - 2)
            connector = ("└" if is_last else "├") + "╌" * (CONN_W - 1)
            _paint(branch_row, branch_col, connector)
            new_max = _render(child, branch_row, branch_col + CONN_W)
            # Paint │ on rows between this branch and the next sibling
            # so the visual connection back to the split point is clear
            if not is_last:
                for r in range(branch_row + 1, new_max + 1):
                    if canvas.get((r, branch_col), " ") == " ":
                        _paint(r, branch_col, "│")
            branch_row = new_max + 1

        return branch_row - 1

    current_row = 0
    for root in roots:
        max_r = _render(root, current_row, 0)
        current_row = max_r + 2  # blank line between separate root trees

    if not canvas:
        return ""

    max_r = max(r for r, _ in canvas)
    max_c = max(c for _, c in canvas)
    lines = []
    for r in range(max_r + 1):
        row_chars = [canvas.get((r, c), " ") for c in range(max_c + 1)]
        lines.append("".join(row_chars).rstrip())
    return "\n".join(lines)

