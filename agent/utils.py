import os
import webbrowser
from jinja2 import Template

REASON_DEFINITIONS = {
    "Contradiction": "The change was introduced to fix a contradiction between two requirements of the same document.",
    "Mistake": "The change fixes a mistake (logically wrong statement) in the requirement.",
    "Typo": "The change was made to correct a typo (wrong spelling/grammar only).",
    "Generalization": "The requirement was changed to be more general (old was too specific).",
    "Clarification": "Rephrasing the same requirement to help understanding WITHOUT changing the meaning/intent.",
    "Meaning": "The requirement meaning/intention is changed (says something different).",
    "Summarization/shortening": "The change is made to summarize/shorten without changing meaning.",
    "Deletion": "An entire REQUIREMENT was redundant and therefore removed. Do NOT use this for removal of metadata, placeholders, or formatting.",
    "Demonstration": "An example or visualization was added to assist in understanding.",
    "New": "A new requirement was added.",
    "Other": "The change does not fit into any of the above categories (e.g. metadata removal, formatting changes, boilerplate)."
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Requirement Evolution Analysis - {{ domain }}</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }
        h1 { color: #2c3e50; }
        .container { max-width: 95%; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .diff-text { font-family: 'Consolas', 'Courier New', monospace; white-space: pre-wrap; background-color: #f8f9fa; padding: 5px; border-radius: 4px; border: 1px solid #e1e4e8; overflow-x: auto; }
        .reason-type { font-weight: bold; color: #2980b9; position: relative; display: inline-block; cursor: help; border-bottom: 1px dotted #2980b9; }
        .old-version { color: #c0392b; text-decoration: line-through; }
        .new-version { color: #27ae60; }
        .meta { margin-bottom: 20px; color: #7f8c8d; }
        
        /* Tooltip container */
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 300px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.9em;
            font-weight: normal;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            line-height: 1.4;
        }

        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        /* Git Style Diff */
        .diff-line { display: block; min-height: 1.2em; }
        .diff-added { background-color: #e6ffed; color: #22863a; }
        .diff-removed { background-color: #ffeef0; color: #b31d28; }
        .diff-header { color: #6f7781; font-weight: bold; }
        
        /* Feedback Form */
        .feedback-cell { display: flex; flex-direction: column; gap: 10px; }
        .feedback-area { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-family: inherit; }
        .reason-select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; }
        .submit-btn { display: block; width: 100%; padding: 15px; background-color: #2ecc71; color: white; border: none; border-radius: 4px; font-size: 1.2em; cursor: pointer; margin-top: 20px; transition: all 0.3s ease; }
        .submit-btn:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.2); opacity: 0.9; }
        label { font-weight: bold; font-size: 0.9em; color: #555; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Requirement Evolution: {{ domain }}</h1>
        <div class="meta">
            <p><strong>Number of Versions:</strong> {{ num_versions }}</p>
        </div>
        
        <form id="feedbackForm">
            <table>
                <colgroup>
                    <col style="width: 50px;">
                    <col style="width: {% if final_mode %}60%{% else %}45%{% endif %};">
                    <col style="width: {% if final_mode %}35%{% else %}25%{% endif %};">
                    {% if not final_mode %}<col style="width: 25%;">{% endif %}
                </colgroup>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Change Details</th>
                        <th>AI Analysis</th>
                        {% if not final_mode %}<th>Your Feedback</th>{% endif %}
                    </tr>
                </thead>
                <tbody>
                    {% for diff in diffs %}
                    <tr>
                        <td>{{ diff.diff_id }}</td>
                        <td>
                            <div><strong>From:</strong> {{ diff.old_version_id }} &rarr; <strong>To:</strong> {{ diff.new_version_id }}</div>
                            {% if diff.old_commit_hash %}
                            <div style="font-size: 0.8em; color: #666; margin-bottom: 5px;">
                                {{ diff.old_commit_hash[:7] }} ({{ diff.old_date }}) &rarr; {{ diff.new_commit_hash[:7] }} ({{ diff.new_date }})
                            </div>
                            {% endif %}
                            <div class="diff-text">{{ diff.html_diff | safe }}</div>
                        </td>
                        <td>
                            <div class="reason-type tooltip">{{ diff.reason_type }}
                                <span class="tooltiptext">{{ definitions[diff.reason_type] }}</span>
                            </div>
                            <p>{{ diff.reason_text }}</p>
                        </td>
                        {% if not final_mode %}
                        <td>
                            <div class="feedback-cell">
                                <div>
                                    <label for="reason_{{ diff.diff_id }}">Correct Reason:</label>
                                    <select id="reason_{{ diff.diff_id }}" name="reason_{{ diff.diff_id }}" class="reason-select">
                                        <option value="">(Confirm or Select correction)</option>
                                        {% for r_type in reason_types %}
                                        <option value="{{ r_type }}">{{ r_type }}</option>
                                        {% endfor %}
                                    </select>
                                </div>
                                
                                <div>
                                    <label for="explanation_{{ diff.diff_id }}">Explanation / Comment:</label>
                                    <textarea id="explanation_{{ diff.diff_id }}" name="explanation_{{ diff.diff_id }}" class="feedback-area" rows="3" placeholder="Explain why the analysis is wrong..."></textarea>
                                </div>
                            </div>
                        </td>
                        {% endif %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            {% if not final_mode %}
            <div style="margin-top: 20px; display: flex; gap: 10px;">
                <button type="button" class="submit-btn" onclick="submitFeedback('retry')" style="flex: 1; background-color: rgb(230, 230, 10); color: #333; font-weight: bold;">Submit Corrections & Re-Analyze</button>
                <button type="button" class="submit-btn" onclick="submitFeedback('finish')" style="flex: 1; background-color: #27ae60;">Finish Analysis (Save Final Report)</button>
            </div>
            {% endif %}
        </form>
    </div>

    {% if not final_mode %}
    <script>
        function submitFeedback(actionType) {
            const formData = {};
            const inputs = document.querySelectorAll('input, select, textarea');
            let hasFeedback = false;
            
            inputs.forEach(input => {
                const val = input.value.trim();
                if (val) {
                    formData[input.name] = val;
                    hasFeedback = true;
                }
            });
            
            formData['action'] = actionType;
            
            if (actionType === 'retry' && !hasFeedback) {
                if(!confirm("No feedback entered. Do you want to approve the current analysis?")) {
                    return;
                }
                formData['action'] = 'approve';
            }

            fetch('http://localhost:8000/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                alert("Action submitted! You can close this tab/window now.");
                window.close();
            })
            .catch((error) => {
                console.error('Error:', error);
                alert("Submitted! (Note: Server might have closed if it received the response)");
            });
        }
    </script>
    {% endif %}
</body>
</html>
"""

def render_html_report(domain: str, num_versions: int, diffs: list, output_path: str = "report.html", reason_types: list = None, is_final: bool = False):
    # Pre-process diffs to add html_diff field
    for diff in diffs:
        raw_diff = diff['diff_text']
        html_lines = []
        for line in raw_diff.splitlines():
            if line.startswith('@@'):
                html_lines.append(f'<span class="diff-line diff-header">{line}</span>')
            elif line.startswith('+'):
                html_lines.append(f'<span class="diff-line diff-added">{line}</span>')
            elif line.startswith('-'):
                html_lines.append(f'<span class="diff-line diff-removed">{line}</span>')
            else:
                html_lines.append(f'<span class="diff-line">{line}</span>')
        diff['html_diff'] = "\n".join(html_lines)
    
    if reason_types is None:
        reason_types = list(REASON_DEFINITIONS.keys())

    template = Template(HTML_TEMPLATE)
    html_content = template.render(
        domain=domain, 
        num_versions=num_versions, 
        diffs=diffs, 
        reason_types=reason_types,
        definitions=REASON_DEFINITIONS,
        final_mode=is_final
    )
    
    # Use absolute path for robustness
    abs_path = os.path.abspath(output_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return abs_path

def open_in_browser(file_path: str):
    webbrowser.open(f"file://{file_path}")
