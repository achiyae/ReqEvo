import os
import webbrowser
from jinja2 import Template

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
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .diff-text { font-family: 'Consolas', 'Courier New', monospace; white-space: pre-wrap; background-color: #f8f9fa; padding: 10px; border-radius: 4px; border: 1px solid #e1e4e8; }
        .reason-type { font-weight: bold; color: #2980b9; }
        .old-version { color: #c0392b; text-decoration: line-through; }
        .new-version { color: #27ae60; }
        .meta { margin-bottom: 20px; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Requirement Evolution: {{ domain }}</h1>
        <div class="meta">
            <p><strong>Number of Versions:</strong> {{ num_versions }}</p>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Diff ID</th>
                    <th>Change Details</th>
                    <th>Analysis</th>
                </tr>
            </thead>
            <tbody>
                {% for diff in diffs %}
                <tr>
                    <td>{{ diff.diff_id }}</td>
                    <td>
                        <div><strong>From Version:</strong> {{ diff.old_version_id }}</div>
                        <div><strong>To Version:</strong> {{ diff.new_version_id }}</div>
                        <div class="diff-text">{{ diff.diff_text }}</div>
                    </td>
                    <td>
                        <div class="reason-type">{{ diff.reason_type }}</div>
                        <p>{{ diff.reason_text }}</p>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

def render_html_report(domain: str, num_versions: int, diffs: list, output_path: str = "report.html"):
    template = Template(HTML_TEMPLATE)
    html_content = template.render(domain=domain, num_versions=num_versions, diffs=diffs)
    
    # Use absolute path for robustness
    abs_path = os.path.abspath(output_path)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return abs_path

def open_in_browser(file_path: str):
    webbrowser.open(f"file://{file_path}")
