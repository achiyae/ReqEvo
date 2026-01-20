from main import build_graph

print("Generating workflow visualization...")
try:
    app = build_graph()
    png_bytes = app.get_graph().draw_mermaid_png()
    
    output_file = "workflow.png"
    with open(output_file, "wb") as f:
        f.write(png_bytes)
        
    print(f"Successfully saved workflow to {output_file}")
    
except Exception as e:
    print(f"Error generating visualization: {e}")
    print("Ensure you have internet access (for mermaid API) or graphviz installed if using low-level drawing.")
