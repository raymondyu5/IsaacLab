import os


def generate_model_viewer_html(root_folder, output_html="index.html"):
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>3D Mesh Viewer</title>",
        "  <script type='module' src='https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js'></script>",
        "  <style>",
        "    body { font-family: sans-serif; padding: 2em; background: #222; color: white; }",
        "    h2 { margin-top: 2em; }",
        "    .gallery { display: flex; flex-wrap: wrap; gap: 20px; }",
        "    model-viewer { width: 300px; height: 300px; background-color: #111; border: 1px solid #444; border-radius: 8px; cursor: pointer; }",
        "    #overlay {",
        "      position: fixed; top: 0; left: 0; width: 100%; height: 100%;",
        "      background: rgba(0,0,0,0.9); display: none; align-items: center; justify-content: center;",
        "      z-index: 9999;",
        "    }",
        "    #overlay model-viewer { width: 90vw; height: 90vh; }",
        "    #overlay-close { position: absolute; top: 20px; right: 30px; font-size: 28px; color: white; cursor: pointer; }",
        "  </style>",
        "</head>",
        "<body>",
        "<h1>3D Mesh Viewer</h1>",

        # Modal overlay
        "<div id='overlay'>",
        "  <div id='overlay-close' onclick='closeOverlay()'>&times;</div>",
        "  <model-viewer id='fullscreen-viewer' camera-controls auto-rotate background-color='#111' shadow-intensity='0.3' exposure='0.6'></model-viewer>",
        "</div>"
    ]
    num_objects = 0
    for category in sorted(os.listdir(root_folder)):
        cat_path = os.path.join(root_folder, category)

        if os.path.isdir(cat_path):

            html_lines.append(f"<h2>{category}</h2>")
            html_lines.append("<div class='gallery'>")
            for file in sorted(os.listdir(cat_path)):
                if file.endswith(".glb"):
                    num_objects += 1
                    rel_path = f"{category}/{file}"
                    html_lines.append(f"""
  <model-viewer 
    src="{rel_path}" 
    alt="{file}" 
    auto-rotate 
    camera-controls 
    exposure="0.6" 
    shadow-intensity="0.3" 
    background-color="#111"
    onclick="expandModel('{rel_path}')">
  </model-viewer>
""")
            html_lines.append("</div>")

    # JavaScript for modal
    html_lines.append("""
<script>
  function expandModel(src) {
    const viewer = document.getElementById('fullscreen-viewer');
    viewer.src = src;
    document.getElementById('overlay').style.display = 'flex';
  }
  function closeOverlay() {
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('fullscreen-viewer').src = '';
  }
</script>
""")

    html_lines.append("</body>")
    html_lines.append("</html>")

    output_path = os.path.join(root_folder, output_html)
    with open(output_path, "w") as f:
        f.write("\n".join(html_lines))

    print(f"✅ HTML saved to: {output_path}")
    print(num_objects)


# Example usage
generate_model_viewer_html(
    "/home/ensu/Documents/weird/IsaacLab/logs/dexycb_right/hand_object_mesh")
