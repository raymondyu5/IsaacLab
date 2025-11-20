import xml.etree.ElementTree as ET

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=[
        "background_texture.zip", "embodiments.zip", "objects.zip"
    ],
    local_dir="/home/ensu/Documents/weird/IsaacLab_assets/assets/cabinet/trash",
    repo_type="dataset",
    resume_download=True,
)

# path = "tools/trash/amazon/urdf/drawer_long/drawer01.urdf"
# tree = ET.parse(path)
# root = tree.getroot()

# for link in root.findall("link"):
#     collisions = link.findall("collision")
#     for col in collisions:
#         vis = ET.Element("visual")
#         for child in list(col):
#             vis.append(child)
#         link.insert(0, vis)  # insert visual before collision

# tree.write("tools/trash/amazon/urdf/drawer_long/drawer01_with_visual.urdf",
#            encoding="utf-8",
#            xml_declaration=True)
# print("✅ Saved cabinet_with_visual.urdf with visuals added.")
