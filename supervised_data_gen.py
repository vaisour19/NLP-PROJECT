import os
import json
import pandas as pd
import random

# --- Configuration ---
Matterport_folder = "Matterport/Matterport"  # path to your Matterport folder
instructions_per_relation = 5  # generate multiple variations per referential statement
output_folder = "vln_instructions"  # folder to save CSVs

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# --- Template variations ---
templates = [
    "Go to the {target_color} {target} that is {relation} the {anchor_color} {anchor}.",
    "Move toward the {target} which is {relation} the {anchor}.",
    "Head near the {target} {relation} the {anchor}.",
    "Walk to the {target_color} {target} positioned {relation} the {anchor_color} {anchor}.",
]

# --- Helper function ---
def generate_sentence(target, anchor, relation, target_color, anchor_color):
    template = random.choice(templates)
    sentence = template.format(
        target=target,
        anchor=anchor,
        relation=relation,
        target_color=random.choice(target_color) if target_color else "",
        anchor_color=random.choice(anchor_color) if anchor_color else ""
    )
    return " ".join(sentence.split())  # clean double spaces

# --- Process all scenes ---
for scene_id in os.listdir(Matterport_folder):
    scene_path = os.path.join(Matterport_folder, scene_id)
    if not os.path.isdir(scene_path):
        continue

    json_file = os.path.join(scene_path, f"{scene_id}_referential_statements.json")
    if not os.path.isfile(json_file):
        print(f"Skipping {scene_id}: referential_statements.json not found")
        continue

    # Load JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    rows = []

    # Parse regions
    regions = data.get("regions", {})
    for region_id, phrases in regions.items():
        for phrase, annotations in phrases.items():
            if not isinstance(annotations, list):
                continue
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue

                target = ann.get("target_class", "")
                relation = ann.get("relation", "")
                target_pos = ann.get("target_position", [0,0,0])
                
                anchor_info = ann.get("anchors", {}).get("anchor_1", {})
                anchor = anchor_info.get("class", "")
                anchor_pos = anchor_info.get("position", [0,0,0])

                target_color = [c for c in ann.get("target_colors", []) if c != "N/A"]
                anchor_color = [c for c in anchor_info.get("color", []) if c != "N/A"]

                # Generate multiple instructions per relation
                for _ in range(instructions_per_relation):
                    sentence = generate_sentence(target, anchor, relation, target_color, anchor_color)
                    rows.append({
                        "scene_id": scene_id,
                        "instruction": sentence,
                        "target_position_x": target_pos[0],
                        "target_position_y": target_pos[1],
                        "target_position_z": target_pos[2],
                        "anchor_position_x": anchor_pos[0],
                        "anchor_position_y": anchor_pos[1],
                        "anchor_position_z": anchor_pos[2],
                        "relation": relation,
                        "target_class": target,
                        "anchor_class": anchor
                    })

    # Save CSV per scene
    if rows:
        df = pd.DataFrame(rows)
        output_csv = os.path.join(output_folder, f"{scene_id}_instructions.csv")
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} instructions for scene {scene_id} → {output_csv}")
    else:
        print(f"No valid instructions found for scene {scene_id}")
