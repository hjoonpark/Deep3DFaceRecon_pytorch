import os, json
from zipfile import ZipFile
from pathlib import Path

output_path = "c:/users/jseo/Downloads/slack_F03QDHN798S"
template_path = "head_template.usda"
obj_path = os.path.join(output_path, "model.obj")

for tex in ["tex1k_102522_stage_3.png", "tex256.jpg"]:
    head_tex_path = os.path.join(output_path, tex)
    if os.path.exists(head_tex_path):
        break

eyes_info_path = os.path.join(output_path, "eyes_info.json")
if os.path.exists(eyes_info_path):
    eyes_info = json.load(open(eyes_info_path))
    eye_info_l, eye_info_r = eyes_info["left"], eyes_info["right"]
else:
    eye_info_l = {'color': 'Brown', 'xform':{'t':[.0,.0,.0], 's':0.1}}
    eye_info_r = eye_info_l

eye_color_l = eye_info_l["color"]
if eye_color_l not in ["Brown", "Grey", "Green", "Blue"]:
    eye_color_l = "Brown"
eye_tex_path = f"eye_{eye_color_l.lower()}.jpg"

template = open(template_path).read()
obj_lines = open(obj_path).readlines()
pnts = [(float(y) for y in x.split()[1:]) for x in obj_lines if x.startswith("v ")]

# manual offset to roughly match computed world xform to template rig
for ef in (eye_info_l, eye_info_r):
    ef["xform"]["t"][2] -= 0.04

head_usda = template.replace("$HEAD_POINTS$", str([tuple(p) for p in pnts]))
head_usda = head_usda.replace("$HEAD_TEX_PATH$", Path(head_tex_path).name)
head_usda = head_usda.replace("$EYE_L_TEX_PATH$", eye_tex_path)
head_usda = head_usda.replace("$EYE_L_TRANS$", str(tuple(eye_info_l["xform"]["t"])))
head_usda = head_usda.replace("$EYE_R_TRANS$", str(tuple(eye_info_r["xform"]["t"])))
open("head_out.usda", "w").write(head_usda)

with ZipFile("head.usdz", "w") as z:
    z.write("head_out.usda")
    z.write(head_tex_path, Path(head_tex_path).name)
    z.write(eye_tex_path)
