import os
import torch
import json
import kagglehub
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
from ultralytics import YOLO
import yaml
model = YOLO("yolov8n.pt")


#___________________downloading datast and converting to yolo dataset



if torch.cuda.is_available():
    print("device is gpu")
    device="cuda"
else:
    print("device is cpu ")
    device="cpu"

W=1280
H=720
class_name=['car', 'truck', 'bus', 'motor', 'bike', 'person', 'rider', 'train']
class_dic={r:i for i,r in enumerate(class_name)}


def convert_yolo(json_file,txt_path):
    os.makedirs(txt_path,exist_ok=True)

    with open(json_file) as f:
        data=json.load(f)

        for item in data:
            lines=[]

            for lab in item.get("labels",[]):
                if "box2d" not in lab or lab["category"] not in class_dic:
                    continue

                box=lab["box2d"]
                x1,y1,x2,y2=box["x1"],box["y1"],box["x2"],box["y2"]

                xc=((x1+x2)/2)/W
                yc=((y1+y2)/2)/H
                w=(x2-x1)/W
                h=(y2-y1)/H

                xc,yc,w,h=[min(max(v,0.0),1.0 )for v in (xc,yc,w,h)]
                lines.append(f"{class_dic[lab['category']]} {xc} {yc} {w} {h}")

            text_name=os.path.splitext(item['name'])[0] +".txt"

            with open(os.path.join(txt_path,text_name),"w") as f:
                f.write("\n".join(lines))








# Download latest version
path = kagglehub.dataset_download("marquis03/bdd100k")

dest="/content/bdd"
sorce=path

assert os.path.exists(f"{sorce}/train/images")

shutil.rmtree(dest,ignore_errors=True)
for split in ["train","val"]:
  image_sorce=f"{sorce}/{split}/images"
  image_dest=f"{dest}/{split}/images"

  os.makedirs(image_dest,exist_ok=True)
  for f in os.listdir(image_sorce):
    os.symlink(os.path.join(image_sorce,f),os.path.join(image_dest,f))

convert_yolo(f"{path}/train/annotations/bdd100k_labels_images_train.json",
             f"{dest}/train/labels")


convert_yolo(f"{path}/val/annotations/bdd100k_labels_images_val.json",
             f"{dest}/val/labels")

#______________________ckeking converting______________
def cheke(split):
  image_dir=f"{dest}/{split}/images"
  label_dir=f"{dest}/{split}/labels"

  image={os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith((".png",".jpeg",".jpg"))}
  label={os.path.splitext(v)[0] for v in os.listdir(label_dir) if v.lower().endswith(".txt")}

  if len(image)!=len(label):
    print("len iimage , label is not same",len(image),len(label))
  else:
    print("converting done")

cheke("train")
cheke("val")

cfg = {
    "path": "/content/bdd",
    "train": "train/images",
    "val": "val/images",
    "nc": len(class_name),
    "names": {i: n for i, n in enumerate(class_name)},
}

with open("/content/bdd/bdd.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)


