from google.colab import drive
from ultralytics import YOLO
from PIL import Image
import gradio as gr
import torch
drive.mount('/content/drive')
model=YOLO("/content/drive/MyDrive/runs/bdd_runs/train/weights/best.pt")


if torch.cuda.is_available():
    print("device is gpu")
    device="cuda"
else:
    print("device is cpu ")
    device="cpu"



def detect(img):
    if img is None:
        raise gr.Error("first upload a picture!")

    r = model.predict(img.convert("RGB"), device=device, conf=0.5, verbose=False)[0]
    plotted = r.plot()                      # آرایه‌ی numpy با رنگ BGR
    return Image.fromarray(plotted[..., ::-1])   # تبدیل به RGB

    
    
            
demo=gr.Interface(fn=detect,
                  inputs=gr.Image(type="pil",label="picture"),
                  outputs=gr.Image(label="predicting"),
                  title="predicting",
                  description="upload a picture",
                  flagging_mode="never")
                          
                  
            
demo.launch(share=True,debug=True)




