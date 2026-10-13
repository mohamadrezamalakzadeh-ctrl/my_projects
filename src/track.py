
model.track("/content/drive/MyDrive/cctv052x2004080517x01652.avi",
           conf=0.5,
           tracker="bytetrack.yaml",
           save=True )

model.export(format="onnx",imgsz=640,simplify=True)
