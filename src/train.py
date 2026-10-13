# __________________model training_______


from google.colab import drive
drive.mount('/content/drive')

model.train(data="/content/bdd/bdd.yaml",
            epochs=5,
            batch=16,
            imgsz=640,
            device=device,
            workers=2,
            project="/content/drive/MyDrive/runs/bdd_runs",
            fraction=0.5
            )

