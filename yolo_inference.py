# .\venv\Scripts\activate
from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('input_videos/match.mp4', save = True)
print("=========================================================================")
print(results[0])
for box in results.boxes:
    print(box)
    