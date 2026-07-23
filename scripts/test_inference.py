from pathlib import Path
from app.inference import PoultryPredictor

predictor = PoultryPredictor()

result = predictor.predict(
    Path(
        "data/raw/Poultry Birds Poo Imagery Dataset for Health Status Prediction A Case of South-West Nigeria/Dataset/Healthy/IMG_1417.JPG"
    )
)

print(result)
