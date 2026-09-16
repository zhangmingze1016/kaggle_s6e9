from pathlib import Path
import pandas as pd

class PredictionSaver:

    def __init__(self, output_dir = "predictions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, ids, predictions, score, target = "Will_Buy_EV"):
        existing = list(
            self.output_dir.glob("prediction_v*.csv")
        )

        version = len(existing) + 1

        predictions_df = pd.DataFrame({"id" : ids, target : predictions})

        filename = (
            self.output_dir 
            / f"prediction_v{version:03d}_auc_{score: 5f}.csv")

        predictions_df.to_csv(filename, index = False)

        print(f"Saved prediction: {filename}")

        return filename
            