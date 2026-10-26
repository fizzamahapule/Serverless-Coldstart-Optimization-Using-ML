from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models import train_all
if __name__ == "__main__":
    metrics, chosen = train_all()
    print("Validation-selected model:", chosen)
    for name, values in metrics.items():
        print("\\n", name)
        for k,v in values.items(): print(f"  {k}: {v}")
