from joblib import load
from pathlib import Path

bundle_path = Path("models/hierarchical.joblib")
obj = load(bundle_path)

print(f"Type: {type(obj).__name__}")
if isinstance(obj, dict):
    print(f"Keys: {list(obj.keys())}")
    print(f"Stage-1 classes: {obj['stage1'].classes_}")
    print(f"Stage-2 heads present: {list(obj['stage2'].keys())}")
    for ci, head in obj['stage2'].items():
        print(f"  head[{ci}].classes_ = {head.classes_}")
else:
    print(f"Flat pipeline. classes_ = {obj.classes_}")