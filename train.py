import argparse
from area_predictor.train import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the area prediction model")
    ap.add_argument("--config", default=None, help="path to config.json")
    try:
        run(ap.parse_args().config)
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise SystemExit(f"ERROR: {e}")
