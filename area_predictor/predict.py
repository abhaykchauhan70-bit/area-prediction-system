def load_bundle(config_path=None):
    cfg = load_config(config_path)
    # FIX: Direct absolute path for Render
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "model.pkl")
    
    from pathlib import Path
    path = Path(model_path)
    
    if not path.exists():
        # Fallback to config path
        path = resolve(cfg, "model_path")
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run 'python train.py' first. Tried also {model_path}")
    
    bundle = joblib.load(path)
    # Ensure bundle has config
    if "config" not in bundle:
        bundle["config"] = cfg
    return bundle
