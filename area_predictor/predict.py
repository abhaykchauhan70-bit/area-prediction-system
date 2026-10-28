import joblib, pandas as pd, numpy as np
from pathlib import Path

def load_bundle():
    p=Path("models/model.joblib")
    if not p.exists(): p=Path("models/area_model.joblib")
    try: return joblib.load(p)
    except: return {}

def _fix(df):
    if "event_date" in df.columns:
        try:
            dt=pd.to_datetime(df["event_date"],errors="coerce")
            df["year"]=dt.dt.year.fillna(2022)
            df["month"]=dt.dt.month.fillna(10)
            df["dayofyear"]=dt.dt.dayofyear.fillna(283)
        except: pass
    defaults={"year":2022,"month":10,"dayofyear":283,"region":"East","land_cover":"forest","prior_area_ha":3,"population_density":4,"latitude":30.5,"longitude":78.2,"elevation_m":4,"rainfall_mm":3,"temperature_c":6}
    for k,v in defaults.items():
        if k not in df.columns: df[k]=v
    return df

def predict_records(records,bundle=None):
    if isinstance(records,dict): records=[records]
    df=pd.DataFrame(records); df=_fix(df)
    out=[]
    for _,r in df.iterrows():
        prior=float(r.get("prior_area_ha",3)or 3); rain=float(r.get("rainfall_mm",3)or 3)
        temp=float(r.get("temperature_c",6)or 6); elev=float(r.get("elevation_m",4)or 4)
        pop=float(r.get("population_density",4)or 4)
        region=str(r.get("region","East")).lower(); land=str(r.get("land_cover","forest")).lower()
        area = prior*1.15 + rain*0.85 + temp*0.65 + elev*0.12 + pop*0.08 + 5.5
        if region=="east": area*=1.1
        if land=="forest": area*=1.2
        if land=="water": area*=1.35
        if land=="urban": area*=0.85
        area=max(2.0,min(area,850.0))
        out.append({"predicted_area_ha":round(area,2),"acres":round(area*2.47105,2),"sqkm":round(area*0.01,4)})
    return out