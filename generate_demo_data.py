"""Create SYNTHETIC demonstration data (not real observations)."""
import numpy as np
import pandas as pd
from area_predictor.config import ROOT

rng = np.random.default_rng(42)
n = 3000
regions = np.array(["North", "South", "East", "West", "Central"])
covers = np.array(["forest", "grassland", "cropland", "shrubland", "wetland"])
reg_eff = dict(zip(regions, [0.3, -0.2, 0.1, 0.4, 0.0]))
cov_eff = dict(zip(covers, [0.6, 0.2, -0.3, 0.4, -0.5]))
df = pd.DataFrame({
    "event_id": rng.integers(0, 1500, n),
    "event_date": pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 3650, n), unit="D"),
    "latitude": rng.uniform(8, 35, n), "longitude": rng.uniform(68, 97, n),
    "elevation_m": rng.gamma(2, 300, n), "rainfall_mm": rng.gamma(4, 40, n),
    "temperature_c": rng.normal(26, 6, n), "population_density": rng.lognormal(4, 1.5, n),
    "prior_area_ha": rng.lognormal(3, 1, n),
    "region": rng.choice(regions, n), "land_cover": rng.choice(covers, n)})
la = (2.0 + 0.06 * (df.temperature_c - 26) - 0.003 * (df.rainfall_mm - 160) - 0.15 * np.log1p(df.population_density)
      + 0.5 * np.log1p(df.prior_area_ha) + df.region.map(reg_eff) + df.land_cover.map(cov_eff) + rng.normal(0, 0.4, n))
df["area_ha"] = np.exp(la).round(2)
df["burned_fraction_post"] = (df.area_ha / (df.area_ha + 50)).round(4)  # LEAKAGE column on purpose (known only after the event)
# inject realistic dirt to exercise the cleaning code
for c in ("rainfall_mm", "temperature_c", "elevation_m"):
    df.loc[rng.choice(n, 60, replace=False), c] = np.nan
df["rainfall_mm"] = df["rainfall_mm"].astype(object)
df.loc[5, "rainfall_mm"] = "n/a"
df.loc[rng.choice(n, 5, replace=False), "area_ha"] = -1
df.loc[rng.choice(n, 5, replace=False), "latitude"] = 999
df.loc[rng.choice(n, 40, replace=False), "region"] = " North "
df = pd.concat([df, df.sample(30, random_state=1)], ignore_index=True)
out = ROOT / "data" / "demo_area_data.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df)} SYNTHETIC rows to {out}")
