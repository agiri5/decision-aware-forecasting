# Data

No raw data is committed. Raw files live outside the repo; point to them with
`DAF_RAW_DATA_DIR` (default: `../Data` relative to the repo root).

| Dataset | Source | Expected path under `$DAF_RAW_DATA_DIR` | License / terms |
|---|---|---|---|
| M5 (Walmart) | Kaggle competition `m5-forecasting-accuracy` | `m5-forecasting-accuracy/` | TODO: verify competition rules permit public derived results |
| Favorita | Kaggle competition `favorita-grocery-sales-forecasting` | `favorita-grocery-sales-forecasting/` | TODO: verify |
| Auto-parts (Hyndman) | R package `expsmooth`, dataset `carparts` | `carparts/` | TODO: verify (GPL package) |

Download (requires a Kaggle account that has accepted each competition's rules):

```bash
kaggle competitions download -c m5-forecasting-accuracy -p "$DAF_RAW_DATA_DIR/m5-forecasting-accuracy"
kaggle competitions download -c favorita-grocery-sales-forecasting -p "$DAF_RAW_DATA_DIR/favorita-grocery-sales-forecasting"
```

Known limitation: observed sales are censored demand (a zero may be a stockout).
