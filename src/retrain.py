"""Reentrena un modelo retador y decide si reemplaza al vigente (punto 3.9).

Uso:
    python src/retrain.py --data data/orvyz_conversaciones_riesgo.csv \
        --champion models/riesgo_v1.joblib --holdout 2026-10 --out models/riesgo_v2.joblib

1. Entrena el retador con todo lo anterior al mes de prueba (`--holdout`).
2. Compara retador y vigente en ese mes, con las mismas empresas y conversaciones.
3. Si el retador gana en PR-AUC y no pierde más de 2 puntos de recall **alertando el mismo volumen
   de conversaciones que el vigente**, se reentrena con todo
   (incluido el mes de prueba) y se guarda en `--out`. El vigente no se toca: el cambio en
   producción se hace después de correr en paralelo, con opción de volver atrás.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import DOMINANT, TARGET, load  # noqa: E402
from monitoring import performance  # noqa: E402
from risk_model import train_risk_model  # noqa: E402

MAX_RECALL_LOSS = 0.02


def compare(champion, challenger, holdout):
    """Recall at each model's own threshold mixes ranking quality with where the cut falls, so the
    decision uses recall at the champion's alert volume (same workload for the team)."""
    rows = {}
    y = holdout[TARGET].values
    volume = (champion.score(holdout) >= champion.thresholds["alerta"]).mean()
    for name, m in [("vigente", champion), ("retador", challenger)]:
        s = m.score(holdout)
        rows[name] = performance(y, s, m.thresholds["alerta"])
        top = np.argsort(-s)[:int(round(volume * len(s)))]
        rows[name]["recall a igual volumen"] = y[top].sum() / y.sum()
    by_company = {name: {e: performance(g[TARGET], m.score(g), m.thresholds["alerta"])["ROC-AUC"]
                         for e, g in holdout.groupby("empresa")}
                  for name, m in [("vigente", champion), ("retador", challenger)]}
    return pd.DataFrame(rows).T, pd.DataFrame(by_company)


def should_promote(summary):
    c, v = summary.loc["retador"], summary.loc["vigente"]
    return bool(c["PR-AUC"] >= v["PR-AUC"] and c["recall a igual volumen"] >= v["recall a igual volumen"] - MAX_RECALL_LOSS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--champion", required=True)
    ap.add_argument("--holdout", required=True, help="mes de prueba, AAAA-MM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[DOMINANT])
    a = ap.parse_args()

    raw = load(a.data)
    raw = raw[~raw.empresa.isin(a.exclude)].reset_index(drop=True)
    champion = joblib.load(a.champion)
    start = pd.Period(a.holdout).start_time
    if pd.Timestamp(champion.meta.get("fecha_max", "1900-01-01")) >= start:
        print(f"AVISO: el vigente se entrenó con datos hasta {champion.meta['fecha_max']}, que incluyen el mes de prueba; "
              "la comparación lo favorece.", file=sys.stderr)
    train, holdout = raw[raw.fecha < start], raw[raw.mes == a.holdout]
    challenger, _ = train_risk_model(train, champion.meta["params"])
    summary, by_company = compare(champion, challenger, holdout)
    c, v = summary.loc["retador"], summary.loc["vigente"]
    promote = should_promote(summary)
    print(summary.round(3).to_string(), "\n\nROC-AUC por empresa:\n", by_company.round(3).to_string())
    report = {"promover": bool(promote), "mes_prueba": a.holdout, "vigente": v.to_dict(), "retador": c.to_dict()}
    if promote:
        final, _ = train_risk_model(raw, champion.meta["params"],
                                    meta={"version": Path(a.out).stem, "reemplaza_a": champion.meta.get("version")})
        joblib.dump(final, a.out, compress=3)
        report["guardado_en"] = a.out
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
