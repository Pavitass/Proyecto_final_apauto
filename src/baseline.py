import sys, warnings, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
warnings.filterwarnings("ignore")
df = pd.read_csv(sys.argv[1])
big = df.empresa.value_counts().idxmax()
df["hs"] = np.sin(2*np.pi*df.hour_local/24); df["hc"] = np.cos(2*np.pi*df.hour_local/24)
df["ig"] = (df.channel == "instagram").astype(int)
for c in ["ai_chars_mean", "cust_chars_mean", "cust_chars_max", "first_ai_reply_s", "cust_gap_mean_s"]:
    df[c] = np.sign(df[c]) * np.log1p(df[c].abs())
X_cols = [c for c in df.columns if c not in ("empresa", "cliente", "fecha", "channel", "hour_local", "needed_human")]
def evaluate(tr, te, label):
    y, yt = tr.needed_human, te.needed_human
    out = [label, f"n_test={len(te)} pos={yt.mean():.0%}"]
    for name, m in [("logreg", make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                    ("gboost", HistGradientBoostingClassifier(class_weight="balanced", random_state=0))]:
        m.fit(tr[X_cols], y); p = m.predict_proba(te[X_cols])[:, 1]
        out.append(f"{name}: ROC-AUC {roc_auc_score(yt, p):.2f} PR-AUC {average_precision_score(yt, p):.2f} (azar {yt.mean():.2f}) recall {recall_score(yt, p>0.5):.2f} prec {precision_score(yt, p>0.5):.2f}")
    print(" | ".join(out))
rest = df[df.empresa != big]
# 1) companies never seen (leave-one-company-out over the rest, pooled)
preds = []
for e in rest.empresa.unique():
    tr, te = rest[rest.empresa != e], rest[rest.empresa == e]
    if te.needed_human.nunique() < 2: continue
    m = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(tr[X_cols], tr.needed_human)
    preds.append(pd.DataFrame({"y": te.needed_human, "p": m.predict_proba(te[X_cols])[:, 1], "e": e}))
P = pd.concat(preds)
print("empresas no vistas (sin", big, ") gboost: ROC-AUC", round(roc_auc_score(P.y, P.p), 2), "PR-AUC", round(average_precision_score(P.y, P.p), 2), "azar", round(P.y.mean(), 2))
print("  por empresa ROC-AUC:", {e: round(roc_auc_score(g.y, g.p), 2) for e, g in P.groupby("e") if g.y.nunique() > 1})
# 2) time split without the dominant company
evaluate(rest[rest.fecha < "2026-10-01"], rest[rest.fecha >= "2026-10-01"], f"tiempo, sin {big}")
# 3) time split with everything
evaluate(df[df.fecha < "2026-10-01"], df[df.fecha >= "2026-10-01"], "tiempo, todo")
m = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(rest[rest.fecha < "2026-10-01"][X_cols], rest[rest.fecha < "2026-10-01"].needed_human)
from sklearn.inspection import permutation_importance
te = rest[rest.fecha >= "2026-10-01"]
imp = permutation_importance(m, te[X_cols], te.needed_human, scoring="roc_auc", n_repeats=5, random_state=0)
print("top variables:", sorted(zip(X_cols, imp.importances_mean.round(3)), key=lambda x: -x[1])[:8])
