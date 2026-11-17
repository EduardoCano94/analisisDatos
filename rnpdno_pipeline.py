#!/usr/bin/env python3
"""
Detección temprana de focos emergentes de desaparición de personas (RNPDNO 2015-2025)

Uso (desde la carpeta del proyecto, con los CSV en ./data):
    python rnpdno_pipeline.py inspect    # 1) ver columnas de cada CSV
    python rnpdno_pipeline.py prepare    # 2) limpieza + panel municipio-mes
    python rnpdno_pipeline.py eda        # 3) análisis exploratorio + gráficas
    python rnpdno_pipeline.py model      # 4) variable objetivo, modelos, evaluación
    python rnpdno_pipeline.py mapas      # 5) mapas (opcional, requiere shapefile INEGI en data/geo)
    python rnpdno_pipeline.py validar    # 4b) evaluación repetida (2022-2025)
    python rnpdno_pipeline.py all        # 2 + 3 + 4 + 4b

IMPORTANTE: no conozco los nombres exactos de columnas de los CSV. El script los
detecta automáticamente, pero corre primero `inspect` y, si algo sale mal,
ajusta las secciones FILES y COLS de abajo.
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
DATA_DIR = Path("data/processed")  # carpeta processed/ del zip de OSF
OUT_DIR = Path("salidas")
FIG_DIR = OUT_DIR / "figuras"

# Si la detección automática de archivos falla, escribe aquí los nombres exactos:
# FILES = {"desaparecidas": "archivo1.csv", "total": "...", "loc_vida": "...", "loc_sin_vida": "..."}
FILES = {}

# Si la detección automática de columnas falla, escribe los nombres exactos:
COLS = {}  # ya no se usa: el esquema del dataset es fijo

START, END = "2015-01", "2025-12"
HORIZONTE = 6          # meses hacia adelante y hacia atrás
PCT_FOCO = 0.90        # foco = top 10 % de municipios
FIN_TRAIN = "2022-12"  # la ventana objetivo termina <= esta fecha
FIN_VAL = "2023-12"    # validación: 2023; prueba: 2024-2025
SEED = 42
USAR_LIGHTGBM = False  # True requiere: pip install lightgbm


# ----------------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------------
def norm(s):
    return str(s).strip().lower().replace(" ", "_")


def find_col(df, patterns, exclude=()):
    for p in patterns:
        for c in df.columns:
            if p in norm(c) and not any(e in norm(c) for e in exclude):
                return c
    return None


def list_csvs():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"No hay CSV en {DATA_DIR.resolve()}. Descarga el dataset de OSF y ponlo ahí.")
    return files


def map_files():
    if FILES:
        return {k: DATA_DIR / v for k, v in FILES.items()}
    found = {}
    for f in list_csvs():
        n = norm(f.stem)
        if "located_dead" in n or "sin_vida" in n:
            found.setdefault("loc_sin_vida", f)
        elif "located_alive" in n or "con_vida" in n:
            found.setdefault("loc_vida", f)
        elif "disappeared_not_located" in n or "desaparec" in n:
            found.setdefault("desaparecidas", f)
        elif "total" in n:
            found.setdefault("total", f)
    if "desaparecidas" not in found:
        sys.exit("No pude identificar el archivo de 'desaparecidas y no localizadas'. "
                 "Edita el diccionario FILES al inicio del script.")
    return found


def read_status(path):
    """Devuelve DataFrame [clave, mes, casos, mujeres] (esquema real del dataset de OSF)."""
    df = pd.read_csv(path, dtype={"cvegeo": str, "cve_estado": str, "cve_mun": str})
    df["clave"] = df["cvegeo"].str.strip().str.zfill(5)  # texto: conserva ceros iniciales
    df["mes"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)), freq="M")
    df["casos"] = df["total"].fillna(0)
    df["mujeres"] = df["female"].fillna(0)
    return df.groupby(["clave", "mes"], as_index=False)[["casos", "mujeres"]].sum()


def catalogo():
    """clave -> estado, municipio (a partir de los CSV de OSF). Se guarda en salidas/."""
    p = OUT_DIR / "catalogo_municipios.csv"
    if p.exists():
        return pd.read_csv(p, dtype={"clave": str})
    partes = []
    for f in map_files().values():
        d = pd.read_csv(f, dtype={"cvegeo": str}, usecols=["cvegeo", "state", "municipality"])
        partes.append(d.rename(columns={"cvegeo": "clave", "state": "estado", "municipality": "municipio"}))
    cat = pd.concat(partes).drop_duplicates("clave")
    cat["clave"] = cat["clave"].str.zfill(5)
    cat["estado"] = cat["estado"].str.title()
    cat["municipio"] = cat["municipio"].str.title()
    OUT_DIR.mkdir(exist_ok=True)
    cat.to_csv(p, index=False)
    return cat


def clave_valida(s):
    """Excluye claves geográficas no resueltas (no 5 dígitos, 000, 998, 999, 99xxx)."""
    ok = s.str.fullmatch(r"\d{5}")
    return ok & ~s.str.endswith("000") & ~s.str.endswith("998") & ~s.str.endswith("999") & ~s.str.startswith("99") & ~s.str.startswith("00")


# ----------------------------------------------------------------------------
# 1) INSPECT
# ----------------------------------------------------------------------------
def cmd_inspect():
    for f in list_csvs():
        df = pd.read_csv(f, dtype=str, nrows=2000)
        print("=" * 80)
        print(f.name, "| columnas:", list(df.columns))
        print(df.head(5).to_string())
    print("\nArchivos detectados:", {k: v.name for k, v in map_files().items()})


# ----------------------------------------------------------------------------
# 2) PREPARE
# ----------------------------------------------------------------------------
def cmd_prepare():
    OUT_DIR.mkdir(exist_ok=True)
    files = map_files()
    meses = pd.period_range(START, END, freq="M")
    data = {}
    for k, p in files.items():
        print(f"Leyendo {k}: {p.name}")
        d = read_status(p)
        n0 = len(d)
        d = d[clave_valida(d["clave"]) & d["mes"].between(meses[0], meses[-1])]
        print(f"  filas: {n0} -> {len(d)} tras excluir claves no válidas / fuera de rango")
        data[k] = d

    claves = sorted(set().union(*[set(d["clave"]) for d in data.values()]))
    idx = pd.MultiIndex.from_product([claves, meses], names=["clave", "mes"])
    panel = pd.DataFrame(index=idx)
    nombres = {"desaparecidas": "desap", "total": "total", "loc_vida": "loc_vida", "loc_sin_vida": "loc_sin_vida"}
    for k, d in data.items():
        s = d.set_index(["clave", "mes"])
        panel[nombres[k]] = s["casos"]
        if k == "desaparecidas":
            panel["desap_mujeres"] = s["mujeres"]
    panel = panel.fillna(0).reset_index()  # meses sin casos -> 0
    panel["mes"] = panel["mes"].astype(str)
    panel.to_csv(OUT_DIR / "panel_municipio_mes.csv.gz", index=False)
    print(f"\nPanel: {len(claves)} municipios x {len(meses)} meses = {len(panel)} filas")
    print(f"Guardado en {OUT_DIR / 'panel_municipio_mes.csv.gz'}")
    catalogo()


def load_panel():
    p = OUT_DIR / "panel_municipio_mes.csv.gz"
    if not p.exists():
        sys.exit("Falta el panel. Corre primero: python rnpdno_pipeline.py prepare")
    df = pd.read_csv(p, dtype={"clave": str})
    df["mes"] = pd.PeriodIndex(df["mes"], freq="M")
    return df


# ----------------------------------------------------------------------------
# 3) EDA
# ----------------------------------------------------------------------------
def cmd_eda():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    df["estado"] = df["clave"].str[:2]
    df["anio"] = df["mes"].dt.year
    df["mes_num"] = df["mes"].dt.month

    # Serie mensual nacional
    m = df.groupby("mes")[["desap", "loc_vida", "loc_sin_vida"]].sum()
    ax = m.plot(figsize=(11, 4), title="Casos mensuales a nivel nacional")
    ax.set_xlabel("Mes"); ax.set_ylabel("Casos"); plt.tight_layout()
    plt.savefig(FIG_DIR / "01_serie_mensual.png", dpi=150); plt.close()

    # Anual
    a = df.groupby("anio")["desap"].sum()
    ax = a.plot(kind="bar", figsize=(8, 4), title="Desaparecidas y no localizadas por año")
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_anual.png", dpi=150); plt.close()

    # Estacionalidad
    s = df.groupby(["anio", "mes_num"])["desap"].sum().groupby("mes_num").mean()
    ax = s.plot(kind="bar", figsize=(8, 4), title="Estacionalidad: promedio mensual (2015-2025)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_estacionalidad.png", dpi=150); plt.close()

    # Por estado y municipio
    nom_est = catalogo().assign(ce=lambda d: d["clave"].str[:2]).drop_duplicates("ce").set_index("ce")["estado"]
    e = df.groupby("estado")["desap"].sum().sort_values(ascending=False).head(15)
    e.index = [nom_est.get(k, k) for k in e.index]
    e.plot(kind="barh", figsize=(7, 5), title="Top 15 estados").invert_yaxis()
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_top_estados.png", dpi=150); plt.close()

    mu = df.groupby("clave")["desap"].sum().sort_values(ascending=False)
    cat = catalogo().set_index("clave")
    et = lambda k: f"{cat['municipio'].get(k, k)}, {cat['estado'].get(k, '')}"
    top15 = mu.head(15)
    top15.index = [et(k) for k in top15.index]
    top15.plot(kind="barh", figsize=(9, 5), title="Top 15 municipios").invert_yaxis()
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_top_municipios.png", dpi=150); plt.close()

    # Concentración (curva acumulada)
    acum = mu.cumsum() / mu.sum()
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(acum) + 1) / len(acum) * 100, acum.values * 100)
    plt.xlabel("% de municipios"); plt.ylabel("% acumulado de casos"); plt.title("Concentración territorial")
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG_DIR / "06_concentracion.png", dpi=150); plt.close()

    resumen = {
        "municipios": df["clave"].nunique(),
        "meses": df["mes"].nunique(),
        "total_desap": int(df["desap"].sum()),
        "pct_mujeres": round(100 * df["desap_mujeres"].sum() / max(df["desap"].sum(), 1), 1),
        "pct_municipios_con_casos": round(100 * (mu > 0).mean(), 1),
        "pct_casos_top10pct_mun": round(100 * mu.head(int(len(mu) * .1)).sum() / mu.sum(), 1),
        "pct_celdas_mun_mes_en_cero": round(100 * (df["desap"] == 0).mean(), 1),
    }
    pd.Series(resumen).to_csv(OUT_DIR / "resumen_eda.csv", header=["valor"])
    print(pd.Series(resumen).to_string())
    print(f"Figuras en {FIG_DIR}")


# ----------------------------------------------------------------------------
# 4) MODELO
# ----------------------------------------------------------------------------
def mat(df, col, claves, meses):
    return df.pivot(index="clave", columns="mes", values=col).reindex(index=claves, columns=meses).fillna(0)


def roll(m, w):
    return m.T.rolling(w, min_periods=w).sum().T


def construir_dataset(df):
    meses = pd.period_range(START, END, freq="M")
    claves = sorted(df["clave"].unique())
    h = HORIZONTE
    M = mat(df, "desap", claves, meses)
    W = mat(df, "desap_mujeres", claves, meses)
    LV = mat(df, "loc_vida", claves, meses) if "loc_vida" in df else M * 0
    LS = mat(df, "loc_sin_vida", claves, meses) if "loc_sin_vida" in df else M * 0

    r3, r6, r12 = roll(M, 3), roll(M, 6), roll(M, 12)
    prev6 = r6.shift(6, axis=1)
    prev3 = r3.shift(3, axis=1)
    est = np.array([c[:2] for c in claves])
    E = M.groupby(est).sum()
    e6 = roll(E, 6)
    tend_est = (e6 / (e6.shift(6, axis=1) + 1))
    tend_est = tend_est.loc[est]
    tend_est.index = M.index

    F = {
        "lag1": M,
        "lag3": r3,
        "lag6": r6,
        "lag12": r12,
        "media_movil_3": r3 / 3,
        "media_movil_6": r6 / 6,
        "crecimiento_6m": (r6 - prev6) / (prev6 + 1),
        "crecimiento_3m": (r3 - prev3) / (prev3 + 1),
        "prop_mujeres_6m": (roll(W, 6) / r6.replace(0, np.nan)).fillna(0),
        "loc_vida_6m": roll(LV, 6),
        "loc_sin_vida_6m": roll(LS, 6),
        "meses_con_casos_12m": roll((M > 0).astype(float), 12),
        "tendencia_estatal": tend_est,
    }

    # Foco: top 10 % de municipios (por columna/mes) en la ventana de 6 meses
    q_pas = r6.quantile(PCT_FOCO, axis=0)
    foco_pas = r6.ge(q_pas, axis=1) & (r6 > 0)
    fut6 = r6.shift(-h, axis=1)  # ventana t+1..t+6
    q_fut = fut6.quantile(PCT_FOCO, axis=0)
    foco_fut = fut6.ge(q_fut, axis=1) & (fut6 > 0)

    n, T = M.shape
    out = pd.DataFrame({k: v.values.ravel() for k, v in F.items()})
    out["clave"] = np.repeat(claves, T)
    out["mes"] = np.tile(meses, n)
    out["mes_fin"] = np.tile(meses.shift(h), n)  # fin de la ventana objetivo
    out["mes_del_anio"] = out["mes"].dt.month
    out["foco_actual"] = foco_pas.values.ravel()
    out["foco_futuro"] = foco_fut.values.ravel()
    out["umbral_foco"] = np.tile(q_pas.values, n)  # casos en 6 meses que definen "foco" en cada mes
    out["valida_futuro"] = np.tile(np.arange(T) + h < T, n)
    out["valida_hist"] = np.tile(np.arange(T) >= 11, n)
    out["y"] = (out["foco_futuro"] & ~out["foco_actual"]).astype(int)
    return out


FEATURES = ["lag1", "lag3", "lag6", "lag12", "crecimiento_6m",
            "crecimiento_3m", "prop_mujeres_6m", "loc_vida_6m", "loc_sin_vida_6m",
            "meses_con_casos_12m", "tendencia_estatal", "mes_del_anio"]


# Líneas base: reglas simples sin aprendizaje. Sirven para saber cuánto aportan realmente los modelos.
BASELINES = {
    "Línea base (crecimiento 6m)": lambda X: X["crecimiento_6m"].values,
    "Línea base (casos 6m)": lambda X: X["lag6"].values,
    "Línea base (persistencia 12m)": lambda X: (X["meses_con_casos_12m"] + 1e-3 * X["lag12"]).values,
}


def crear_modelos(pos_w):
    """Regresión logística, Random Forest y boosting (LightGBM > XGBoost > HistGradientBoosting)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    modelos = {
        "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=SEED),
    }
    # Boosting de árboles: HistGradientBoosting de scikit-learn (equivalente a LightGBM, sin dependencias
    # extra). Para usar LightGBM, pon USAR_LIGHTGBM = True al inicio del script.
    if USAR_LIGHTGBM:
        from lightgbm import LGBMClassifier
        modelos["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                             scale_pos_weight=pos_w, random_state=SEED, verbose=-1)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        modelos["Boosting (HistGradientBoosting)"] = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED)
    return modelos


def calibrador(s_fit, y_fit):
    """Calibración isotónica: convierte el puntaje del modelo en la frecuencia realmente observada
    de focos emergentes. Se ajusta con predicciones fuera de muestra. Devuelve una función."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(s_fit, float), np.asarray(y_fit))
    return lambda x: iso.predict(np.asarray(x, float))


def tabla_calibracion(y, p, bins=10):
    """Probabilidad media predicha vs. frecuencia observada por decil de puntaje."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decil"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False) + 1
    return d.groupby("decil").agg(prob_media=("p", "mean"), frecuencia_observada=("y", "mean"), n=("y", "size")).round(4)


def cmd_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance

    OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    ds = construir_dataset(df)
    ds.to_pickle(OUT_DIR / "dataset_modelado.pkl")

    # Solo municipios que NO son foco actualmente, con historial y futuro observable
    base = ds[ds["valida_hist"] & ds["valida_futuro"] & ~ds["foco_actual"]].copy()
    ft, fv = pd.Period(FIN_TRAIN, "M"), pd.Period(FIN_VAL, "M")
    tr = base[base["mes_fin"] <= ft]
    va = base[(base["mes_fin"] > ft) & (base["mes_fin"] <= fv)]
    te = base[base["mes_fin"] > fv]
    print("Umbral de foco (mediana de casos en 6 meses que definen 'foco', por año):")
    print(ds.groupby(ds["mes"].dt.year)["umbral_foco"].median().to_string())
    print(f"Filas  train={len(tr)}  val={len(va)}  test={len(te)}")
    print(f"Focos emergentes (prevalencia)  train={tr.y.mean():.4f}  val={va.y.mean():.4f}  test={te.y.mean():.4f}")
    if min(tr.y.sum(), va.y.sum(), te.y.sum()) == 0:
        sys.exit("Alguno de los conjuntos no tiene casos positivos; revisa el panel.")

    Xtr, ytr = tr[FEATURES], tr["y"]
    Xva, yva = va[FEATURES], va["y"]
    Xte, yte = te[FEATURES], te["y"]
    pos_w = (ytr == 0).sum() / max(ytr.sum(), 1)

    modelos = crear_modelos(pos_w)

    def mejor_umbral(y, s):
        p, r, t = precision_recall_curve(y, s)
        f = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
        return t[np.argmax(f)]

    def evaluar(nombre, s_va, s_te):
        u = mejor_umbral(yva, s_va)
        pred = (s_te >= u).astype(int)
        return {"modelo": nombre,
                "PR-AUC_val": average_precision_score(yva, s_va),
                "PR-AUC_test": average_precision_score(yte, s_te),
                "umbral(val)": u,
                "precision_test": precision_score(yte, pred, zero_division=0),
                "recall_test": recall_score(yte, pred),
                "F1_test": f1_score(yte, pred)}

    res, scores_te, ajustados = [], {}, {}
    # Líneas base (reglas simples)
    for nb, f in BASELINES.items():
        res.append(evaluar(nb, f(Xva), f(Xte)))
        scores_te[nb] = f(Xte)
    for nombre, mdl in modelos.items():
        print(f"Entrenando {nombre} ...")
        mdl.fit(Xtr, ytr)
        s_va, s_te = mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1]
        res.append(evaluar(nombre, s_va, s_te))
        scores_te[nombre] = s_te
        ajustados[nombre] = mdl

    R = pd.DataFrame(res).round(4)
    R["prevalencia_test"] = round(yte.mean(), 4)  # PR-AUC de un azar ~ prevalencia
    R.to_csv(OUT_DIR / "metricas_modelos.csv", index=False)
    print("\n", R.to_string(index=False))

    # Curvas PR
    plt.figure(figsize=(7, 5))
    for n, s in scores_te.items():
        p, r, _ = precision_recall_curve(yte, s)
        plt.plot(r, p, label=f"{n} (AP={average_precision_score(yte, s):.3f})")
    plt.axhline(yte.mean(), ls="--", c="gray", label="Azar")
    plt.xlabel("Recall"); plt.ylabel("Precisión"); plt.title("Curvas Precisión-Recall (prueba 2024-2025)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(FIG_DIR / "07_curvas_pr.png", dpi=150); plt.close()

    # Mejor modelo por PR-AUC en validación (entre los ajustados)
    mejor = R[R["modelo"].isin(ajustados)].sort_values("PR-AUC_val", ascending=False).iloc[0]
    nombre = mejor["modelo"]
    print(f"\nMejor modelo en validación: {nombre}")

    # Importancia de variables (permutación sobre prueba)
    pi = permutation_importance(ajustados[nombre], Xte, yte, scoring="average_precision",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    imp.plot(kind="barh", figsize=(7, 5), title=f"Importancia de variables – {nombre}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "08_importancia.png", dpi=150); plt.close()
    imp.sort_values(ascending=False).to_csv(OUT_DIR / "importancia_variables.csv", header=["importancia"])

    # Calibración (isotónica). Comprobación honesta: se ajusta con validación (modelo entrenado
    # hasta 2022) y se mide en prueba. Brier más bajo = probabilidades más realistas.
    from sklearn.metrics import brier_score_loss
    s_va_m = ajustados[nombre].predict_proba(Xva)[:, 1]
    s_te_m = ajustados[nombre].predict_proba(Xte)[:, 1]
    p_te_cal = calibrador(s_va_m, yva)(s_te_m)
    brier = {"Brier prueba, puntaje sin calibrar": brier_score_loss(yte, s_te_m),
             "Brier prueba, calibrado (isotónica)": brier_score_loss(yte, p_te_cal),
             "Brier prueba, predecir siempre la prevalencia de entrenamiento": brier_score_loss(yte, np.full(len(yte), ytr.mean()))}
    pd.Series(brier).round(4).to_csv(OUT_DIR / "calibracion_brier.csv", header=["valor"])
    tc = tabla_calibracion(yte, p_te_cal)
    tc.to_csv(OUT_DIR / "calibracion_deciles.csv")
    print("\nCalibración (prueba):"); print(pd.Series(brier).round(4).to_string()); print(tc.to_string())
    plt.figure(figsize=(5, 5))
    plt.plot(tc["prob_media"], tc["frecuencia_observada"], "o-", label="Calibrado (isotónica)")
    mx = float(max(tc["prob_media"].max(), tc["frecuencia_observada"].max())) * 1.1
    plt.plot([0, mx], [0, mx], "--", c="gray", label="Calibración perfecta")
    plt.xlabel("Probabilidad media predicha"); plt.ylabel("Frecuencia observada"); plt.title(f"Calibración en prueba – {nombre}")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG_DIR / "11_calibracion.png", dpi=150); plt.close()

    # Reentrenar con train+val; el calibrador se ajusta con las predicciones fuera de muestra
    # sobre prueba (2024-2025) del modelo reentrenado; luego se puntúa el último mes.
    final = modelos[nombre]
    final.fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
    calibrar = calibrador(final.predict_proba(Xte)[:, 1], yte)
    ult = ds[(ds["mes"] == ds["mes"].max()) & ~ds["foco_actual"]].copy()
    ult["puntaje"] = final.predict_proba(ult[FEATURES])[:, 1]
    ult["probabilidad"] = calibrar(ult["puntaje"].values)
    ult = ult.merge(catalogo(), on="clave", how="left")
    cols = ["clave", "estado", "municipio", "probabilidad", "puntaje", "lag6", "lag12", "crecimiento_6m"]
    ult[cols].sort_values(["probabilidad", "puntaje"], ascending=False).to_csv(OUT_DIR / "riesgo_todos_municipios.csv", index=False)
    top = ult.sort_values(["probabilidad", "puntaje"], ascending=False).head(50)[cols]
    top.to_csv(OUT_DIR / "top50_municipios_riesgo.csv", index=False)
    print(f"\nMunicipios puntuados (no-foco en el último mes): {len(ult)}")
    print("Top 10 municipios por riesgo (probabilidad calibrada de ser foco emergente en 6 meses):")
    print(top.head(10).round(3).to_string(index=False))
    print("\nNota: son estimaciones del modelo, no hechos que necesariamente ocurrirán.")



def calibracion_rodante(base, anios=(2023, 2024, 2025)):
    """Evalúa la calibración como se usaría en la práctica: para cada año, el calibrador se ajusta con
    predicciones fuera de muestra del año anterior (modelo entrenado hasta 2 años antes) y se mide en
    ese año con un modelo entrenado hasta el año anterior. Modelo: regresión logística."""
    from sklearn.metrics import brier_score_loss
    filas, deciles = [], []
    for anio in anios:
        P = lambda a: pd.Period(f"{a}-12", "M")
        tr_a = base[base["mes_fin"] <= P(anio - 2)]
        cal = base[base["mes_fin"].dt.year == anio - 1]
        tr_b = base[base["mes_fin"] <= P(anio - 1)]
        te = base[base["mes_fin"].dt.year == anio]
        m_a = crear_modelos(1)["Regresión logística"].fit(tr_a[FEATURES], tr_a["y"])
        f_cal = calibrador(m_a.predict_proba(cal[FEATURES])[:, 1], cal["y"])
        m_b = crear_modelos(1)["Regresión logística"].fit(tr_b[FEATURES], tr_b["y"])
        p = f_cal(m_b.predict_proba(te[FEATURES])[:, 1])
        b_mod = brier_score_loss(te["y"], p)
        b_ref = brier_score_loss(te["y"], np.full(len(te), tr_b["y"].mean()))
        filas.append({"anio": anio, "prevalencia": te["y"].mean(), "prob_media_predicha": p.mean(),
                      "brier_calibrado": b_mod, "brier_prevalencia_train": b_ref,
                      "skill_brier": 1 - b_mod / b_ref})
        d = tabla_calibracion(te["y"], p).reset_index()
        d.insert(0, "anio", anio)
        deciles.append(d)
    R = pd.DataFrame(filas).round(4)
    R.to_csv(OUT_DIR / "calibracion_rodante.csv", index=False)
    pd.concat(deciles).to_csv(OUT_DIR / "calibracion_rodante_deciles.csv", index=False)
    print("\nCalibración rodante (calibrador del año anterior, medido en el año):")
    print(R.to_string(index=False))


# ----------------------------------------------------------------------------
# 4b) VALIDAR: evaluación repetida con orígenes móviles (años de prueba 2022-2025)
# ----------------------------------------------------------------------------
def cmd_validar():
    from sklearn.metrics import average_precision_score, roc_auc_score
    OUT_DIR.mkdir(exist_ok=True)
    ds = construir_dataset(load_panel())
    base = ds[ds["valida_hist"] & ds["valida_futuro"] & ~ds["foco_actual"]].copy()
    K = 50  # presupuesto de alertas: los 50 municipios mejor rankeados cada mes
    filas = []
    for anio in (2022, 2023, 2024, 2025):
        tr = base[base["mes_fin"] <= pd.Period(f"{anio - 1}-12", "M")]
        te = base[base["mes_fin"].dt.year == anio]
        if tr["y"].sum() == 0 or te["y"].sum() == 0:
            continue
        pos_w = (tr["y"] == 0).sum() / max(tr["y"].sum(), 1)
        modelos = crear_modelos(pos_w)
        scores = {nb: f(te) for nb, f in BASELINES.items()}
        for nom, mdl in modelos.items():
            print(f"[{anio}] entrenando {nom} ...")
            mdl.fit(tr[FEATURES], tr["y"])
            scores[nom] = mdl.predict_proba(te[FEATURES])[:, 1]
        for nom, sc in scores.items():
            d = te[["mes", "y"]].assign(s=sc)
            pk = d.groupby("mes").apply(lambda g: g.nlargest(K, "s")["y"].mean()).mean()
            filas.append({"anio_prueba": anio, "modelo": nom, "filas_prueba": len(te),
                          "prevalencia": te["y"].mean(),
                          "PR-AUC": average_precision_score(te["y"], sc),
                          "ROC-AUC": roc_auc_score(te["y"], sc),
                          f"precision@{K}_por_mes": pk})
    R = pd.DataFrame(filas)
    R.round(4).to_csv(OUT_DIR / "validacion_repetida.csv", index=False)
    res = R.groupby("modelo").agg(PR_AUC_media=("PR-AUC", "mean"), PR_AUC_desv=("PR-AUC", "std"),
                                  PR_AUC_min=("PR-AUC", "min"), PR_AUC_max=("PR-AUC", "max"),
                                  prevalencia_media=("prevalencia", "mean"),
                                  precision_top50_media=(f"precision@{K}_por_mes", "mean")).round(4)
    res.to_csv(OUT_DIR / "validacion_repetida_resumen.csv")
    print("\nPR-AUC por año de prueba:")
    print(R.pivot(index="modelo", columns="anio_prueba", values="PR-AUC").round(3).to_string())
    print("\nResumen (4 años de prueba):"); print(res.to_string())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pv = R.pivot(index="anio_prueba", columns="modelo", values="PR-AUC")
    ax = pv.plot(marker="o", figsize=(8, 4.5), title="PR-AUC por año de prueba (origen móvil)")
    ax.plot(pv.index, R.groupby("anio_prueba")["prevalencia"].first().values, "k--", label="Azar (prevalencia)")
    ax.set_xticks(list(pv.index)); ax.set_ylabel("PR-AUC"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG_DIR / "12_validacion_repetida.png", dpi=150); plt.close()
    calibracion_rodante(base)

# ----------------------------------------------------------------------------
# 5) MAPAS (requiere geopandas + shapefile municipal del INEGI)
# ----------------------------------------------------------------------------
def detectar_clave(g, claves, min_overlap=0.5):
    """Devuelve la serie de claves de 5 dígitos de un mapa municipal con columnas desconocidas.
    Prueba (1) cada columna como clave completa y (2) pares de columnas estado+municipio,
    y se queda con la que más coincide con las claves del panel."""
    cols = [c for c in g.columns if c.lower() != "geometry"]
    best, best_s = 0.0, None

    def prueba(serie):
        nonlocal best, best_s
        ov = serie.isin(claves).mean()
        if ov > best:
            best, best_s = ov, serie

    for c in cols:
        prueba(g[c].astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(5))
    for ce in cols:
        for cm in cols:
            if ce != cm:
                prueba(g[ce].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
                       + g[cm].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3))
    if best_s is None or best < min_overlap:
        print(f"El mapa no trae clave INEGI utilizable. Columnas: {cols}")
        return None
    print(f"Clave municipal detectada automáticamente (coincidencia {best:.0%} con el panel)")
    return best_s


ESTADOS = {"01": "aguascalientes", "02": "baja california", "03": "baja california sur", "04": "campeche",
           "05": "coahuila", "06": "colima", "07": "chiapas", "08": "chihuahua", "09": "ciudad de mexico",
           "10": "durango", "11": "guanajuato", "12": "guerrero", "13": "hidalgo", "14": "jalisco",
           "15": "mexico", "16": "michoacan", "17": "morelos", "18": "nayarit", "19": "nuevo leon",
           "20": "oaxaca", "21": "puebla", "22": "queretaro", "23": "quintana roo", "24": "san luis potosi",
           "25": "sinaloa", "26": "sonora", "27": "tabasco", "28": "tamaulipas", "29": "tlaxcala",
           "30": "veracruz", "31": "yucatan", "32": "zacatecas"}


def _nm(x):
    """Normaliza un nombre: arregla acentos mal codificados, quita acentos/espacios/símbolos."""
    import unicodedata
    x = str(x)
    try:
        x = x.encode("latin1").decode("utf8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()
    return "".join(ch for ch in x.lower() if ch.isalpha())


def clave_por_nombre(g):
    """Asigna clave INEGI a un mapa que solo trae nombres de estado (NAME_1) y municipio (NAME_2)."""
    import difflib
    if not {"NAME_1", "NAME_2"} <= set(g.columns):
        sys.exit(f"El mapa no trae clave ni NAME_1/NAME_2. Columnas: {list(g.columns)}. "
                 "Pégame la lista y una fila de ejemplo.")
    cat = catalogo()
    cat = cat[cat["clave"].str[:2].isin(ESTADOS)].copy()
    cat["k"] = cat["municipio"].map(_nm)
    por_edo = {e: dict(zip(d["k"], d["clave"])) for e, d in cat.groupby(cat["clave"].str[:2])}
    est_norm = {_nm(v): k for k, v in ESTADOS.items()}
    est_norm["distritofederal"] = "09"

    def cod_estado(n):
        n = _nm(n)
        if n in est_norm:
            return est_norm[n]
        for k, c in est_norm.items():
            if k.startswith(n) or n.startswith(k):
                return c
        m = difflib.get_close_matches(n, list(est_norm), n=1, cutoff=0.8)
        return est_norm[m[0]] if m else None

    cache = {n: cod_estado(n) for n in g["NAME_1"].unique()}
    out = []
    for e, mun in zip(g["NAME_1"], g["NAME_2"]):
        ce, k = cache[e], _nm(mun)
        d = por_edo.get(ce, {})
        if k in d:
            out.append(d[k]); continue
        cand = [v for kk, v in d.items() if len(k) >= 5 and (kk.startswith(k) or k.startswith(kk))]
        if len(cand) == 1:
            out.append(cand[0]); continue
        m = difflib.get_close_matches(k, list(d), n=1, cutoff=0.85)
        out.append(d[m[0]] if m else None)
    res = pd.Series(out, index=g.index)

    # cobertura: ¿qué parte de los municipios (y de los casos) del panel quedó dibujable?
    pan = load_panel().groupby("clave")["desap"].sum()
    usados = set(res.dropna())
    print(f"Unión por nombre: {res.notna().mean():.0%} de los polígonos con clave; "
          f"{pan.index.isin(usados).mean():.0%} de los municipios del panel y "
          f"{pan[pan.index.isin(usados)].sum() / pan.sum():.0%} de los casos quedaron en el mapa")
    return res.fillna("SIN_CLAVE")


def cmd_mapas():
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("Falta geopandas: pip install geopandas")
    geo = [p for ext in ("*.shp", "*.geojson", "*.json") for p in sorted(Path("data/geo").rglob(ext))]
    shp = [p for p in geo if "mun" in p.name.lower() or "mgem" in p.name.lower()] or geo
    if not shp:
        sys.exit("Falta el mapa municipal. Pon el shapefile ('00mun.shp') o un GeoJSON municipal "
                 "del INEGI dentro de data/geo/")
    print("Usando:", shp[0])
    g = gpd.read_file(shp[0])
    g.columns = [c.upper() if c.lower() != "geometry" else c for c in g.columns]
    if "CVEGEO" not in g.columns:
        k = detectar_clave(g, set(load_panel()["clave"]))
        g["CVEGEO"] = k if k is not None else clave_por_nombre(g)
    g["clave"] = g["CVEGEO"].astype(str).str.zfill(5)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_panel()
    tot = df.groupby("clave")["desap"].sum().rename("casos").reset_index()
    m = g.merge(tot, on="clave", how="left").fillna({"casos": 0})
    m["log_casos"] = np.log10(m["casos"] + 1)
    ax = m.plot(column="log_casos", cmap="OrRd", legend=True, figsize=(10, 7),
                edgecolor="none", legend_kwds={"label": "log10(casos + 1)", "shrink": .6})
    ax.set_axis_off(); ax.set_title("Desaparecidas y no localizadas por municipio, 2015-2025")
    plt.tight_layout(); plt.savefig(FIG_DIR / "09_mapa_casos.png", dpi=150); plt.close()

    r = OUT_DIR / "riesgo_todos_municipios.csv"
    if r.exists():
        rk = pd.read_csv(r, dtype={"clave": str})[["clave", "probabilidad"]]
        m2 = g.merge(rk, on="clave", how="left")
        ax = m2.plot(column="probabilidad", cmap="viridis", legend=True, figsize=(10, 7), edgecolor="none",
                     missing_kwds={"color": "lightgrey", "label": "Foco actual / sin dato"},
                     legend_kwds={"label": "Probabilidad calibrada estimada (6 meses)", "shrink": .6})
        ax.set_axis_off(); ax.set_title("Riesgo estimado de convertirse en foco emergente (estimación del modelo)")
        plt.tight_layout(); plt.savefig(FIG_DIR / "10_mapa_riesgo.png", dpi=150); plt.close()
    print("Mapas guardados en", FIG_DIR)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["inspect", "prepare", "eda", "model", "validar", "mapas", "all"])
    a = ap.parse_args()
    if a.paso == "inspect":
        cmd_inspect()
    elif a.paso == "prepare":
        cmd_prepare()
    elif a.paso == "eda":
        cmd_eda()
    elif a.paso == "model":
        cmd_model()
    elif a.paso == "validar":
        cmd_validar()
    elif a.paso == "mapas":
        cmd_mapas()
    else:
        cmd_prepare(); cmd_eda(); cmd_model(); cmd_validar()


if __name__ == "__main__":
    main()