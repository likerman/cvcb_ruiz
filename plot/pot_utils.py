# pot_utils.py
# ------------------------------------------------------------
# Utilidades para estimación del potencial geotermoeléctrico (método volumétrico)
# y métricas térmicas a partir de outputs Underworld/UWGeodynamics.
#
# Convenciones (coherentes con tu notebook):
# - Coordenadas x,z leídas de mesh.h5 (o mesh-*.h5) y convertidas a km mediante m_to_km.
# - Temperatura leída de temperature-<ts>.h5 (o temperatureField-<ts>.h5).
# - Conversión K->°C: se aplica T_C = T_raw - temp_offset.
#   (por default, temp_offset=273.15 si temp_is_kelvin=True; si no, usa temp_offset provisto)
# - Perfil a "depth_bsl_km": z_target = z_surf_km - depth_bsl_km
# - Reservorio efectivo: Ares_km2 = L_int_km * Lstrike_km ; Vres_km3 = Ares_km2 * h_km
# - Potencia eléctrica equivalente promedio: Pe_MW (MW) sobre life_years
# - Normalización por área: Pe_MW_km2 = Pe_MW / Ares_km2  (MW/km^2 ≡ W/m^2 numéricamente)
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import glob
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import h5py


FAMILY_LABEL_MAP_ES = {
    "Chamber_Radius": "Tamaño",
    "Chamber_Temp": "Temperatura",
    "Chamber_Depth": "Profundidad",
    "Chamber_Pulse": "Pulsos",
}

# ----------------------------
# I/O helpers
# ----------------------------

def timestep_indices_in(model_dir: str) -> List[int]:
    """
    Devuelve lista ordenada de timesteps detectados por archivos de temperatura.
    Soporta:
      - temperature-<ts>.h5
      - temperatureField-<ts>.h5
    """
    cands = sorted(
        glob.glob(os.path.join(model_dir, "temperature-*.h5"))
        + glob.glob(os.path.join(model_dir, "temperatureField-*.h5"))
    )
    out: List[int] = []
    for p in cands:
        m = re.search(r"-(\d+)\.h5$", os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def _first_numeric_dataset(h5: h5py.File) -> Optional[np.ndarray]:
    """Devuelve el primer dataset numérico (DFS) encontrado en el archivo."""
    def _walk(g: h5py.Group):
        for k in g.keys():
            obj = g[k]
            if isinstance(obj, h5py.Dataset):
                try:
                    arr = obj[()]
                    if np.issubdtype(np.asarray(arr).dtype, np.number):
                        return np.asarray(arr)
                except Exception:
                    pass
            elif isinstance(obj, h5py.Group):
                r = _walk(obj)
                if r is not None:
                    return r
        return None
    return _walk(h5)


def read_coords(model_dir: str) -> np.ndarray:
    """
    Lee mesh.h5 y retorna array (N, D) de vértices.
    Busca datasets típicos: 'vertices' u otro dataset numérico.
    """
    mesh_path = os.path.join(model_dir, "mesh.h5")
    if not os.path.exists(mesh_path):
        # fallback: mesh-*.h5
        ms = sorted(glob.glob(os.path.join(model_dir, "mesh-*.h5")))
        if not ms:
            raise FileNotFoundError(f"No encuentro mesh.h5 en {model_dir}")
        mesh_path = ms[-1]

    with h5py.File(mesh_path, "r") as f:
        if "vertices" in f:
            V = np.asarray(f["vertices"][()])
        else:
            V = _first_numeric_dataset(f)
            if V is None:
                raise RuntimeError(f"No encontré dataset numérico en {mesh_path}")
            V = np.asarray(V)
    V = V.reshape((-1, V.shape[-1])) if V.ndim > 1 else V.reshape((-1, 1))
    return V


def _coords_xz_from_mesh(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae x y z desde vertices. Asume:
    - 2D: (x,z) o (x,y) -> tomamos [0] y [1]
    - 3D: (x,y,z) -> [0] y [2]
    """
    V = np.asarray(vertices)
    if V.ndim != 2 or V.shape[1] < 2:
        raise ValueError(f"vertices debe ser (N,D) con D>=2. Got {V.shape}")

    if V.shape[1] >= 3:
        x = V[:, 0]
        z = V[:, 2]
    else:
        x = V[:, 0]
        z = V[:, 1]
    return x, z


def find_temperature_file(model_dir: str, timestep: int) -> Tuple[str, int]:
    """
    Encuentra archivo de temperatura más cercano al timestep pedido.
    Devuelve (path, ts_usado).
    """
    cands = sorted(
        glob.glob(os.path.join(model_dir, "temperature-*.h5"))
        + glob.glob(os.path.join(model_dir, "temperatureField-*.h5"))
    )
    if not cands:
        raise FileNotFoundError(f"No hay temperature-*.h5 en {model_dir}")

    pairs = []
    for p in cands:
        m = re.search(r"-(\d+)\.h5$", os.path.basename(p))
        if m:
            pairs.append((int(m.group(1)), p))
    if not pairs:
        # si no pudimos parsear, devolvemos el primero
        return cands[0], timestep

    idxs = np.array([i for i, _ in pairs], dtype=int)
    j = int(np.argmin(np.abs(idxs - int(timestep))))
    return pairs[j][1], int(pairs[j][0])


def load_xzT(
    model_dir: str,
    timestep: int,
    m_to_km: float = 1.0,
    temp_offset: float = 273.15,
    coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Devuelve x_km, z_km, T_C, ts_usado.

    temp_offset se resta siempre: T_C = T_raw - temp_offset.
    """
    V = read_coords(model_dir) if coords is None else np.asarray(coords)
    x_raw, z_raw = _coords_xz_from_mesh(V)

    t_path, ts_used = find_temperature_file(model_dir, timestep)
    with h5py.File(t_path, "r") as f:
        Tarr = _first_numeric_dataset(f)
    if Tarr is None:
        raise RuntimeError(f"No encontré dataset numérico en {t_path}")
    T_raw = np.asarray(Tarr).reshape(-1)

    n = min(len(x_raw), len(T_raw))
    x_raw, z_raw, T_raw = x_raw[:n], z_raw[:n], T_raw[:n]

    x_km = x_raw * float(m_to_km)
    z_km = z_raw * float(m_to_km)
    T_C = T_raw - float(temp_offset)
    return x_km, z_km, T_C, int(ts_used)


def load_time(model_dir: str, timestep: int, time_unit: float = 1.0) -> float:
    """
    Lee timeField-<ts>.h5 (o projTimeField-<ts>.h5) y devuelve el tiempo escalar.
    """
    cands = sorted(
        glob.glob(os.path.join(model_dir, "timeField-*.h5"))
        + glob.glob(os.path.join(model_dir, "projTimeField-*.h5"))
    )
    if not cands:
        raise FileNotFoundError(f"No hay timeField-*.h5 en {model_dir}")

    pairs = []
    for p in cands:
        m = re.search(r"-(\d+)\.h5$", os.path.basename(p))
        if m:
            pairs.append((int(m.group(1)), p))
    if not pairs:
        chosen = cands[0]
    else:
        idxs = np.array([i for i, _ in pairs])
        j = int(np.argmin(np.abs(idxs - int(timestep))))
        chosen = pairs[j][1]

    with h5py.File(chosen, "r") as f:
        arr = _first_numeric_dataset(f)
    if arr is None:
        raise RuntimeError(f"No se pudo leer tiempo en {chosen}")
    return float(np.nanmean(arr)) * float(time_unit)


# ----------------------------
# Perfil a profundidad y métricas Lx
# ----------------------------

def _estimate_tol_z_km(z_km: np.ndarray) -> float:
    """
    Estima un espesor de selección razonable para una "corte" horizontal
    a partir del espaciado típico en z.
    """
    z = np.asarray(z_km, dtype=float)
    if z.size < 10:
        return 0.1
    # usar z redondeado para aproximar niveles
    zr = np.unique(np.round(z, 4))
    if len(zr) < 5:
        return 0.1
    dz = np.diff(np.sort(zr))
    dz = dz[dz > 0]
    if dz.size == 0:
        return 0.1
    dz0 = float(np.percentile(dz, 5))  # más robusto que min
    return max(0.5 * dz0, 1e-3)


def Tx_profile_at_depth_bsl(
    model_dir: str,
    ts: int,
    z_surf_km: float,
    depth_bsl_km: float,
    tol_z_km: Optional[float] = None,
    temp_is_kelvin: bool = False,
    temp_offset: float = 273.15,
    m_to_km: float = 1.0,
    coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construye perfil 1D T(x) a una profundidad depth_bsl_km bajo superficie.

    Retorna:
      x_prof_km (1D), T_prof_C (1D), ts_used
    """
    # offset efectivo
    if temp_offset is None:
        temp_offset_eff = 273.15 if temp_is_kelvin else 0.0
    else:
        temp_offset_eff = float(temp_offset)

    x_km, z_km, T_C, ts_used = load_xzT(
        model_dir, ts, m_to_km=m_to_km, temp_offset=temp_offset_eff, coords=coords
    )

    z_target = float(z_surf_km) - float(depth_bsl_km)
    if tol_z_km is None:
        tol = _estimate_tol_z_km(z_km)
    else:
        tol = float(tol_z_km)

    # máscara de slice horizontal
    dz = np.abs(z_km - z_target)
    mask = dz <= tol
    if np.sum(mask) < 10:
        # fallback: tomar el percentil más cercano para asegurar puntos
        q = np.percentile(dz, 0.5)
        mask = dz <= max(q, tol)

    x = x_km[mask]
    T = T_C[mask]
    if x.size == 0:
        return np.array([]), np.array([]), int(ts_used)

    # colapsar múltiples puntos por x: agrupar por x redondeado
    xr = np.round(x.astype(float), 4)
    df = pd.DataFrame({"x": xr, "T": T.astype(float)})
    prof = df.groupby("x", as_index=False)["T"].mean().sort_values("x")
    return prof["x"].to_numpy(), prof["T"].to_numpy(), int(ts_used)


def compute_Lx_metrics_from_profile(
    x_km: np.ndarray,
    T_C: np.ndarray,
    Th_C: float,
) -> Dict[str, Any]:
    """
    Dado un perfil T(x), computa:
      - intervals: lista de (x0, x1) con T>=Th
      - L_int_km: suma de longitudes
      - Lmax_km: máxima longitud de parche
      - Npatches: cantidad de parches
    """
    x = np.asarray(x_km, dtype=float)
    T = np.asarray(T_C, dtype=float)
    if x.size == 0:
        return {"intervals": [], "L_int_km": 0.0, "Lmax_km": 0.0, "Npatches": 0}

    # ordenar por x
    order = np.argsort(x)
    x = x[order]
    T = T[order]

    ok = np.isfinite(T) & np.isfinite(x)
    x = x[ok]; T = T[ok]
    if x.size == 0:
        return {"intervals": [], "L_int_km": 0.0, "Lmax_km": 0.0, "Npatches": 0}

    hot = T >= float(Th_C)

    intervals: List[Tuple[float, float]] = []
    in_seg = False
    x0 = None
    for i in range(len(x)):
        if hot[i] and not in_seg:
            in_seg = True
            x0 = x[i]
        if in_seg and (not hot[i] or i == len(x) - 1):
            # cerrar segmento en el último punto caliente
            if hot[i] and i == len(x) - 1:
                x1 = x[i]
            else:
                x1 = x[i-1] if i > 0 else x[i]
            if x0 is not None and x1 >= x0:
                intervals.append((float(x0), float(x1)))
            in_seg = False
            x0 = None

    lengths = [b - a for a, b in intervals if b >= a]
    L_int = float(np.sum(lengths)) if lengths else 0.0
    Lmax = float(np.max(lengths)) if lengths else 0.0
    return {"intervals": intervals, "L_int_km": L_int, "Lmax_km": Lmax, "Npatches": int(len(intervals))}


def mean_T_over_intervals(
    x_km: np.ndarray,
    T_C: np.ndarray,
    intervals: Sequence[Tuple[float, float]],
    mode: str = "mean",
) -> float:
    """
    Temperatura representativa dentro de los intervalos (parches) calientes.
    mode:
      - 'mean' (default)
      - 'min'
      - 'p10'
    """
    x = np.asarray(x_km, dtype=float)
    T = np.asarray(T_C, dtype=float)
    if x.size == 0 or len(intervals) == 0:
        return float("nan")

    mask = np.zeros_like(x, dtype=bool)
    for a, b in intervals:
        mask |= (x >= a) & (x <= b)

    vals = T[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")

    mode = (mode or "mean").lower().strip()
    if mode == "min":
        return float(np.min(vals))
    if mode in ("p10", "p_10", "p10%"):
        return float(np.percentile(vals, 10))
    return float(np.mean(vals))


# ----------------------------
# Tabla de potencial y receta volumétrica
# ----------------------------

def build_potential_table(
    MODEL_DIRS: Sequence[str],
    ts: int,
    z_surf_km: float,
    depth_bsl_km: float,
    Th_list: Sequence[float],
    Lstrike_km_list: Sequence[float],
    h_km_list: Sequence[float],
    tol_z_km: Optional[float] = None,
    temp_is_kelvin: bool = False,
    temp_offset: float = 273.15,
    m_to_km: float = 1.0,
    Trep_mode: str = "mean",
) -> pd.DataFrame:
    """
    Devuelve DataFrame con métricas térmicas y geometría para cada combinación (Th, Lstrike, h)
    y cada modelo en MODEL_DIRS.
    """
    rows: List[Dict[str, Any]] = []

    for mdir in MODEL_DIRS:
        coords = read_coords(mdir)  # cache local por modelo
        x_prof, T_prof, ts_used = Tx_profile_at_depth_bsl(
            mdir, ts, z_surf_km=z_surf_km, depth_bsl_km=depth_bsl_km,
            tol_z_km=tol_z_km,
            temp_is_kelvin=temp_is_kelvin, temp_offset=temp_offset,
            m_to_km=m_to_km, coords=coords
        )

        for Th in Th_list:
            mx = compute_Lx_metrics_from_profile(x_prof, T_prof, Th_C=float(Th))
            intervals = mx["intervals"]
            L_int_km = float(mx["L_int_km"])
            Lmax_km  = float(mx["Lmax_km"])
            Npatches = int(mx["Npatches"])
            Tmean_above = mean_T_over_intervals(x_prof, T_prof, intervals, mode=Trep_mode)

            for Lstrike in Lstrike_km_list:
                for h in h_km_list:
                    Ares_km2 = float(L_int_km) * float(Lstrike)
                    Vres_km3 = float(Ares_km2) * float(h)
                    rows.append({
                        "model_dir": str(mdir),
                        "ts": int(ts_used),
                        "Th_C": float(Th),
                        "Lstrike_km": float(Lstrike),
                        "h_km": float(h),
                        "L_int_km": float(L_int_km),
                        "Lmax_km": float(Lmax_km),
                        "Npatches": int(Npatches),
                        "Tmean_above_C": float(Tmean_above) if np.isfinite(Tmean_above) else np.nan,
                        "Ares_km2": float(Ares_km2),
                        "Vres_km3": float(Vres_km3),
                    })

    return pd.DataFrame(rows)


def eta_from_T_simple(T_C: float) -> float:
    """
    Eficiencia simple (placeholder):
    - 0 si T<=80°C
    - lineal hasta 0.14 a 240°C
    - satura en 0.14
    """
    T = float(T_C)
    if not np.isfinite(T):
        return 0.0
    if T <= 80.0:
        return 0.0
    if T >= 240.0:
        return 0.14
    return 0.14 * (T - 80.0) / (240.0 - 80.0)


def apply_volumetric_recipe(
    df0: pd.DataFrame,
    phi: float = 0.10,
    rho_r: float = 2700.0,
    c_r: float = 1000.0,
    rho_f: float = 1000.0,
    c_f: float = 4200.0,
    Tref_C: float = 15.0,
    Rf: float = 0.15,
    life_years: float = 30.0,
) -> pd.DataFrame:
    """
    Aplica receta volumétrica a df0:
      Q = (rho c_eff) * V * (Trep - Tref)
      Qr = Rf * Q
      Pe = eta(Trep) * Qr / life_seconds
    """
    df = df0.copy()

    # conversiones
    df["Vres_m3"] = pd.to_numeric(df["Vres_km3"], errors="coerce") * 1e9
    Trep = pd.to_numeric(df["Tmean_above_C"], errors="coerce")
    dT = Trep - float(Tref_C)

    rho_c_eff = (1.0 - float(phi)) * float(rho_r) * float(c_r) + float(phi) * float(rho_f) * float(c_f)
    life_seconds = float(life_years) * 365.25 * 24.0 * 3600.0

    # energia térmica in situ
    Q = rho_c_eff * df["Vres_m3"] * dT
    Q = pd.to_numeric(Q, errors="coerce")

    # recuperable
    Qr = float(Rf) * Q

    # eficiencia
    eta = Trep.apply(eta_from_T_simple)

    # potencia
    Pe_W = eta * Qr / life_seconds

    # reglas de anulación (coherentes)
    bad = (~np.isfinite(Pe_W)) | (~np.isfinite(Trep)) | (~np.isfinite(dT)) | (df["Vres_m3"] <= 0) | (dT <= 0)
    Pe_W = Pe_W.mask(bad, other=0.0)

    df["Pe_W"] = Pe_W.astype(float)
    df["Pe_MW"] = df["Pe_W"] / 1e6
    return df


def add_pe_area_norm(
    df: pd.DataFrame,
    pe_col: str = "Pe_MW",
    area_col: str = "Ares_km2",
    out_col: str = "Pe_MW_km2",
) -> pd.DataFrame:
    """
    Agrega potencia normalizada por área.

    Definición:
      Pe_norm = Pe_MW / Ares_km2

    Unidades:
      - Pe_MW_km2: MW/km² (numéricamente equivalente a W/m²).
      - Pe_W_m2:   W/m² (misma magnitud numérica).

    Compatibilidad:
      Además de `out_col`, siempre crea/actualiza también:
        - Pe_MW_km2 (columna canónica)
        - PeA_MW_km2 (alias histórico)
    """
    out = df.copy()
    P = pd.to_numeric(out[pe_col], errors="coerce")
    A = pd.to_numeric(out[area_col], errors="coerce")

    Pe_norm = np.where((A > 0) & np.isfinite(A) & np.isfinite(P), P / A, 0.0)

    # columna solicitada
    out[out_col] = Pe_norm

    # columnas estándar/compatibilidad
    out["Pe_MW_km2"] = Pe_norm
    out["PeA_MW_km2"] = Pe_norm

    # W/m² equivalente (numéricamente igual)
    out["Pe_W_m2"] = Pe_norm
    return out
def list_model_dirs_all_families(base_dir: Path) -> List[Path]:
    """
    Busca carpetas de familia que empiezan con 'Chamber_' dentro de base_dir,
    y devuelve lista de subdirectorios (escenarios) existentes.
    """
    base = Path(base_dir)
    fams = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("Chamber_")])
    out: List[Path] = []
    for fam in fams:
        for scen in sorted([p for p in fam.iterdir() if p.is_dir() and not p.name.startswith(".")]):
            out.append(scen)
    return out


def parse_param_from_scenario_name(scenario: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Parsea nombre del escenario para extraer (param_name, param_value).
    Reglas compatibles con el notebook:
      - Depth_<val>_...
      - Temp_<val>_...
      - Pulse_<val>_...
      - R_<int>_...  (radio)
    """
    m = re.match(r"^(Depth|Temp|Pulse)_(\-?\d+(?:\.\d+)?)_", scenario)
    if m:
        return m.group(1), float(m.group(2))
    m = re.match(r"^(R)_(\d+)_", scenario)
    if m:
        return m.group(1), float(m.group(2))
    return None, None


def add_metadata(df: pd.DataFrame, model_path: Any) -> pd.DataFrame:
    """
    Agrega columnas:
      - family: nombre carpeta padre (Chamber_*)
      - scenario: nombre carpeta del escenario
      - param_name / param_value según parse_param_from_scenario_name
      - model_path: path completo
    """
    p = Path(str(model_path))
    fam = p.parent.name
    scenario = p.name
    pname, pval = parse_param_from_scenario_name(scenario)

    out = df.copy()
    out["family"] = fam
    out["scenario"] = scenario
    out["param_name"] = pname
    out["param_value"] = pval
    out["model_path"] = str(p)
    return out


def plot_steady_pe_bands_family(
    df_steady_full: pd.DataFrame,
    value_col: str = "Pe_MW_km2",
    T_STEADY_KA: float = 800.0,
    family_label_map: Optional[Dict[str, str]] = None,
    family_order: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10, 4.8),
    out_png: Optional[str] = "Fig_4_1_steady_PeA_bandas_por_familia.png",
    show: bool = True,
    # ---- compat notebook (aliases) ----
    crit_col: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
    outpath: Optional[Any] = None,
):
    """
    Figura tipo errorbar min–med–max por familia y umbral (Th).

    Retrocompatible:
      - crit_col  -> value_col
      - label_map -> family_label_map
      - outpath   -> out_png
    """
    # ---- aliases ----
    if crit_col is not None:
        value_col = crit_col
    if label_map is not None:
        family_label_map = label_map
    if outpath is not None:
        out_png = str(outpath)

    if family_label_map is None:
        family_label_map = FAMILY_LABELS_ES

    # Bands
    bands_scenario = build_bands_scenario_from_steady(df_steady_full, value_col=value_col)
    bands_family = build_bands_family(bands_scenario).copy()

    families = list(bands_family["family"].unique())
    if family_order is not None:
        families = [f for f in family_order if f in families]

    Ths = sorted(bands_family["Th_C"].unique())
    x = np.arange(len(families))
    w = 0.22

    fig, ax = plt.subplots(figsize=figsize)

    for i, Th in enumerate(Ths):
        sub = bands_family[bands_family["Th_C"] == Th].set_index("family").reindex(families)
        xi = x + (i - (len(Ths) - 1) / 2.0) * w

        ymed = sub["Pe_med"].values.astype(float)
        ymin = sub["Pe_min"].values.astype(float)
        ymax = sub["Pe_max"].values.astype(float)

        ax.errorbar(
            xi, ymed,
            yerr=[ymed - ymin, ymax - ymed],
            fmt="o", capsize=3,
            label=f"Th={int(Th)}°C"
        )

    ax.set_xticks(x)
    ax.set_xticklabels([family_label_map.get(f, f) for f in families])

    if value_col in ("Pe_MW_km2", "PeA_MW_km2", "Pe_W_m2"):
        ax.set_ylabel("Pe/A (MW/km²) — banda sobre (Lstrike,h)")
        ax.set_title(f"Steady state (t={T_STEADY_KA:.0f} ka): bandas de Pe/A por familia y umbral")
    else:
        ax.set_ylabel("Pe (MW) — banda sobre (Lstrike,h)")
        ax.set_title(f"Steady state (t={T_STEADY_KA:.0f} ka): bandas de Pe por familia y umbral")

    ax.legend()
    plt.tight_layout()

    if out_png is not None:
        fig.savefig(str(out_png), dpi=300)

    if show:
        plt.show()

    return fig, ax



def build_bands_scenario_from_steady(df_steady_full: pd.DataFrame, value_col: str = "Pe_MW_km2") -> pd.DataFrame:
    """
    Construye bandas por escenario y umbral tomando, para cada (Lstrike,h),
    el valor de value_col y resumiendo como min/med/max sobre geometrías.
    Espera columnas: family, scenario, Th_C y value_col (y opcionalmente Lstrike_km, h_km).
    """
    if value_col not in df_steady_full.columns:
        raise KeyError(f"'{value_col}' no está en df_steady_full.columns")

    gcols = ["family", "scenario", "Th_C"]
    for extra in ["param_name", "param_value"]:
        if extra in df_steady_full.columns:
            gcols.insert(2, extra)  # mantener cerca

    bands = (
        df_steady_full
        .groupby(gcols, as_index=False)
        .agg(
            Pe_min=(value_col, "min"),
            Pe_med=(value_col, "median"),
            Pe_max=(value_col, "max"),
            L_int_km=("L_int_km", "median"),
            Npatches=("Npatches", "median"),
            Tmean_above_C=("Tmean_above_C", "median"),
        )
    )
    return bands

def build_bands_family(bands_scenario: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega bandas por familia y Th a partir de bandas por escenario.
    """
    return (
        bands_scenario
        .groupby(["family", "Th_C"], as_index=False)
        .agg(
            Pe_min=("Pe_min", "min"),
            Pe_med=("Pe_med", "median"),
            Pe_max=("Pe_max", "max"),
            L_int_med=("L_int_km", "median"),
            Npatches_med=("Npatches", "median"),
            Tmean_med=("Tmean_above_C", "median"),
        )
    )
# ============================================================
# PEAK / ESTADOS MÁXIMOS (máximo en el tiempo)
# ============================================================

def iter_timesteps(
    model_dir: str,
    step_every: Optional[int] = None,
    ts_min: Optional[int] = None,
    ts_max: Optional[int] = None,
    skip: Optional[Iterable[int]] = None,
) -> List[int]:
    """
    Lista de timesteps disponibles, con filtros opcionales:
      - step_every: submuestreo (cada N pasos)
      - ts_min / ts_max: ventana de ts (inclusive)
      - skip: conjunto/lista de ts a excluir
    """
    idxs = timestep_indices_in(model_dir)
    if ts_min is not None:
        idxs = [ts for ts in idxs if ts >= int(ts_min)]
    if ts_max is not None:
        idxs = [ts for ts in idxs if ts <= int(ts_max)]
    if skip is not None:
        sk = set(int(s) for s in skip)
        idxs = [ts for ts in idxs if ts not in sk]
    if step_every is not None and int(step_every) > 1:
        idxs = idxs[::int(step_every)]
    return idxs


def build_potential_table_one_model_cached_coords(
    model_dir: str,
    ts: int,
    coords: np.ndarray,
    z_surf_km: float,
    depth_bsl_km: float,
    Th_list: Sequence[float],
    Lstrike_km_list: Sequence[float],
    h_km_list: Sequence[float],
    tol_z_km: Optional[float] = None,
    temp_is_kelvin: bool = False,
    temp_offset: float = 273.15,
    m_to_km: float = 1.0,
    Trep_mode: str = "mean",
) -> pd.DataFrame:
    """
    Variante de build_potential_table() para 1 modelo, reutilizando coords (mesh)
    para evitar re-leer mesh.h5 en cada timestep.
    """
    rows: List[Dict[str, Any]] = []

    x_prof, T_prof, ts_used = Tx_profile_at_depth_bsl(
        model_dir, ts, z_surf_km=z_surf_km, depth_bsl_km=depth_bsl_km,
        tol_z_km=tol_z_km,
        temp_is_kelvin=temp_is_kelvin, temp_offset=temp_offset,
        m_to_km=m_to_km, coords=coords
    )

    for Th in Th_list:
        mx = compute_Lx_metrics_from_profile(x_prof, T_prof, Th_C=float(Th))
        intervals = mx["intervals"]
        L_int_km = float(mx["L_int_km"])
        Lmax_km  = float(mx["Lmax_km"])
        Npatches = int(mx["Npatches"])
        Tmean_above = mean_T_over_intervals(x_prof, T_prof, intervals, mode=Trep_mode)

        for Lstrike in Lstrike_km_list:
            for h in h_km_list:
                Ares_km2 = float(L_int_km) * float(Lstrike)
                Vres_km3 = float(Ares_km2) * float(h)
                rows.append({
                    "model_dir": str(model_dir),
                    "ts": int(ts_used),
                    "Th_C": float(Th),
                    "Lstrike_km": float(Lstrike),
                    "h_km": float(h),
                    "L_int_km": float(L_int_km),
                    "Lmax_km": float(Lmax_km),
                    "Npatches": int(Npatches),
                    "Tmean_above_C": float(Tmean_above) if np.isfinite(Tmean_above) else np.nan,
                    "Ares_km2": float(Ares_km2),
                    "Vres_km3": float(Vres_km3),
                })

    return pd.DataFrame(rows)


def build_peak_table_one_model(
    model_dir: str,
    ts_list: Sequence[int],
    z_surf_km: float,
    depth_bsl_km: float,
    Th_list: Sequence[float],
    Lstrike_km_list: Sequence[float],
    h_km_list: Sequence[float],
    # receta volumétrica
    phi: float = 0.10,
    rho_r: float = 2700.0,
    c_r: float = 1000.0,
    rho_f: float = 1000.0,
    c_f: float = 4200.0,
    Tref_C: float = 15.0,
    Rf: float = 0.15,
    life_years: float = 30.0,
    # unidades / conversión
    tol_z_km: Optional[float] = None,
    temp_is_kelvin: bool = False,
    temp_offset: float = 273.15,
    m_to_km: float = 1.0,
    Trep_mode: str = "mean",
    # criterio de máximo
    peak_col: str = "Pe_MW_km2",
    # tiempo (para reportar t_peak_ka)
    time_unit: float = 1.0,
    to_ka: float = 1000.0,
) -> pd.DataFrame:
    """
    Computa "estados máximos" como máximo en el tiempo del criterio `peak_col`,
    para cada combinación (Th, Lstrike, h) de un único modelo (scenario).

    Devuelve un DF con la fila "óptima" por (Th, Lstrike, h), incluyendo:
      - ts_peak: timestep donde se alcanza el máximo
      - t_peak_ka: tiempo asociado (si hay timeField)
      - peak_of: nombre de la variable maximizada
    """
    coords = read_coords(model_dir)

    best_val: Dict[Tuple[float, float, float], float] = {}
    best_row: Dict[Tuple[float, float, float], Dict[str, Any]] = {}

    for ts in ts_list:
        df0 = build_potential_table_one_model_cached_coords(
            model_dir=str(model_dir),
            ts=int(ts),
            coords=coords,
            z_surf_km=z_surf_km,
            depth_bsl_km=depth_bsl_km,
            Th_list=Th_list,
            Lstrike_km_list=Lstrike_km_list,
            h_km_list=h_km_list,
            tol_z_km=tol_z_km,
            temp_is_kelvin=temp_is_kelvin,
            temp_offset=temp_offset,
            m_to_km=m_to_km,
            Trep_mode=Trep_mode,
        )

        df = apply_volumetric_recipe(
            df0,
            phi=phi, rho_r=rho_r, c_r=c_r, rho_f=rho_f, c_f=c_f,
            Tref_C=Tref_C, Rf=Rf, life_years=life_years
        )
        df = add_pe_area_norm(df, out_col="Pe_MW_km2")

        if peak_col not in df.columns:
            raise KeyError(f"peak_col='{peak_col}' no está en el DF. Disponibles: {sorted(df.columns)}")

        for r in df.to_dict(orient="records"):
            key = (float(r["Th_C"]), float(r["Lstrike_km"]), float(r["h_km"]))
            val = float(r.get(peak_col, 0.0)) if np.isfinite(r.get(peak_col, np.nan)) else 0.0

            if key not in best_val or val > best_val[key]:
                best_val[key] = val

                ts_peak = int(r.get("ts", ts))
                try:
                    t_peak_ka = float(load_time(model_dir, ts_peak, time_unit=time_unit) * float(to_ka))
                except Exception:
                    t_peak_ka = float("nan")

                row = r.copy()
                row["ts_peak"] = ts_peak
                row["t_peak_ka"] = t_peak_ka
                row["peak_of"] = str(peak_col)
                best_row[key] = row

    out = pd.DataFrame(list(best_row.values()))
    if "ts" in out.columns:
        out = out.rename(columns={"ts": "ts_used"})
    return out


def build_peak_table_all_models(
    model_dirs: Sequence[Any],
    z_surf_km: float,
    depth_bsl_km: float,
    Th_list: Sequence[float],
    Lstrike_km_list: Sequence[float],
    h_km_list: Sequence[float],
    # selección de timesteps
    step_every: Optional[int] = 5,
    ts_min: Optional[int] = None,
    ts_max: Optional[int] = None,
    skip: Optional[Iterable[int]] = None,
    # receta
    phi: float = 0.10,
    rho_r: float = 2700.0,
    c_r: float = 1000.0,
    rho_f: float = 1000.0,
    c_f: float = 4200.0,
    Tref_C: float = 15.0,
    Rf: float = 0.15,
    life_years: float = 30.0,
    # unidades
    tol_z_km: Optional[float] = None,
    temp_is_kelvin: bool = False,
    temp_offset: float = 273.15,
    m_to_km: float = 1.0,
    Trep_mode: str = "mean",
    # criterio de máximo
    peak_col: str = "Pe_MW_km2",
    # tiempo
    time_unit: float = 1.0,
    to_ka: float = 1000.0,
) -> pd.DataFrame:
    """
    Wrapper: corre build_peak_table_one_model() para una lista de escenarios (paths),
    y concatena resultados con metadata (family, scenario, param_*).
    """
    dfs: List[pd.DataFrame] = []
    skipped: List[Tuple[str, str, str]] = []

    for mp in model_dirs:
        mp_str = str(mp)
        try:
            ts_list = iter_timesteps(mp_str, step_every=step_every, ts_min=ts_min, ts_max=ts_max, skip=skip)
            if len(ts_list) == 0:
                raise RuntimeError("No hay timesteps tras aplicar filtros (ts_min/ts_max/step_every/skip).")

            dpk = build_peak_table_one_model(
                model_dir=mp_str,
                ts_list=ts_list,
                z_surf_km=z_surf_km,
                depth_bsl_km=depth_bsl_km,
                Th_list=Th_list,
                Lstrike_km_list=Lstrike_km_list,
                h_km_list=h_km_list,
                phi=phi, rho_r=rho_r, c_r=c_r, rho_f=rho_f, c_f=c_f,
                Tref_C=Tref_C, Rf=Rf, life_years=life_years,
                tol_z_km=tol_z_km,
                temp_is_kelvin=temp_is_kelvin,
                temp_offset=temp_offset,
                m_to_km=m_to_km,
                Trep_mode=Trep_mode,
                peak_col=peak_col,
                time_unit=time_unit,
                to_ka=to_ka,
            )

            dpk = add_metadata(dpk, mp)
            dfs.append(dpk)

        except Exception as e:
            skipped.append((mp_str, type(e).__name__, str(e)))

    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    out.attrs["skipped"] = skipped
    return out
