# plot_utils.py
# Utilidades para leer outputs de Underworld y armar perfiles/diagnósticos térmicos.
#
# Objetivo de esta versión:
# - Mantener la API existente y los resultados (figuras) sin cambios deliberados.
# - Reducir lecturas redundantes (mesh/coords) y repetir menos lógica.
# - Agregar helpers para seleccionar modelos por nombre sin estar comentando código.
#
# Unidades asumidas:
# - Coordenadas en metros en los HDF5 de mesh => convertimos a km (factor m_to_km).
# - Temperatura en Kelvin en HDF5 de temperatura, a menos que indiques lo contrario con temp_offset.
# - timeField en las unidades definidas por modeltime en tu Modelo (p.ej., Myr si usaste u.megayear).

from __future__ import annotations

import os
import re
import glob
import json
import pickle
import hashlib
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import h5py

__all__ = [
    # descubrimiento / selección
    "discover_model_dirs", "parse_model_name", "select_models", "make_label_map",
    # lectura / conteo
    "timestep_indices_in", "read_coords", "find_temperature_file", "load_xzT", "load_time",
    # perfiles “midline”
    "midline_mask", "binned_profile",
    # grilla + derivados
    "grid_T", "isotherm_depths_along_x", "compute_dTdz_and_heatflow", "hot_area_in_window",
    # series temporales
    "z_window_mask", "stress_second_invariant", "load_timeseries_for_model",
    "load_time_for_timestep", "load_timeseries_for_model_cached",
    # figuras rápidas (sin estilos)
    "plot_T_section", "plot_series_line", "plot_series_band", "plot_iso_depths",
    # suavizado
    "smooth_1d_nan_savgol", "smooth_1d_nan_median",
]

# ------------------------------------------------------------
# Helpers generales (cache / hashing)
# ------------------------------------------------------------

def _safe_json(obj: Any) -> Any:
    """Convierte cosas típicas (np types, sets) a algo json-serializable."""
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted(list(obj))
    if is_dataclass(obj):
        return _safe_json(asdict(obj))
    try:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except Exception:
        pass
    return obj

def _hash_params(params: dict) -> str:
    payload = json.dumps(_safe_json(params), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

def _model_fingerprint(model_dir: str) -> dict:
    """
    Fingerprint barato para detectar cambios:
    - último timestep disponible (por temperatura)
    - cantidad de timesteps (por temperatura)
    - mtime del último archivo de temperatura
    """
    temp_files = sorted(
        glob.glob(os.path.join(model_dir, "temperature-*.h5"))
        + glob.glob(os.path.join(model_dir, "temperatureField-*.h5"))
    )
    if not temp_files:
        return {"n": 0, "last": None, "mtime": None}

    last_file = temp_files[-1]
    base = os.path.basename(last_file)
    m = re.search(r"-(\d+)\.h5$", base)
    ts = int(m.group(1)) if m else base
    return {"n": len(temp_files), "last": ts, "mtime": os.path.getmtime(last_file)}

def _cache_path(model_dir: str, cache_dir: str, key: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(model_dir))
    return os.path.join(cache_dir, f"{tag}__{key}.pkl")

# ------------------------------------------------------------
# Descubrimiento / selección de modelos (para evitar comentar código)
# ------------------------------------------------------------

def discover_model_dirs(
    base_dir: str,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    ignore: Sequence[str] = (".ipynb_checkpoints",),
) -> List[str]:
    """
    Devuelve subcarpetas dentro de base_dir.

    - include/exclude: listas de patrones tipo glob (p.ej. ['Depth_*_512x512*']).
      Si include es None, toma todo.
    - ignore: carpetas a ignorar siempre.
    """
    all_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d not in set(ignore)
    ]
    if include:
        keep = set()
        for pat in include:
            keep.update(glob.fnmatch.filter(all_dirs, pat))
        all_dirs = [d for d in all_dirs if d in keep]
    if exclude:
        drop = set()
        for pat in exclude:
            drop.update(glob.fnmatch.filter(all_dirs, pat))
        all_dirs = [d for d in all_dirs if d not in drop]
    return sorted(all_dirs)

_MODEL_REGEXES = {
    # ejemplos típicos: Depth_-10.0_steady-state_512x512 / Temp_900 ...
    "depth": re.compile(r"(?:^|_)Depth_(-?\d+(?:\.\d+)?)"),
    "temp":  re.compile(r"(?:^|_)Temp_(\d{2,4})"),
    "res":   re.compile(r"(\d{2,4}x\d{2,4})"),
    "radius": re.compile(r"(?:^|_)R_(\d+(?:\.\d+)?)"),
}

def parse_model_name(name: str) -> Dict[str, Any]:
    """
    Intenta extraer metadata del nombre del modelo. No asume un naming único.

    Devuelve claves opcionales: depth (float), temp (int), res (str), radius (float).
    """
    out: Dict[str, Any] = {"name": name}
    m = _MODEL_REGEXES["depth"].search(name)
    if m:
        try: out["depth"] = float(m.group(1))
        except Exception: pass

    m = _MODEL_REGEXES["temp"].search(name)
    if m:
        try: out["temp"] = int(m.group(1))
        except Exception: pass

    m = _MODEL_REGEXES["res"].search(name)
    if m:
        out["res"] = m.group(1)

    m = _MODEL_REGEXES["radius"].search(name)
    if m:
        try: out["radius"] = float(m.group(1))
        except Exception: pass

    return out

def select_models(
    model_dirs: Sequence[str],
    temps: Optional[Sequence[int]] = None,
    depths: Optional[Sequence[Union[int, float]]] = None,
    res: Optional[Sequence[str]] = None,
    radius: Optional[Sequence[Union[int, float]]] = None,
    sort_by: Optional[str] = None,
) -> List[str]:
    """
    Filtra una lista de nombres de carpetas con reglas simples en base al nombre.

    - temps: [700, 750, ...] busca 'Temp_700' etc.
    - depths: [-10.0, -12.0] busca 'Depth_-10.0' etc (tolerancia por string, no numérica).
    - res: ['256x256', '512x512']
    - radius: [4,5] busca 'R_4', etc.
    - sort_by: 'depth'|'temp'|'res'|'radius' o None.

    Nota: para máxima previsibilidad, el filtro usa parsing por regex y, para depths,
    compara el valor float parseado (si existe).
    """
    temps_set = set(map(int, temps)) if temps else None
    res_set = set(res) if res else None

    depths_set = set(float(d) for d in depths) if depths else None
    radius_set = set(float(r) for r in radius) if radius else None

    kept = []
    meta = {}
    for d in model_dirs:
        m = parse_model_name(d)
        meta[d] = m

        if temps_set is not None:
            if m.get("temp") not in temps_set:
                continue
        if res_set is not None:
            if m.get("res") not in res_set:
                continue
        if depths_set is not None:
            if m.get("depth") is None or float(m["depth"]) not in depths_set:
                continue
        if radius_set is not None:
            if m.get("radius") is None or float(m["radius"]) not in radius_set:
                continue

        kept.append(d)

    if sort_by:
        key = sort_by.lower()
        def _k(name: str):
            v = meta[name].get(key)
            return (v is None, v)  # None al final
        kept = sorted(kept, key=_k)

    return kept

def make_label_map(
    model_dirs: Sequence[str],
    field: str = "temp",
    fmt: Optional[str] = None,
    fallback: str = "name",
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Construye label_map a partir del nombre del modelo.

    field:
      - 'temp'   -> '900°C'
      - 'depth'  -> '-10.0 km'
      - 'res'    -> '512x512'
      - 'radius' -> '4 km'
      - 'name'   -> nombre tal cual

    fmt permite customizar, p.ej. fmt='{temp} °C' o fmt='z_top={depth} km'
    overrides pisa cualquier key.
    """
    field = field.lower()
    overrides = overrides or {}
    out: Dict[str, str] = {}
    for d in model_dirs:
        m = parse_model_name(d)
        if field == "temp" and m.get("temp") is not None:
            lab = f"{m['temp']}°C"
        elif field == "depth" and m.get("depth") is not None:
            lab = f"{m['depth']} km"
        elif field == "res" and m.get("res") is not None:
            lab = str(m["res"])
        elif field == "radius" and m.get("radius") is not None:
            lab = f"{m['radius']} km"
        else:
            lab = d if fallback == "name" else str(m.get(fallback, d))

        if fmt:
            try:
                lab = fmt.format(**m)
            except Exception:
                pass

        out[d] = lab

    out.update(overrides)
    return out

# ------------------------------------------------------------
# Lectura básica y utilidades
# ------------------------------------------------------------

def timestep_indices_in(dirpath: str, patterns: Sequence[str] = ("temperature-*.h5", "temperatureField-*.h5")) -> List[int]:
    files: List[str] = []
    for pat in patterns:
        files += glob.glob(os.path.join(dirpath, pat))
    idxs: List[int] = []
    for p in files:
        m = re.search(r"-(\d+)\.h5$", p)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(set(idxs))

def _first_numeric_dataset(h5: h5py.File) -> Optional[np.ndarray]:
    """Devuelve el primer dataset numérico encontrado (heurística robusta)."""
    def walk(gr):
        for _, v in gr.items():
            if isinstance(v, h5py.Dataset) and v.dtype.kind in "fiu" and v.ndim > 0:
                return np.array(v[...])
            if isinstance(v, h5py.Group):
                w = walk(v)
                if w is not None:
                    return w
        return None
    return walk(h5)

@lru_cache(maxsize=32)
def read_coords(model_dir: str) -> np.ndarray:
    """Lee coords desde mesh.h5 (coordinates / vertices / centroids). Devuelve (N,2|3)."""
    mesh_path = os.path.join(model_dir, "mesh.h5")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Falta mesh.h5 en {model_dir}")
    with h5py.File(mesh_path, "r") as f:
        for key in ["coordinates", "/coordinates", "mesh/vertices", "/mesh/vertices",
                    "vertices", "centroids", "/centroids"]:
            if key in f:
                return np.array(f[key][...])

        # Fallback: primer dataset 2D/3D
        def walk2d(gr):
            for _, v in gr.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 2 and v.shape[1] in (2, 3):
                    return np.array(v[...])
                if isinstance(v, h5py.Group):
                    w = walk2d(v)
                    if w is not None:
                        return w
            return None

        arr = walk2d(f)
        if arr is None:
            raise RuntimeError("No encontré coords 2D/3D en mesh.h5")
        return arr

def _coords_xz_from_mesh(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae x,z desde coords 2D o 3D, siguiendo tu convención."""
    if P.shape[1] == 2:
        return P[:, 0], P[:, 1]
    return P[:, 0], P[:, 2]

def find_temperature_file(model_dir: str, timestep: int) -> Tuple[str, Optional[int]]:
    """Busca temperature-<i>.h5 o temperatureField-<i>.h5; si no hay exacto, toma el más cercano."""
    cands = sorted(
        glob.glob(os.path.join(model_dir, "temperature-*.h5"))
        + glob.glob(os.path.join(model_dir, "temperatureField-*.h5"))
    )
    if not cands:
        raise FileNotFoundError(f"No hay temperature*.h5 en {model_dir}")

    pairs: List[Tuple[int, str]] = []
    for p in cands:
        m = re.search(r"-(\d+)\.h5$", p)
        if m:
            pairs.append((int(m.group(1)), p))

    if not pairs:
        return cands[0], None

    idxs = np.array([i for i, _ in pairs])
    j = int(np.argmin(np.abs(idxs - timestep)))
    return pairs[j][1], int(idxs[j])

def load_xzT(
    model_dir: str,
    timestep: int,
    m_to_km: float = 1e-3,
    temp_offset: float = 0.0,
    coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]:
    """
    Devuelve x_km, z_km, T (ya con offset), ts_usado.

    - coords: si lo pasás, evita releer mesh.h5 (útil en loops).
    """
    P = read_coords(model_dir) if coords is None else np.asarray(coords)
    x_raw, z_raw = _coords_xz_from_mesh(P)

    t_path, ts_used = find_temperature_file(model_dir, timestep)
    with h5py.File(t_path, "r") as f:
        T = _first_numeric_dataset(f)
    if T is None:
        raise RuntimeError(f"No encontré dataset numérico en {t_path}")
    T = np.asarray(T).reshape(-1)

    n = min(len(x_raw), len(T))
    x_raw, z_raw, T = x_raw[:n], z_raw[:n], T[:n]

    x_km = x_raw * m_to_km
    z_km = z_raw * m_to_km
    T_val = T - temp_offset
    return x_km, z_km, T_val, ts_used

def load_time(model_dir: str, timestep: int, time_unit: float = 1.0) -> float:
    """
    Lee timeField-<timestep>.h5 (o projTimeField) y devuelve el tiempo escalar.
    """
    cands = sorted(
        glob.glob(os.path.join(model_dir, "timeField-*.h5"))
        + glob.glob(os.path.join(model_dir, "projTimeField-*.h5"))
    )
    if not cands:
        raise FileNotFoundError(f"No hay timeField-*.h5 en {model_dir}")

    pairs = []
    for p in cands:
        m = re.search(r"-(\d+)\.h5$", p)
        if m:
            pairs.append((int(m.group(1)), p))

    idxs = np.array([i for i, _ in pairs])
    j = int(np.argmin(np.abs(idxs - timestep)))
    chosen = pairs[j][1]

    with h5py.File(chosen, "r") as f:
        arr = _first_numeric_dataset(f)
    if arr is None:
        raise RuntimeError(f"No se pudo leer tiempo en {chosen}")
    return float(np.nanmean(arr)) * time_unit

# ------------------------------------------------------------
# Perfiles “midline”
# ------------------------------------------------------------

def midline_mask(x: np.ndarray, z: np.ndarray, tol_rel: float = 0.005) -> np.ndarray:
    """Máscara de puntos cercanos a x medio. tol_rel es fracción del ancho."""
    xmin, xmax = float(np.min(x)), float(np.max(x))
    xmid = 0.5 * (xmin + xmax)
    tol = tol_rel * (xmax - xmin)
    return np.abs(x - xmid) <= tol

def binned_profile(z: np.ndarray, val: np.ndarray, nbins: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    """Promedia 'val' en bins de z (en km). Devuelve z_centers y promedio."""
    zmin, zmax = float(np.min(z)), float(np.max(z))
    bins = np.linspace(zmin, zmax, nbins + 1)
    idx = np.digitize(z, bins) - 1
    prof = np.full(nbins, np.nan, dtype=float)
    for i in range(nbins):
        m = idx == i
        if np.any(m):
            prof[i] = float(np.nanmean(val[m]))
    zc = 0.5 * (bins[:-1] + bins[1:])
    return zc, prof

# ------------------------------------------------------------
# Grilla y diagnósticos
# ------------------------------------------------------------

def grid_T(x: np.ndarray, z: np.ndarray, T: np.ndarray, nx: int = 400, nz: int = 240, method: str = "linear"):
    """
    Interpola T(x,z) a una grilla regular. Devuelve (xi, zi, Ti).
    """
    from scipy.interpolate import griddata
    xi = np.linspace(x.min(), x.max(), nx)
    zi = np.linspace(z.min(), z.max(), nz)
    Xi, Zi = np.meshgrid(xi, zi)
    Ti = griddata((x, z), T, (Xi, Zi), method=method)
    return xi, zi, Ti

def compute_dTdz_and_heatflow(xi, zi, Ti, conductivity_WmK: float = 2.5):
    """Calcula dT/dz (°C/m) y q = -k dT/dz (W/m²) de forma robusta."""
    zi = np.asarray(zi)
    Ti = np.asarray(Ti)
    nz, nx = Ti.shape

    if nz < 2 or not np.isfinite(zi).all() or float(zi.max()) == float(zi.min()):
        dTdz = np.full_like(Ti, np.nan, dtype=float)
        q = np.full_like(Ti, np.nan, dtype=float)
        return dTdz, q

    dz_m = float(zi[1] - zi[0]) * 1000.0  # km -> m
    if dz_m <= 0 or not np.isfinite(dz_m):
        dTdz = np.full_like(Ti, np.nan, dtype=float)
        q = np.full_like(Ti, np.nan, dtype=float)
        return dTdz, q

    dTdz = np.full_like(Ti, np.nan, dtype=float)

    for j in range(nx):
        col = Ti[:, j]
        finite = np.isfinite(col)
        if np.count_nonzero(finite) < 3:
            continue
        zf = zi[finite]
        tf = col[finite]
        col_filled = np.interp(zi, zf, tf, left=np.nan, right=np.nan)
        dcol = np.gradient(col_filled, dz_m)
        dTdz[:, j] = dcol

    q = -conductivity_WmK * dTdz
    return dTdz, q

def hot_area_in_window(xi, zi, Ti, T_thresh: float = 250.0, z_min: float = 1.0, z_max: float = 3.0) -> float:
    """Área 2D (km²) donde T>=T_thresh dentro de la ventana [z_min, z_max] (km)."""
    mz = (zi >= z_min) & (zi <= z_max)
    if not np.any(mz) or Ti is None:
        return float("nan")
    hot = (Ti[mz, :] >= T_thresh)
    dx_km = np.gradient(xi)
    dz_km = np.gradient(zi[mz])
    area_km2 = float(np.nansum(hot * dz_km[:, None] * dx_km[None, :]))
    return area_km2

# ------------------------------------------------------------
# Plots rápidos
# ------------------------------------------------------------

def plot_T_section(xi, zi, Ti, vmin=None, vmax=None, title=None):
    """pcolormesh sin cmap explícito (dejar default). Devuelve fig, ax."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 3.6))
    ax = fig.add_subplot(111)
    pcm = ax.pcolormesh(xi, zi, Ti, shading="auto", vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel("Distancia (km)")
    ax.set_ylabel("Profundidad (km)")
    if title:
        ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label="Temperatura (°C)")
    fig.tight_layout()
    return fig, ax

# ------------------------------------------------------------
# Series temporales (evolución)
# ------------------------------------------------------------

def z_window_mask(z: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    """Máscara booleana para una ventana de profundidad [zmin, zmax] en km."""
    z = np.asarray(z)
    return (z >= zmin) & (z <= zmax)

def stress_second_invariant(S: np.ndarray) -> np.ndarray:
    """Segundo invariante de esfuerzo en 2D (esperado [sxx, syy, sxy])."""
    S = np.asarray(S)
    if S.ndim == 2 and S.shape[1] >= 3:
        sxx, syy, sxy = S[:, 0], S[:, 1], S[:, 2]
        tau2 = 0.5 * ((sxx - syy) ** 2 + sxx ** 2 + syy ** 2 + 6 * (sxy ** 2))
        return np.sqrt(np.clip(tau2, 0, None))
    return np.linalg.norm(S, axis=1) if S.ndim == 2 else np.abs(S)

def load_time_for_timestep(model_path: str, ts: int, prefer: str = "timeField") -> float:
    """Lee el tiempo del timestep ts desde timeField-ts.h5 o projTimeField-ts.h5."""
    if prefer == "timeField":
        candidates = [
            os.path.join(model_path, f"timeField-{ts}.h5"),
            os.path.join(model_path, f"projTimeField-{ts}.h5"),
        ]
    else:
        candidates = [
            os.path.join(model_path, f"projTimeField-{ts}.h5"),
            os.path.join(model_path, f"timeField-{ts}.h5"),
        ]

    for p in candidates:
        if os.path.exists(p):
            with h5py.File(p, "r") as f:
                t = _first_numeric_dataset(f)
            if t is None:
                continue
            t = np.asarray(t).reshape(-1)
            return float(t[0])
    raise FileNotFoundError(f"No encontré timeField/projTimeField para ts={ts} en {model_path}")

def load_timeseries_for_model(
    model_path: str,
    temp_offset: float = 273.15,
    m_to_km: float = 1.0,  # <-- CLAVE: tus coords ya están en km
    nx: int = 300,
    nz: int = 200,
    k_WmK: float = 2.5,
    z_win_mean: Tuple[float, float] = (0.0, 2.0),
    z_for_q: float = 0.5,
    isotherms: Sequence[Union[int, float]] = (100, 150, 200, 250),
    time_to_ka: Optional[float] = None,
    time_field_prefer: str = "timeField",
    # --- cámara ---
    x_cam: Optional[float] = None,
    dx_cam: float = 4.0,
    cam_stat: str = "mean",  # "mean" o "median"
    # --- submuestreo / filtros ---
    step_every: Optional[int] = None,
    ts_min: Optional[int] = None,
    ts_max: Optional[int] = None,
):
    """
    Carga diagnósticos para un modelo (serie temporal).

    Nota importante sobre unidades:
    - En tu caso, read_coords/_coords_xz_from_mesh ya devuelven x,z en km.
      Por eso m_to_km debe ser 1.0 (no 1e-3).
    - Si alguna vez tus coords vinieran en metros, el auto-check abajo
      las convierte a km usando m_to_km.
    """
    import os
    import numpy as np
    import h5py
    from scipy.interpolate import griddata

    idxs = timestep_indices_in(model_path)
    if not idxs:
        return None

    if ts_min is not None:
        idxs = [ts for ts in idxs if ts >= ts_min]
    if ts_max is not None:
        idxs = [ts for ts in idxs if ts <= ts_max]

    if step_every is not None and step_every > 1:
        full = idxs[:]
        idxs = full[::step_every]
        if full and idxs and idxs[-1] != full[-1]:
            idxs.append(full[-1])

    isotherms = [float(t) for t in list(isotherms)]

    results = {
        "time": [],
        "iso_depths": {float(Tiso): [] for Tiso in isotherms},
        "T_mean_0_2km": [],
        "q_at_z": [],
        "w_mean_0_2km": [],
        "logeta_mean_0_5km": [],
        "logeta_p10_0_5km": [],
        "logeta_p90_0_5km": [],
        "tauII_mean_0_5km": [],
        "iso_depths_cam": {float(Tiso): [] for Tiso in isotherms},
    }

    # --- coords (una vez) ---
    P = read_coords(model_path)
    x_raw, z_raw = _coords_xz_from_mesh(P)

    # --------- AUTO-CHECK UNIDADES (evita doble conversión) ----------
    # Si abs(z) > 100 -> probablemente metros => convertir a km usando m_to_km (típicamente 1e-3)
    # Si abs(z) <= 100 -> probablemente km => NO convertir (m_to_km debe ser 1.0 en tu caso)
    z_absmax = float(np.nanmax(np.abs(z_raw))) if np.size(z_raw) else np.nan
    if np.isfinite(z_absmax) and z_absmax > 100.0:
        # venían en metros
        x0 = x_raw * m_to_km
        z0 = z_raw * m_to_km
    else:
        # venían en km
        x0 = x_raw.copy()
        z0 = z_raw.copy()

    # --- grilla (una vez) ---
    xi = np.linspace(x0.min(), x0.max(), nx)
    zi = np.linspace(z0.min(), z0.max(), nz)
    Xi, Zi = np.meshgrid(xi, zi)

    # ventana somera (una vez)
    mwin_T = z_window_mask(z0, *z_win_mean)

    for ts in idxs:
        try:
            t_raw = load_time_for_timestep(model_path, ts, prefer=time_field_prefer)
        except Exception:
            t_raw = float(ts)

        t_out = t_raw if (time_to_ka is None) else (t_raw * time_to_ka)
        results["time"].append(t_out)

        # --- temperatura: leer solo el archivo del step ---
        t_path, _ = find_temperature_file(model_path, ts)
        with h5py.File(t_path, "r") as f:
            Tarr = _first_numeric_dataset(f)

        if Tarr is None:
            for Tiso in isotherms:
                results["iso_depths"][float(Tiso)].append(np.nan)
                results["iso_depths_cam"][float(Tiso)].append(np.nan)
            results["T_mean_0_2km"].append(np.nan)
            results["q_at_z"].append(np.nan)
            results["w_mean_0_2km"].append(np.nan)
            results["logeta_mean_0_5km"].append(np.nan)
            results["logeta_p10_0_5km"].append(np.nan)
            results["logeta_p90_0_5km"].append(np.nan)
            results["tauII_mean_0_5km"].append(np.nan)
            continue

        T = np.asarray(Tarr).reshape(-1) - temp_offset
        n = min(T.size, x0.size)
        x = x0[:n]; z = z0[:n]; T = T[:n]

        results["T_mean_0_2km"].append(float(np.nanmean(T[mwin_T[:n]])))

        # --- interpolación en grilla ---
        Ti = griddata((x, z), T, (Xi, Zi), method="linear")

        for Tiso in isotherms:
            z_iso = isotherm_depths_along_x(xi, zi, Ti, Tiso)
            results["iso_depths"][float(Tiso)].append(float(np.nanmean(z_iso)))

            if x_cam is None:
                results["iso_depths_cam"][float(Tiso)].append(np.nan)
            else:
                mcam = (xi >= (x_cam - dx_cam)) & (xi <= (x_cam + dx_cam))
                z_loc = z_iso[mcam] if np.any(mcam) else np.array([np.nan])
                if cam_stat == "median":
                    results["iso_depths_cam"][float(Tiso)].append(float(np.nanmedian(z_loc)))
                else:
                    results["iso_depths_cam"][float(Tiso)].append(float(np.nanmean(z_loc)))

        dTdz, q = compute_dTdz_and_heatflow(xi, zi, Ti, conductivity_WmK=k_WmK)
        jz = int(np.argmin(np.abs(zi - z_for_q)))
        results["q_at_z"].append(float(np.nanmean(q[jz, :])))

        # velocidad vertical
        vpath = os.path.join(model_path, f"velocityField-{ts}.h5")
        w_mean = np.nan
        if os.path.exists(vpath):
            with h5py.File(vpath, "r") as fv:
                V = _first_numeric_dataset(fv)
            if V is not None:
                V = np.asarray(V)
                if V.ndim == 1:
                    V = V.reshape(-1, 2)
                w = V[:, 1] if V.shape[1] >= 2 else np.full(len(V), np.nan)
                w_mean = float(np.nanmean(w[:n][mwin_T[:n]]))
        results["w_mean_0_2km"].append(w_mean)

        # viscosidad
        eta_path = os.path.join(model_path, f"projViscosityField-{ts}.h5")
        if os.path.exists(eta_path):
            with h5py.File(eta_path, "r") as fe:
                ETA = _first_numeric_dataset(fe)
            if ETA is not None:
                ETA = np.asarray(ETA).reshape(-1)
                nn = min(len(ETA), n)
                le = np.log10(np.maximum(ETA[:nn], 1e-30))
                z_clip = z[:nn]
                m05 = z_window_mask(z_clip, 0.0, 5.0)
                vals = le[m05]
                results["logeta_mean_0_5km"].append(float(np.nanmean(vals)))
                results["logeta_p10_0_5km"].append(float(np.nanpercentile(vals, 10)))
                results["logeta_p90_0_5km"].append(float(np.nanpercentile(vals, 90)))
            else:
                results["logeta_mean_0_5km"].append(np.nan)
                results["logeta_p10_0_5km"].append(np.nan)
                results["logeta_p90_0_5km"].append(np.nan)
        else:
            results["logeta_mean_0_5km"].append(np.nan)
            results["logeta_p10_0_5km"].append(np.nan)
            results["logeta_p90_0_5km"].append(np.nan)

        # esfuerzo II
        s_path = os.path.join(model_path, f"projStressField-{ts}.h5")
        if os.path.exists(s_path):
            with h5py.File(s_path, "r") as fs:
                S = _first_numeric_dataset(fs)
            if S is not None:
                nn = min(n, len(S))
                tauII = stress_second_invariant(np.asarray(S)[:nn])
                results["tauII_mean_0_5km"].append(float(np.nanmean(tauII[z[:nn] <= 5.0])))
            else:
                results["tauII_mean_0_5km"].append(np.nan)
        else:
            results["tauII_mean_0_5km"].append(np.nan)

    # ordenar por tiempo
    order = np.argsort(np.asarray(results["time"], dtype=float))
    for k in list(results.keys()):
        if k in ("iso_depths", "iso_depths_cam"):
            for Tiso in results[k]:
                arr = np.asarray(results[k][Tiso], dtype=float)[order]
                results[k][Tiso] = arr.tolist()
        else:
            results[k] = (np.asarray(results[k], dtype=float)[order]).tolist()

    return results


def load_timeseries_for_model_cached(
    model_dir: str,
    cache_dir: Optional[str] = None,
    force: bool = False,
    **kwargs
):
    """
    Cachea el resultado de load_timeseries_for_model.
    - cache_dir: carpeta donde guardar pkl (por defecto <model_dir>/.cache_timeseries)
    - force: recalcula aunque exista cache válido
    - kwargs: mismos argumentos que load_timeseries_for_model
    """
    if cache_dir is None:
        cache_dir = os.path.join(model_dir, ".cache_timeseries")

    params_key = _hash_params(kwargs)
    cache_file = _cache_path(model_dir, cache_dir, params_key)
    fp = _model_fingerprint(model_dir)

    if (not force) and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                obj = pickle.load(f)
            if obj.get("fingerprint") == fp and obj.get("params_key") == params_key:
                return obj["series"]
        except Exception:
            pass

    series = load_timeseries_for_model(model_dir, **kwargs)

    payload = {"fingerprint": fp, "params_key": params_key, "series": series}
    with open(cache_file, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return series

# ------------------------------------------------------------
# Suavizado 1D con NaNs
# ------------------------------------------------------------

from scipy.signal import savgol_filter, medfilt

def smooth_1d_nan_savgol(y, window: int = 11, poly: int = 2):
    """
    Suaviza y(t) con Savitzky–Golay manejando NaNs:
    - interpola NaNs solo dentro del rango finito
    - aplica savgol
    - vuelve a poner NaNs fuera del rango finito original
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return y.copy()

    m = np.isfinite(y)
    if m.sum() < 3:
        return y.copy()

    x = np.arange(n)
    y_filled = y.copy()

    i0, i1 = np.where(m)[0][0], np.where(m)[0][-1]
    y_filled[i0:i1+1] = np.interp(x[i0:i1+1], x[m], y[m])

    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y.copy()

    poly = min(int(poly), w - 1)
    ys = savgol_filter(y_filled, window_length=w, polyorder=poly, mode="interp")

    ys[:i0] = np.nan
    ys[i1+1:] = np.nan
    return ys

def smooth_1d_nan_median(y, k: int = 7):
    """Mediana móvil (mata picos). Maneja NaNs interpolando dentro del rango finito."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return y.copy()

    m = np.isfinite(y)
    if m.sum() < 3:
        return y.copy()

    x = np.arange(n)
    y_filled = y.copy()
    i0, i1 = np.where(m)[0][0], np.where(m)[0][-1]
    y_filled[i0:i1+1] = np.interp(x[i0:i1+1], x[m], y[m])

    kk = int(k)
    if kk % 2 == 0:
        kk += 1
    kk = min(kk, n if n % 2 == 1 else n - 1)
    if kk < 3:
        return y.copy()

    ys = medfilt(y_filled, kernel_size=kk)
    ys[:i0] = np.nan
    ys[i1+1:] = np.nan
    return ys

# ------------------------------------------------------------
# Funciones de apoyo para ploteo (resolución de labels / leyendas)
# ------------------------------------------------------------

def resolve_label(name: str, label_map: Optional[Dict[str, str]] = None) -> str:
    """
    Resuelve un label 'lindo' usando:
    1) match exacto
    2) match por prefijo
    3) match por substring
    """
    label_map = label_map or {}
    if name in label_map:
        return label_map[name]
    for k, v in label_map.items():
        if name.startswith(k):
            return v
    for k, v in label_map.items():
        if k in name:
            return v
    return name

def plot_series_line(
    series_dict: Dict[str, dict],
    key: str,
    ylabel: str,
    title: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
    legend_loc: str = "none",   # "none", "inside", "right", "bottom"
    fig_kwargs: Optional[dict] = None,
    ax=None,
    grid: bool = True,
    panel_label: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    created_fig = False
    if ax is None:
        fig = plt.figure(**(fig_kwargs or {"figsize": (6, 4)}))
        ax = fig.add_subplot(111)
        created_fig = True
    else:
        fig = ax.figure

    for name, s in series_dict.items():
        ax.plot(s["time"], s[key], label=resolve_label(name, label_map), lw=2)

    ax.set_xlabel("Tiempo (ka)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if grid:
        ax.grid(True, ls="--", alpha=0.35)

    if panel_label is not None:
        ax.text(0.94, 0.98, panel_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold")

    if legend_loc == "inside":
        ax.legend(frameon=False)
    elif legend_loc == "right":
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    elif legend_loc == "bottom":
        # leyenda común debajo del eje
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=5, frameon=False)
    else:
        pass

    if created_fig:
        fig.tight_layout()

    return fig, ax

def plot_series_band(
    series_dict: Dict[str, dict],
    mean_key: str,
    p10_key: str,
    p90_key: str,
    ylabel: str,
    title: Optional[str] = None,
    legend_out: bool = True,
    fig_kwargs: Optional[dict] = None,
):
    """Igual que plot_series_line pero con banda p10–p90."""
    import matplotlib.pyplot as plt
    fig = plt.figure(**(fig_kwargs or {"figsize": (6, 4)}))
    ax = fig.add_subplot(111)
    for name, s in series_dict.items():
        t = np.asarray(s["time"])
        m = np.asarray(s[mean_key])
        p10 = np.asarray(s[p10_key])
        p90 = np.asarray(s[p90_key])
        ax.plot(t, m, label=name)
        ax.fill_between(t, p10, p90, alpha=0.15)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend_out:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
    else:
        ax.legend(frameon=False)
        fig.tight_layout()
    return fig, ax

# ------------------------------------------------------------
# Isotermas
# ------------------------------------------------------------

def isotherm_depths_along_x(xi, zi, Ti, Tiso):
    """
    Profundidad/cota (en las unidades de `zi`) de la isoterma Tiso(x),
    recorriendo SIEMPRE desde la "superficie" (borde más frío) hacia abajo (más caliente).

    Esto evita errores cuando la convención de `z` cambia (z positivo hacia arriba o hacia abajo).

    Parameters
    ----------
    xi, zi : 1D arrays
        Grilla en x y z.
    Ti : 2D array (nz, nx)
        Temperatura interpolada en la grilla (Zi, Xi).
    Tiso : float
        Umbral de isoterma.

    Returns
    -------
    z_iso : 1D array (nx,)
        z donde se cruza Tiso en cada x. NaN si no hay cruce.
    """
    import numpy as np

    xi = np.asarray(xi)
    zi = np.asarray(zi)
    Ti = np.asarray(Ti)

    z_iso = np.full_like(xi, np.nan, dtype=float)

    if Ti.ndim != 2 or Ti.shape[0] != zi.size:
        raise ValueError(f"Ti debe ser (nz, nx) = ({zi.size}, {xi.size}), pero es {Ti.shape}")

    # ---------- 1) auto-detectar "superficie" como el borde más frío ----------
    # Tomo 1 fila cerca de cada extremo (más estable que usar toda la fila)
    i_top = int(np.nanargmax(zi))   # índice donde z es máximo
    i_bot = int(np.nanargmin(zi))   # índice donde z es mínimo

    T_top = np.nanmean(Ti[i_top, :])
    T_bot = np.nanmean(Ti[i_bot, :])

    # Superficie = borde más frío
    surface_is_top = np.isfinite(T_top) and np.isfinite(T_bot) and (T_top <= T_bot)

    # ---------- 2) orden de barrido desde superficie hacia profundidad ----------
    # Si la superficie está en zi.max -> recorrer z decreciente
    # Si la superficie está en zi.min -> recorrer z creciente
    if surface_is_top:
        order = np.argsort(zi)[::-1]   # desde z grande (superficie) a z chico
    else:
        order = np.argsort(zi)         # desde z chico (superficie) a z grande

    z_sorted = zi[order]

    # ---------- 3) buscar primer cruce (de frío a caliente) en cada columna ----------
    for j in range(Ti.shape[1]):
        col = Ti[:, j]
        if np.all(~np.isfinite(col)):
            continue

        col_sorted = col[order]
        finite = np.isfinite(col_sorted)
        if np.count_nonzero(finite) < 2:
            continue

        zf = z_sorted[finite]
        tf = col_sorted[finite]

        # Si ya estamos por encima de la isoterma en la superficie,
        # la isoterma está en superficie (o no está definida hacia arriba).
        if tf[0] >= Tiso:
            z_iso[j] = zf[0]
            continue

        # buscar primer cruce ascendente
        k = np.where((tf[:-1] < Tiso) & (tf[1:] >= Tiso))[0]
        if k.size == 0:
            z_iso[j] = np.nan
            continue

        i = k[0]
        z1, z2 = zf[i], zf[i + 1]
        t1, t2 = tf[i], tf[i + 1]

        if np.isfinite(t1) and np.isfinite(t2) and (t2 - t1) != 0:
            z_iso[j] = z1 + (Tiso - t1) * (z2 - z1) / (t2 - t1)
        else:
            z_iso[j] = z2

    return z_iso



def plot_iso_depths(
    series_dict: Dict[str, dict],
    Tisos: Union[Sequence[Union[int, float]], int, float, str],
    ncols: int = 2,
    figsize=(8, 6),
    legend_ncol: int = 3,
    xlim=None,
    ylim=None,
    label_map: Optional[Dict[str, str]] = None,
    sort_legend: bool = False,
    y_transform=None,  # opcional: función y -> y'
    surface_m: float = 4300.0,           # superficie (m s.n.m.)
    autoscale_if_too_small: bool = True, # heurística para arreglar z_km 1000× chico
):
    """
    Grilla de "distancia bajo superficie" para isotermas vs tiempo.

    Convención (como tu figura buena):
      y(m) = z_iso(km)*1000 - surface_m
      (distancia relativa a la superficie, negativa hacia abajo)

    Fixes incluidos:
    - elimina offset/científico del eje Y (no más -4.296e3).
    - si z_km viene 1000× más chico (p.ej. ~0.003 km en vez de ~3.7 km),
      reescala automáticamente por 1000.
    - evita figuras "vacías": si ylim es None, decide entre un rango estándar
      (-4000,-500) o un rango automático en base a datos.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil
    import string
    from matplotlib.ticker import ScalarFormatter

    label_map = label_map or {}

    # Normalizar Tisos a lista de floats
    if isinstance(Tisos, (int, float, str)):
        Tisos = [Tisos]
    Tisos = [float(t) for t in list(Tisos)]

    # Chequeo isotermas faltantes
    missing = []
    for Tiso in Tisos:
        for name, s in series_dict.items():
            if "iso_depths" not in s or float(Tiso) not in s["iso_depths"]:
                missing.append((name, Tiso))
    if missing:
        msg = "\n".join([f"  {m[0]} (T={m[1]}°C)" for m in missing])
        raise KeyError(f"Isotermas faltantes:\n{msg}")

    nrows = ceil(len(Tisos) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).flatten()

    panel_letters = list(string.ascii_lowercase)
    all_t = []
    all_y = []  # para decidir ylim si hace falta

    for i, (ax, Tiso) in enumerate(zip(axes, Tisos)):
        plotted_any = False

        for name, s in series_dict.items():
            t = np.asarray(s.get("time", []), dtype=float)
            z_km = np.asarray(s["iso_depths"][float(Tiso)], dtype=float)

            if t.size == 0 or z_km.size == 0:
                continue
            if np.all(~np.isfinite(z_km)):
                continue

            # ---------- AUTO-FIX UNIDADES ----------
            # Esperable si es km s.n.m: ~3–6 km (porque surface=4.3 km)
            # Si viene ~0.003–0.006, está 1000× chico -> multiplicar por 1000.
            if autoscale_if_too_small:
                med = float(np.nanmedian(np.abs(z_km[np.isfinite(z_km)])))
                if 0 < med < 0.1:
                    z_km = z_km * 1000.0

            # distancia bajo superficie (m), negativa hacia abajo
            y = z_km * 1000.0 - float(surface_m)

            if y_transform is not None:
                y = y_transform(y)

            if np.all(~np.isfinite(y)):
                continue

            ax.plot(t, y, label=resolve_label(name, label_map))
            all_t.append(t)
            all_y.append(y)
            plotted_any = True

        ax.set_title(f"{Tiso:.0f} °C")
        ax.grid(True, ls="--", alpha=0.3)

        # ---------- FIX OFFSET “-4.296e3” ----------
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_visible(False)

        ax.text(
            0.98, 0.98, f"{panel_letters[i]})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, fontweight="bold"
        )

        if not plotted_any:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    # apagar axes sobrantes
    for ax in axes[len(Tisos):]:
        ax.axis("off")

    # xlim auto + ticks estándar
    if xlim is None and all_t:
        tmin = float(np.nanmin(np.concatenate(all_t)))
        tmax = float(np.nanmax(np.concatenate(all_t)))
        for ax in axes[:len(Tisos)]:
            ax.set_xlim((tmin, tmax))
            ax.set_xticks(np.arange(0, 1001, 200))

    # ---------- Decidir ylim si NO lo pasaron ----------
    if ylim is None and all_y:
        yy = np.concatenate([np.asarray(v, float) for v in all_y])
        yy = yy[np.isfinite(yy)]
        if yy.size:
            ymin, ymax = float(np.nanmin(yy)), float(np.nanmax(yy))

            # Si cae en el orden “esperado” (tipo -4000 a -500), usamos el estándar.
            # Si no, usamos un rango automático (para no dejar la figura vacía).
            if (ymin < -500) and (ymax > -4500):
                target = (-4000, -500)
            else:
                pad = 0.05 * (ymax - ymin) if ymax > ymin else 50.0
                target = (ymin - pad, ymax + pad)

            for ax in axes[:len(Tisos)]:
                ax.set_ylim(target)

    fig.text(0.5, 0.1, "Tiempo (ka)", ha="center")
    fig.text(-0.04, 0.5, "Distancia bajo superficie (m)", va="center", rotation="vertical")

    # leyenda común
    handles, labels = [], []
    for ax in axes[:len(Tisos)]:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    if sort_legend and labels:
        pairs = sorted(zip(labels, handles), key=lambda p: p[0])
        labels, handles = zip(*pairs)

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=legend_ncol,
                   frameon=False, bbox_to_anchor=(0.5, 0.03))

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    return fig, axes




    
# ------------------------------------------------------------
# Timesteps comunes + selección por ka (helpers)
# ------------------------------------------------------------

def common_timesteps(model_dirs: Sequence[Union[str, os.PathLike]]) -> List[int]:
    """Devuelve la intersección de timesteps disponibles en todos los modelos.

    Parameters
    ----------
    model_dirs
        Lista de directorios de modelos (paths). Cada uno debe contener outputs con índices de timestep.

    Returns
    -------
    list[int]
        Timesteps comunes (ordenados).
    """
    sets: List[set] = []
    for mdir in model_dirs:
        idxs = timestep_indices_in(str(mdir))
        sets.append(set(idxs))
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def select_timesteps_by_targets_ka(
    model_ref_dir: Union[str, os.PathLike],
    ts_list: Sequence[int],
    targets_ka: Sequence[float],
    tol_ka: float = 1.0,
    time_unit: float = 1.0,
) -> Tuple[List[int], Dict[int, float]]:
    """Selecciona timesteps cuyos tiempos físicos estén cerca de una lista objetivo (ka).

    Usa `model_ref_dir` como referencia para mapear `ts -> t_ka` (igual que tu workflow original).

    Returns
    -------
    selected_ts : list[int]
        Timesteps seleccionados (sin duplicados, preservando el orden de targets_ka).
    ts_to_tka : dict[int, float]
        Mapeo completo para los ts_list provistos.
    """
    ts_list = list(ts_list)
    if not ts_list:
        return [], {}

    ts_to_tka: Dict[int, float] = {}
    for ts in ts_list:
        t_ka = load_time(str(model_ref_dir), int(ts), time_unit=time_unit) * 1000.0
        ts_to_tka[int(ts)] = float(t_ka)

    selected: List[int] = []
    for target in targets_ka:
        best_ts = min(ts_list, key=lambda ts: abs(ts_to_tka[int(ts)] - float(target)))
        if abs(ts_to_tka[int(best_ts)] - float(target)) <= float(tol_ka):
            selected.append(int(best_ts))
        else:
            # Mantengo el print (no excepción) para replicar comportamiento de notebook.
            print(
                f"AVISO: no encontré timestep cerca de {target} ka "
                f"(mejor: {ts_to_tka[int(best_ts)]:.2f} ka, ts={int(best_ts)})"
            )

    # eliminar duplicados preservando orden
    selected = list(dict.fromkeys(selected))
    return selected, ts_to_tka


def plot_profiles_grid(
    model_dirs: Sequence[Union[str, os.PathLike]],
    label_map: Optional[Dict[str, str]] = None,
    *,
    select_ka: Sequence[float] = (20, 40, 100, 200, 300, 400),
    tol_ka: float = 1.0,
    ts_min: Optional[int] = None,
    ts_max: Optional[int] = None,
    step_every: Optional[int] = None,
    skip: Optional[Sequence[int]] = None,
    tol_x: float = 0.01,
    nbins: int = 120,
    ncols: int = 3,
    m_to_km: float = 1e-3,
    temp_offset: float = 273.15,
    xticks: Optional[np.ndarray] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Tuple[float, float] = (-25, 4.5),
    y_scale: float = 1000.0,
    legend_ncol: int = 8,
    suptitle: str = "Perfiles de temperatura vs. profundidad (x = 15 km)",
    base_dir=None,
):
    """Reproduce la figura de perfiles T(z) en una grilla de subplots (un panel por timestep).

    Esta función encapsula la celda que venías usando en el notebook:
    - Intersección de timesteps comunes.
    - Selección por tiempos físicos (ka) con tolerancia.
    - Para cada timestep: perfil midline + binning + plot por modelo.
    - Una sola leyenda global.

    Nota sobre compatibilidad:
    - Para preservar el look de tus figuras previas, por defecto mantiene `y_scale=1000.0` y `ylim=(-25, 4.5)`,
      igual que tu snippet original (aunque la unidad del eje puede depender de tu setup).
    """
    import matplotlib.pyplot as plt
    from math import ceil
    import string
    import os
    from pathlib import Path

    # --- normalizar model_dirs a paths existentes ---
    mdirs = []
    for d in model_dirs:
        p = Path(d)
        if not p.is_dir() and base_dir is not None:
            p = Path(base_dir) / str(d)
        mdirs.append(p)
    
    # chequeo rápido + mensaje útil
    bad = [str(p) for p in mdirs if not p.is_dir()]
    if bad:
        raise RuntimeError(
            "Hay modelos que no existen como directorio. "
            "Si estás pasando nombres, usá base_dir=... o pasá paths completos.\n"
            f"Ejemplos inválidos: {bad[:5]}"
        )
    
    model_dirs = mdirs
    if label_map is None:
        label_map = {p: os.path.basename(p) for p in model_dirs}
    else:
        # normalizo keys a str para evitar mismatch con Path
        label_map = {str(k): v for k, v in label_map.items()}

    # 1) timesteps comunes
    common_ts = common_timesteps(model_dirs)

    # 2) filtros opcionales
    if ts_min is not None:
        common_ts = [ts for ts in common_ts if ts >= int(ts_min)]
    if ts_max is not None:
        common_ts = [ts for ts in common_ts if ts <= int(ts_max)]
    if step_every is not None:
        common_ts = common_ts[:: int(step_every)]
    if skip:
        skip_set = set(int(s) for s in skip)
        common_ts = [ts for ts in common_ts if ts not in skip_set]

    if not common_ts:
        raise RuntimeError("No hay timesteps comunes entre modelos (o filtraste todo).")

    # 3) selección por ka (usando el primer modelo como referencia)
    sel_ts, ts_to_tka = select_timesteps_by_targets_ka(
        model_dirs[0], common_ts, select_ka, tol_ka=tol_ka, time_unit=1.0
    )
    if not sel_ts:
        raise RuntimeError("No se pudieron seleccionar timesteps por ka con la tolerancia dada.")

    # 4) grilla
    nrows = ceil(len(sel_ts) / int(ncols))
    fig, axes = plt.subplots(
        nrows, int(ncols),
        figsize=(3.2*int(ncols), 6*nrows),
        sharex=False, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    panel_letters = list(string.ascii_lowercase)

    legend_handles = None
    legend_labels  = None

    if xticks is None:
        xticks = np.arange(0, 1201, 200)

    for i, ts in enumerate(sel_ts):
        ax = axes[i]

        for mdir in model_dirs:
            x, z, T, ts_used = load_xzT(
                mdir, ts,
                m_to_km=m_to_km,
                temp_offset=temp_offset
            )
            mask = midline_mask(x, z, tol_rel=tol_x)
            z_line, T_line = z[mask], T[mask]
            zc, T_prof = binned_profile(z_line, T_line, nbins=nbins)

            name = str(mdir)
            label = resolve_label(name, label_map)  # usa exact/
            ax.plot(T_prof, y_scale * zc, label=label)

        ax.set_title(f"{ts_to_tka[int(ts)]:.0f} ka")

        ax.text(
            0.91, 0.98,
            f"{panel_letters[i]})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12,
            fontweight="bold",
        )

        if i % int(ncols) == 0:
            ax.set_ylabel("Profundidad (km)")
        if i // int(ncols) == nrows - 1:
            ax.set_xlabel("Temperatura (°C)")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_xticks(xticks)
        ax.set_ylim(*ylim)
        if xlim is not None:
            ax.set_xlim(*xlim)

    # apagar paneles vacíos
    for j in range(len(sel_ts), len(axes)):
        axes[j].axis("off")

    # leyenda común afuera
    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=int(legend_ncol),
            frameon=False,
        )

    fig.suptitle(suptitle, y=1.02)

    # quitar xlabel individuales y poner uno común
    for ax in axes:
        ax.set_xlabel("")
    fig.supxlabel("Temperatura (°C)", y=0.01)

    fig.tight_layout()
    return fig, axes, sel_ts, ts_to_tka



import os
import numpy as np
import h5py

def h5_read_first(path, keys=("data", "vertices", "coordinates")):
    with h5py.File(path, "r") as f:
        for k in keys:
            if k in f:
                return np.array(f[k])
        raise KeyError(f"No encontré {keys} en {path}. Keys: {list(f.keys())}")

def file_for_ts(model_dir: str, pattern: str, ts: int) -> str:
    """
    pattern ejemplo: "velocityField-{ts}.h5"
    """
    return os.path.join(model_dir, pattern.format(ts=int(ts)))

def read_velocity_mesh(model_dir: str, ts: int, pattern="velocityField-{ts}.h5"):
    """
    Devuelve (u,w) en el orden de vertices de mesh.
    Soporta layouts comunes: (N,2), (N,3), (2,N), (3,N), flattened.
    """
    path = file_for_ts(model_dir, pattern, ts)
    V = h5_read_first(path, keys=("data",)).astype(float)

    if V.ndim == 2:
        # (N,2) o (N,3)
        if V.shape[0] >= 2 and V.shape[1] in (2, 3):
            # OJO: esto sería (N,2) solo si V.shape[0]==N. No lo sabemos todavía.
            pass

        # Caso 1: (N,2) o (N,3)
        if V.shape[1] in (2, 3):
            u = V[:, 0]
            w = V[:, -1]
            return u, w

        # Caso 2: (2,N) o (3,N)
        if V.shape[0] in (2, 3):
            u = V[0, :]
            w = V[-1, :]
            return u, w

    # Caso 3: flattened
    Vf = V.ravel()
    if Vf.size % 2 == 0:
        Vr = Vf.reshape(-1, 2)
        return Vr[:, 0], Vr[:, 1]

    raise RuntimeError(f"Layout de velocidad no soportado: shape={V.shape} en {path}")


import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors

def plot_section_projmat_isotherms_velocity_ts(
    model_dir: str,
    ts: int,
    isotherms=(100,150,200,250),
    mesh_pattern="mesh.h5",
    temp_pattern="temperature-{ts}.h5",
    projmat_pattern="projMaterialField-{ts}.h5",
    vel_pattern="velocityField-{ts}.h5",
    temp_offset=273.15,
    cmap_material="Spectral_r",
    quiver_step=12,
    quiver_scale=1e-5,
    quiver_width=0.0030,
    quiver_alpha=1.0,
    iso_color="k",
    iso_lw=1.5,
    iso_alpha=1.0,
    iso_label=True,
    title=None,
    xlim=None,
    ylim=None,
    figsize=(7,6),
):
    # --- mesh coords ---
    mesh_xy = h5_read_first(os.path.join(model_dir, mesh_pattern), keys=("vertices","data","coordinates"))
    x_raw = mesh_xy[:, 0].astype(float)
    z_raw = mesh_xy[:, -1].astype(float)

    # detectar si coords están en m o km
    Lx = float(np.nanmax(x_raw) - np.nanmin(x_raw))
    to_km = 1e-3 if Lx > 1000.0 else 1.0

    x = x_raw * to_km
    z = z_raw * to_km

    # convención: z negativa hacia abajo
    if np.nanmedian(z) > 0:
        z = -z

    tri = mtri.Triangulation(x, z)

    # --- temperature (mismo ts) ---
    T = h5_read_first(file_for_ts(model_dir, temp_pattern, ts), keys=("data",)).ravel().astype(float)
    if np.nanmin(T) > 150.0:
        T = T - temp_offset

    # --- proj material (mismo ts) ---
    matp = h5_read_first(file_for_ts(model_dir, projmat_pattern, ts), keys=("data",)).ravel()
    matp_i = np.rint(matp).astype(int)

    if matp_i.size != x.size:
        raise RuntimeError(
            f"projMaterial size ({matp_i.size}) != n_vertices mesh ({x.size}). "
            "Si está cell-centered hay que adaptar."
        )

    ids = np.unique(matp_i)
    ids = np.sort(ids)
    levels = np.arange(ids.min() - 0.5, ids.max() + 1.5, 1.0)
    cmap = plt.get_cmap(cmap_material, len(ids))
    norm = mcolors.BoundaryNorm(levels, ncolors=len(ids))

    # --- velocity (mismo ts) ---
    u, w = read_velocity_mesh(model_dir, ts, pattern=vel_pattern)

    # si coords a km, vel a km en el mismo factor (consistencia geométrica)
    u = u * to_km
    w = w * to_km

    # submuestreo quiver
    idx = np.arange(0, x.size, max(1, int(quiver_step)))
    while idx.size > 3000:
        quiver_step = int(np.ceil(quiver_step * 1.3))
        idx = np.arange(0, x.size, quiver_step)

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    ax.tricontourf(tri, matp_i, levels=levels, cmap=cmap, norm=norm)

    Q = ax.quiver(
        x[idx], z[idx], u[idx], w[idx],
        color="w",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=quiver_width,
        alpha=quiver_alpha
    )

    iso_vals = [float(v) for v in isotherms]
    cs = ax.tricontour(tri, T, levels=iso_vals, colors=iso_color, linewidths=iso_lw, alpha=iso_alpha)

    if iso_label:
        fmt = {lv: f"{int(lv)}°C" for lv in iso_vals}
        ax.clabel(cs, cs.levels, inline=True, fontsize=11, fmt=fmt)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Z (km)")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_aspect("equal", adjustable="box")

    if title is not None:
        ax.set_title(title)

    ax.invert_yaxis()    

    return fig, ax
