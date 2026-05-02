# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:14:07 2026

@author: kamal.kanyal
"""

# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   NDVI GT Cluster Cleaner  v1.0                                            ║
║   Ground Truth cleaning for multi-temporal Sentinel-2 crop classification  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WORKFLOW:                                                                  ║
║  1. Load CSV/XLSX  →  pick UID, Lat, Lon, WKT, Crop columns               ║
║  2. Pick raster folder (one GeoTIFF per date) + Red/NIR band indices       ║
║  3. Extract NDVI time-series per GT row (polygon median or centroid)       ║
║  4. Auto-flag bad rows (flat curve / no crop signal)                       ║
║  5. Cluster each crop by NDVI time-series (K-means on full curve vector)   ║
║  6. Review clusters, delete bad GTs, rename cluster labels                 ║
║  7. Export cleaned CSV/XLSX with Crop_Cluster column                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  REQUIREMENTS:                                                              ║
║    pip install pandas openpyxl rasterio pyproj shapely scikit-learn        ║
║                matplotlib                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import glob
import json
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Optional heavy deps — warn on import but don't crash ─────────────────────
try:
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

try:
    from pyproj import Transformer
    PYPROJ_OK = True
except ImportError:
    PYPROJ_OK = False

try:
    from shapely import wkt as shapely_wkt
    from shapely.geometry import mapping
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR TOKENS  (matches v3.1 palette)
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":        "#F7F9FC",
    "panel":     "#FFFFFF",
    "border":    "#D6DFE8",
    "acc":       "#1A6FBF",
    "acc_hover": "#145999",
    "acc_light": "#EAF2FB",
    "success":   "#1E7D4E",
    "warn":      "#B85C00",
    "danger":    "#C0392B",
    "danger_bg": "#FDECEA",
    "text":      "#1C2B3A",
    "sub":       "#4E6478",
    "row_even":  "#F2F6FB",
    "row_odd":   "#FFFFFF",
    "del_bg":    "#FFF3E0",
    "highlight": "#D4365A",
}

PALETTE = ["#1A6FBF","#E05C1A","#1E7D4E","#8B3DBF",
           "#B8860B","#1A9BA8","#D4365A","#556B2F",
           "#E67E22","#2980B9","#8E44AD","#27AE60"]

FONT_HEAD  = ("Segoe UI", 12, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)

STEP_LABELS = ["Load GT", "Raster & NDVI", "Auto-Flag", "Cluster", "Review & Export"]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════
class GTModel:
    """All data logic — no Tkinter references here."""

    def __init__(self):
        self.df          = None
        self.filepath    = ""
        # column names chosen by user
        self.uid_col     = ""
        self.lat_col     = ""
        self.lon_col     = ""
        self.wkt_col     = ""
        self.crop_col    = ""
        # raster config
        self.raster_dir  = ""
        self.red_band    = 1
        self.nir_band    = 2
        self.raster_files: list = []   # sorted list of GeoTIFF paths
        self.dates: list = []          # extracted date strings, one per raster
        # computed
        self.ndvi_cols: list = []      # column names of NDVI time-series in df
        # auto-flag thresholds
        self.max_ndvi_thresh  = 0.3
        self.var_ndvi_thresh  = 0.005
        # clusters: {crop_name: {"subclusters": {label: [idx,...]}, "deleted": [], "n_done": 0, "all_idx": []}}
        self.clusters: dict = {}
        # custom labels: {original_label: custom_label}
        self.custom_labels: dict = {}
        # Add this inside GTModel.__init__(), near the other self. declarations
        self.cluster_visibility: dict = {}   # {subcluster_label: bool}
        

    # ── File loading ──────────────────────────────────────────────────────────
    def load_file(self, path: str) -> list:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            self.df = pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            self.df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type. Use CSV or XLSX.")
        self.filepath = path
        self.clusters = {}
        self.ndvi_cols = []
        self._ensure_meta_cols()
        return list(self.df.columns)

    def _ensure_meta_cols(self):
        for col in ("ndvi_flag", "ndvi_flag_reason", "Crop_Cluster"):
            if col not in self.df.columns:
                self.df[col] = ""

    def set_columns(self, uid, lat, lon, wkt, crop):
        self.uid_col  = uid
        self.lat_col  = lat
        self.lon_col  = lon
        self.wkt_col  = wkt
        self.crop_col = crop

    # ── Raster discovery ──────────────────────────────────────────────────────
    def scan_raster_dir(self, path: str) -> list:
        self.raster_dir = path
        tifs = sorted(glob.glob(os.path.join(path, "*.tif")) +
                      glob.glob(os.path.join(path, "*.tiff")))
        if not tifs:
            raise FileNotFoundError(f"No .tif files found in:\n{path}")
        self.raster_files = tifs
        self.dates = [self._extract_date(os.path.basename(p)) for p in tifs]
        return tifs

    @staticmethod
    def _extract_date(filename: str) -> str:
        """Try to find a YYYY-MM-DD or YYYYMMDD pattern in the filename."""
        m = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2})", filename)
        if m:
            raw = m.group(1).replace("_", "-")
            return raw[:4] + "-" + raw[4:6] + "-" + raw[6:] if len(raw) == 8 else raw
        return os.path.splitext(filename)[0]

    # ── NDVI extraction ───────────────────────────────────────────────────────
    def extract_ndvi_series(self, progress_cb=None) -> pd.DataFrame:
        """
        For each GT row × each raster file → compute NDVI.
        Priority: WKT polygon (median of pixels inside) → Lat/Lon centroid.
        Returns the updated DataFrame.
        """
        if not RASTERIO_OK:
            raise ImportError("rasterio is required.  pip install rasterio")
        if not PYPROJ_OK:
            raise ImportError("pyproj is required.  pip install pyproj")

        n_dates = len(self.raster_files)
        ndvi_matrix = np.full((len(self.df), n_dates), np.nan)

        for t_idx, raster_path in enumerate(self.raster_files):
            if progress_cb:
                progress_cb(t_idx, n_dates, os.path.basename(raster_path))

            with rasterio.open(raster_path) as src:
                crs_str = src.crs.to_string() if src.crs else None
                if crs_str is None:
                    continue

                for row_pos, (df_idx, row) in enumerate(self.df.iterrows()):
                    ndvi_val = np.nan

                    # ── Try WKT polygon first ────────────────────────────────
                    if (SHAPELY_OK and self.wkt_col and
                            self.wkt_col in self.df.columns and
                            pd.notna(row.get(self.wkt_col, None)) and
                            str(row[self.wkt_col]).strip()):
                        try:
                            geom = shapely_wkt.loads(str(row[self.wkt_col]))
                            # reproject geometry to raster CRS if needed
                            transformer = Transformer.from_crs(
                                "EPSG:4326", crs_str, always_xy=True)
                            from shapely.ops import transform as shp_transform
                            geom_proj = shp_transform(
                                lambda x, y, z=None: transformer.transform(x, y),
                                geom)
                            geom_json = [mapping(geom_proj)]
                            try:
                                out_image, _ = rasterio_mask(
                                    src, geom_json, crop=True, nodata=np.nan)
                                red_px = out_image[self.red_band - 1].flatten().astype(float)
                                nir_px = out_image[self.nir_band - 1].flatten().astype(float)
                                red_px[red_px <= 0] = np.nan
                                nir_px[nir_px <= 0] = np.nan
                                denom = nir_px + red_px
                                with np.errstate(invalid="ignore", divide="ignore"):
                                    px_ndvi = np.where(denom > 0,
                                                       (nir_px - red_px) / denom,
                                                       np.nan)
                                valid = px_ndvi[~np.isnan(px_ndvi)]
                                if len(valid) > 0:
                                    ndvi_val = float(np.median(valid))
                            except Exception:
                                pass  # fall through to centroid
                        except Exception:
                            pass

                    # ── Fallback: Lat/Lon centroid ───────────────────────────
                    if np.isnan(ndvi_val) and self.lat_col and self.lon_col:
                        try:
                            lat = float(row[self.lat_col])
                            lon = float(row[self.lon_col])
                            transformer = Transformer.from_crs(
                                "EPSG:4326", crs_str, always_xy=True)
                            x, y = transformer.transform(lon, lat)
                            samples = list(src.sample([(x, y)],
                                                       indexes=[self.red_band, self.nir_band]))
                            if samples:
                                red_v = float(samples[0][0])
                                nir_v = float(samples[0][1])
                                if red_v > 0 and nir_v > 0:
                                    denom = nir_v + red_v
                                    ndvi_val = (nir_v - red_v) / denom if denom > 0 else np.nan
                        except Exception:
                            pass

                    ndvi_matrix[row_pos, t_idx] = ndvi_val

        # ── Attach NDVI columns to df ────────────────────────────────────────
        self.ndvi_cols = [f"NDVI_{d}" for d in self.dates]
        # drop old if re-computing
        self.df = self.df.drop(columns=[c for c in self.ndvi_cols if c in self.df.columns],
                               errors="ignore")
        ndvi_df = pd.DataFrame(ndvi_matrix, index=self.df.index, columns=self.ndvi_cols)
        self.df = pd.concat([self.df, ndvi_df], axis=1)
        return self.df

    # ── Auto-flagging ─────────────────────────────────────────────────────────
    def auto_flag(self, max_thresh: float = None, var_thresh: float = None):
        """
        Flag rows as 'auto_rejected' if:
          - max NDVI across dates < max_thresh  (no crop peak)
          - variance of NDVI series < var_thresh (flat / no temporal variation)
        Sets df['ndvi_flag'] and df['ndvi_flag_reason'].
        """
        if not self.ndvi_cols:
            raise ValueError("Extract NDVI first.")
        mt = max_thresh if max_thresh is not None else self.max_ndvi_thresh
        vt = var_thresh if var_thresh is not None else self.var_ndvi_thresh
        self.max_ndvi_thresh = mt
        self.var_ndvi_thresh = vt

        flags, reasons = [], []
        for idx, row in self.df.iterrows():
            series = pd.to_numeric(row[self.ndvi_cols], errors="coerce").values
            valid  = series[~np.isnan(series)]
            if len(valid) == 0:
                flags.append("auto_rejected"); reasons.append("all_nan"); continue
            max_val  = float(np.nanmax(series))
            var_val  = float(np.nanvar(series))
            problems = []
            if max_val < mt:
                problems.append(f"max_ndvi={max_val:.3f}<{mt}")
            if var_val < vt:
                problems.append(f"variance={var_val:.4f}<{vt}")
            if problems:
                flags.append("auto_rejected"); reasons.append("|".join(problems))
            else:
                flags.append("ok"); reasons.append("")

        self.df["ndvi_flag"]        = flags
        self.df["ndvi_flag_reason"] = reasons
        return self.df

    # ── Cluster initialisation ────────────────────────────────────────────────
    def init_clusters(self):
        """Build cluster dict from non-flagged rows, grouped by crop."""
        if not self.crop_col:
            raise ValueError("Set crop column first.")
        good_df = self.df[self.df["ndvi_flag"] == "ok"]
        self.clusters = {}
        for crop in sorted(good_df[self.crop_col].dropna().unique()):
            name = str(crop)
            idx  = list(good_df[good_df[self.crop_col] == crop].index)
            self.clusters[name] = {
                "subclusters": {},
                "deleted":     [],
                "n_done":      0,
                "all_idx":     idx,
            }

    def split_crop(self, crop: str, n: int):
        """K-means on NDVI time-series vectors for one crop."""
        if not SKLEARN_OK:
            raise ImportError("scikit-learn required.  pip install scikit-learn")
        if not self.ndvi_cols:
            raise ValueError("Extract NDVI first.")
        all_idx = self.clusters[crop]["all_idx"]
        if len(all_idx) < n:
            raise ValueError(f"'{crop}' has {len(all_idx)} rows — cannot make {n} clusters.")

        X = self.df.loc[all_idx, self.ndvi_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        # fill NaN with column median (robust for missing dates)
        X = X.fillna(X.median())
        X = X.fillna(0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)
        labels = KMeans(n_clusters=n, random_state=42, n_init="auto").fit_predict(Xs)

        # ── Auto-label by peak NDVI date ─────────────────────────────────────
        # Compute mean peak date index per cluster → Early < Normal < Late
        cluster_peak = {}
        new_subs_raw = {}
        for i in range(n):
            key = f"{crop}_{i+1}"
            idxs = [all_idx[j] for j, lbl in enumerate(labels) if lbl == i]
            new_subs_raw[key] = idxs
            if idxs:
                sub_ndvi = self.df.loc[idxs, self.ndvi_cols].apply(
                    pd.to_numeric, errors="coerce")
                mean_curve = sub_ndvi.mean(axis=0).values
                peak_idx   = int(np.nanargmax(mean_curve))
                cluster_peak[key] = peak_idx
            else:
                cluster_peak[key] = 999

        # Sort clusters by peak date → assign Early/Normal/Late
        sorted_keys = sorted(cluster_peak, key=lambda k: cluster_peak[k])
        season_labels = ["Early", "Normal", "Late"] + [f"Group{i}" for i in range(4, 20)]
        new_subs = {}
        for rank, old_key in enumerate(sorted_keys):
            season = season_labels[rank] if rank < len(season_labels) else f"G{rank+1}"
            new_label = f"{crop}_{season}"
            new_subs[new_label] = new_subs_raw[old_key]
            # Store default custom label (user can rename)
            self.custom_labels[new_label] = new_label

        self.clusters[crop]["subclusters"] = new_subs
        self.clusters[crop]["deleted"]     = []
        self.clusters[crop]["n_done"]      = n
        
        # Initialize visibility for new clusters
        for new_label in new_subs.keys():
             self.cluster_visibility[new_label] = True

    def rename_cluster(self, old_label: str, new_label: str):
        """Rename a subcluster label and update visibility tracking."""
        for crop, cv in self.clusters.items():
            if old_label in cv["subclusters"]:
                cv["subclusters"][new_label] = cv["subclusters"].pop(old_label)
                # Update visibility
                if old_label in self.cluster_visibility:
                    self.cluster_visibility[new_label] = self.cluster_visibility.pop(old_label)
                else:
                    self.cluster_visibility[new_label] = True
                if old_label in self.custom_labels:
                    self.custom_labels[new_label] = self.custom_labels.pop(old_label)
                else:
                    self.custom_labels[new_label] = new_label
                return

    def delete_subcluster(self, crop: str, sub: str):
        rows = self.clusters[crop]["subclusters"].pop(sub, [])
        self.clusters[crop]["deleted"].extend(rows)

    def remove_gt_row(self, crop: str, sub: str, idx: int):
        lst = self.clusters[crop]["subclusters"].get(sub, [])
        if idx in lst:
            lst.remove(idx)
            self.clusters[crop]["deleted"].append(idx)

    # ── Mean NDVI curve per subcluster ────────────────────────────────────────
    def mean_curve(self, crop: str, sub: str) -> np.ndarray:
        idxs = self.clusters[crop]["subclusters"].get(sub, [])
        if not idxs or not self.ndvi_cols:
            return np.array([])
        vals = self.df.loc[idxs, self.ndvi_cols].apply(pd.to_numeric, errors="coerce")
        return vals.mean(axis=0).values

    def row_curve(self, row_idx: int) -> np.ndarray:
        if not self.ndvi_cols:
            return np.array([])
        return pd.to_numeric(
            self.df.loc[row_idx, self.ndvi_cols], errors="coerce").values

    # ── Build Crop_Cluster output column ─────────────────────────────────────
    def build_output_col(self) -> pd.Series:
        result = pd.Series("", index=self.df.index, dtype=str)
        # auto-rejected rows
        rej_mask = self.df["ndvi_flag"] == "auto_rejected"
        result[rej_mask] = "Auto_Rejected"
        # clustered rows
        for crop, cv in self.clusters.items():
            for sub, idxs in cv["subclusters"].items():
                for i in idxs:
                    result.at[i] = sub
            for i in cv.get("deleted", []):
                result.at[i] = f"{crop}_Deleted"
        return result

    # ── Export ────────────────────────────────────────────────────────────────
    def export(self, out_dir: str) -> list:
        os.makedirs(out_dir, exist_ok=True)
        df_out = self.df.copy()
        df_out["Crop_Cluster"] = self.build_output_col()

        # Split into cleaned (keep) and rejected
        bad_mask     = df_out["Crop_Cluster"].isin(["Auto_Rejected"]) | \
                       df_out["Crop_Cluster"].str.endswith("_Deleted", na=False)
        cleaned_df   = df_out[~bad_mask]
        rejected_df  = df_out[bad_mask]

        saved = []
        for name, frame in [("gt_full",     df_out),
                             ("gt_cleaned",  cleaned_df),
                             ("gt_rejected", rejected_df)]:
            csv_p  = os.path.join(out_dir, f"{name}.csv")
            xlsx_p = os.path.join(out_dir, f"{name}.xlsx")
            frame.to_csv(csv_p, index=False)
            try:
                frame.to_excel(xlsx_p, index=False)
                saved += [csv_p, xlsx_p]
            except Exception:
                saved.append(csv_p)

        # Cluster summary JSON
        def _safe(o): return o.item() if hasattr(o, "item") else str(o)
        jpath = os.path.join(out_dir, "cluster_summary.json")
        summary = {}
        for crop, cv in self.clusters.items():
            summary[crop] = {
                "subclusters": {s: len(idxs)
                                for s, idxs in cv["subclusters"].items()},
                "deleted": len(cv.get("deleted", [])),
            }
        with open(jpath, "w") as fh:
            json.dump(summary, fh, indent=2, default=_safe)
        saved.append(jpath)
        return saved


# ══════════════════════════════════════════════════════════════════════════════
#  WIDGET HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def card(parent, title="", padx=14, pady=10, fill="x", expand=False, pady_outer=(0, 6)):
    outer = tk.Frame(parent, bg=C["border"])
    outer.pack(fill=fill, expand=expand, pady=pady_outer)
    inner = tk.Frame(outer, bg=C["panel"], padx=padx, pady=pady)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    if title:
        tk.Label(inner, text=title, font=FONT_HEAD,
                 bg=C["panel"], fg=C["acc"]).pack(anchor="w", pady=(0, 6))
    return inner


def accent_btn(parent, text, cmd, bg=None, hover=None):
    bg    = bg or C["acc"]
    hover = hover or C["acc_hover"]
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg="white",
                  activebackground=hover, activeforeground="white",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  cursor="hand2", padx=12, pady=5, bd=0)
    b.bind("<Enter>", lambda _: b.config(bg=hover))
    b.bind("<Leave>", lambda _: b.config(bg=bg))
    return b


def ghost_btn(parent, text, cmd):
    return tk.Button(parent, text=text, command=cmd,
                     bg=C["panel"], fg=C["acc"],
                     activebackground=C["acc_light"], activeforeground=C["acc"],
                     relief="flat", font=FONT_SMALL, cursor="hand2",
                     padx=9, pady=4, bd=0,
                     highlightbackground=C["border"], highlightthickness=1)


def danger_btn(parent, text, cmd):
    return tk.Button(parent, text=text, command=cmd,
                     bg=C["danger_bg"], fg=C["danger"],
                     activebackground="#FADBD8", activeforeground=C["danger"],
                     relief="flat", font=FONT_SMALL, cursor="hand2",
                     padx=9, pady=4, bd=0,
                     highlightbackground="#F1948A", highlightthickness=1)


def combo(parent, variable, values=(), width=20, label="", label_width=0):
    f = tk.Frame(parent, bg=C["panel"])
    if label:
        lw = label_width or len(label) + 2
        tk.Label(f, text=label, bg=C["panel"], fg=C["text"],
                 font=FONT_BODY, width=lw, anchor="w").pack(side="left")
    cb = ttk.Combobox(f, textvariable=variable, values=list(values),
                      state="readonly", width=width)
    cb.pack(side="left")
    return f, cb


# ══════════════════════════════════════════════════════════════════════════════
#  STEP BAR
# ══════════════════════════════════════════════════════════════════════════════
class StepBar(tk.Canvas):
    R = 14
    def __init__(self, parent, **kw):
        super().__init__(parent, height=64, bg=C["bg"],
                         bd=0, highlightthickness=0, **kw)
        self.cur = 0
        self.bind("<Configure>", lambda _: self._draw())

    def set_step(self, idx):
        self.cur = idx
        self._draw()

    def _draw(self):
        self.delete("all")
        W = self.winfo_width()
        if W < 10:
            return
        n  = len(STEP_LABELS)
        sp = W / (n + 1)
        cx = [sp * (i + 1) for i in range(n)]
        cy = 26
        for i in range(n - 1):
            self.create_line(cx[i] + self.R + 2, cy, cx[i+1] - self.R - 2, cy,
                             fill=C["success"] if i < self.cur else C["border"], width=2)
        for i, (x, lbl) in enumerate(zip(cx, STEP_LABELS)):
            if   i < self.cur:  bg, fg, ring = C["success"], "white", C["success"]
            elif i == self.cur: bg, fg, ring = C["acc"],     "white", C["acc"]
            else:               bg, fg, ring = "white",      C["sub"], C["border"]
            self.create_oval(x-self.R, cy-self.R, x+self.R, cy+self.R,
                             fill=bg, outline=ring, width=2)
            self.create_text(x, cy, text=str(i+1), fill=fg,
                             font=("Segoe UI", 9, "bold"))
            self.create_text(x, cy+self.R+8, text=lbl,
                             fill=C["text"] if i == self.cur else C["sub"],
                             font=("Segoe UI", 8,
                                   "bold" if i == self.cur else "normal"),
                             anchor="n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NDVI GT Cluster Cleaner  v1.0")
        self.geometry("1440x900")
        self.minsize(1100, 720)
        self.configure(bg=C["bg"])
        self._apply_ttk()

        self.model = GTModel()
        self._cur  = 0

        # ── pick-event state ─────────────────────────────────────────────────
        self._pick_map     = {}   # {Line2D: (crop, sub, row_idx)}
        self._sub_tv_map   = {}   # {(crop, sub): Treeview}
        self._sub_show_fns = {}   # {(crop, sub): fn that expands rows}

        self._build_ui()
        self._goto(0)
     
        
     
        
    def _toggle_cluster_visibility(self, sub_label: str, var: tk.BooleanVar):
        """Called when user checks/unchecks a cluster"""
        self.model.cluster_visibility[sub_label] = var.get()
        
        crop = self._current_crop()
        if crop:
            self._cl4_draw_plot(crop)           # Refresh Step 4 plot
            if self._cur == 4:                  # If in Step 5
                self._draw_overview_plot()      # Refresh overview plot
                
     
    def _set_all_visibility(self, state: bool):
        """Show or hide all clusters for the current crop"""
        crop = self._current_crop()
        if not crop or crop not in self.model.clusters:
            return
        for sub in list(self.model.clusters[crop]["subclusters"].keys()):
            self.model.cluster_visibility[sub] = state
            if (crop, sub) in getattr(self, '_cluster_vis_vars', {}):
                self._cluster_vis_vars[(crop, sub)].set(state)
        
        self._cl4_draw_plot(crop)
        if self._cur == 4:   # Step 5
            self._draw_overview_plot()

            
     
    # ── ttk style ─────────────────────────────────────────────────────────────
    def _apply_ttk(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        for w in ("TFrame", "TLabel"):
            s.configure(w, background=C["bg"])
        s.configure("TCombobox", fieldbackground="white", background="white")
        s.configure("Treeview", rowheight=24, font=FONT_SMALL,
                    background="white", fieldbackground="white", foreground=C["text"])
        s.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"),
                    background=C["acc_light"], foreground=C["acc"])
        s.map("Treeview",
              background=[("selected", C["acc_light"])],
              foreground=[("selected", C["acc"])])

    # ── Shell ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.step_bar = StepBar(self)
        self.step_bar.pack(fill="x", padx=24, pady=(14, 6))
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=16)

        self.content = tk.Frame(self, bg=C["bg"])
        self.content.pack(fill="both", expand=True, padx=16, pady=(10, 0))

        self._frames = [
            self._build_step1(self.content),   # Load GT
            self._build_step2(self.content),   # Raster & NDVI
            self._build_step3(self.content),   # Auto-Flag
            self._build_step4(self.content),   # Cluster
            self._build_step5(self.content),   # Review & Export
        ]

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=16)
        nb = tk.Frame(self, bg=C["bg"], pady=10)
        nb.pack(fill="x", padx=24)
        self.btn_back = ghost_btn(nb, "◀  Back", self._back)
        self.btn_back.pack(side="left")
        self.lbl_status = tk.Label(nb, text="", bg=C["bg"], fg=C["sub"], font=FONT_SMALL)
        self.lbl_status.pack(side="left", padx=16)
        self.btn_next = accent_btn(nb, "Next  ▶", self._next)
        self.btn_next.pack(side="right")

    def _goto(self, idx):
        for i, f in enumerate(self._frames):
            if i == idx:
                f.pack(fill="both", expand=True)
            else:
                f.pack_forget()
        self._cur = idx
        self.step_bar.set_step(idx)
        self.btn_back.config(state="normal" if idx > 0 else "disabled")
        if idx == len(self._frames) - 1:
            self.btn_next.config(text="💾  Export", command=self._do_export)
        else:
            self.btn_next.config(text="Next  ▶", command=self._next)
        self._set_status("")

    def _next(self):
        if not self._validate(self._cur):
            return
        # side effects on advance
        if self._cur == 0:
            self._refresh_raster_step()
        if self._cur == 2:
            self._do_auto_flag()
            self._populate_cluster_list()
        if self._cur == 3:
            self._refresh_review()
        if self._cur < len(self._frames) - 1:
            self._goto(self._cur + 1)

    def _back(self):
        if self._cur > 0:
            self._goto(self._cur - 1)

    def _validate(self, step):
        m = self.model
        if step == 0:
            if m.df is None:
                self._set_status("⚠  Load a file first.")
                return False
            if not self.var_lat.get() or not self.var_lon.get() or not self.var_crop.get():
                self._set_status("⚠  Select Lat, Lon, and Crop columns.")
                return False
            m.set_columns(self.var_uid.get(), self.var_lat.get(),
                          self.var_lon.get(), self.var_wkt.get(), self.var_crop.get())
        elif step == 1:
            if not m.raster_files:
                self._set_status("⚠  Scan a raster folder first.")
                return False
            if not m.ndvi_cols:
                self._set_status("⚠  Extract NDVI first.")
                return False
        elif step == 3:
            pending = [c for c, v in m.clusters.items() if v["n_done"] == 0]
            if pending:
                ok = messagebox.askyesno(
                    "Unclustered Crops",
                    f"{len(pending)} crop(s) not clustered:\n  "
                    + ", ".join(pending[:8]) + "\n\nContinue to review anyway?")
                if not ok:
                    return False
        return True

    def _set_status(self, msg):
        self.lbl_status.config(text=msg)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 — Load GT File
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step1(self, parent):
        f = tk.Frame(parent, bg=C["bg"])

        top = card(f, "Step 1 — Load Ground-Truth File  (CSV or XLSX)", fill="x")

        # file picker row
        row0 = tk.Frame(top, bg=C["panel"]); row0.pack(fill="x", pady=(0, 10))
        self.var_file = tk.StringVar(value="No file selected")
        tk.Label(row0, textvariable=self.var_file, bg=C["panel"], fg=C["sub"],
                 font=FONT_SMALL, anchor="w", width=70).pack(side="left")
        accent_btn(row0, "📂  Browse", self._load_gt_file).pack(side="left", padx=(10, 0))

        # column pickers
        col_frame = tk.Frame(top, bg=C["panel"]); col_frame.pack(fill="x")

        self.var_uid  = tk.StringVar()
        self.var_lat  = tk.StringVar()
        self.var_lon  = tk.StringVar()
        self.var_wkt  = tk.StringVar()
        self.var_crop = tk.StringVar()

        pickers = [("UID / Field ID (optional)", self.var_uid),
                   ("Latitude column",           self.var_lat),
                   ("Longitude column",          self.var_lon),
                   ("WKT polygon column (opt.)", self.var_wkt),
                   ("Crop / Class column",       self.var_crop)]
        self._col_combos = {}
        for lbl_text, var in pickers:
            r = tk.Frame(col_frame, bg=C["panel"]); r.pack(fill="x", pady=2)
            tk.Label(r, text=lbl_text, bg=C["panel"], fg=C["text"],
                     font=FONT_BODY, width=28, anchor="w").pack(side="left")
            cb = ttk.Combobox(r, textvariable=var, state="readonly", width=30)
            cb.pack(side="left")
            self._col_combos[lbl_text] = cb

        cb_crop = self._col_combos["Crop / Class column"]
        cb_crop.bind("<<ComboboxSelected>>", self._preview_crops)

        # crop chips
        self.frm_crop_chips = tk.Frame(top, bg=C["panel"])
        self.frm_crop_chips.pack(fill="x", pady=(8, 0))

        # preview
        prev = card(f, "File Preview (first 20 rows)",
                    fill="both", expand=True, pady_outer=(6, 0))
        pf = tk.Frame(prev, bg=C["panel"]); pf.pack(fill="both", expand=True)
        self.prev_tree = ttk.Treeview(pf, show="headings", height=12)
        vsb = ttk.Scrollbar(pf, orient="vertical",   command=self.prev_tree.yview)
        hsb = ttk.Scrollbar(pf, orient="horizontal", command=self.prev_tree.xview)
        self.prev_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.prev_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        pf.rowconfigure(0, weight=1); pf.columnconfigure(0, weight=1)
        return f

    def _load_gt_file(self):
        path = filedialog.askopenfilename(
            title="Select GT file",
            filetypes=[("Table files", "*.csv *.xlsx *.xls"), ("All", "*.*")])
        if not path:
            return
        try:
            cols = self.model.load_file(path)
        except Exception as e:
            messagebox.showerror("Load Error", str(e)); return

        self.var_file.set(os.path.basename(path))
        for cb in self._col_combos.values():
            cb["values"] = cols

        # smart guess
        low = {c.lower(): c for c in cols}
        guesses = {"uid": self.var_uid, "field_id": self.var_uid, "id": self.var_uid,
                   "lat": self.var_lat, "latitude": self.var_lat,
                   "lon": self.var_lon, "longitude": self.var_lon,
                   "wkt": self.var_wkt, "geometry": self.var_wkt, "geom": self.var_wkt,
                   "crop": self.var_crop, "crop_name": self.var_crop, "class": self.var_crop}
        for key, var in guesses.items():
            if key in low and not var.get():
                var.set(low[key])

        # preview
        tbl = self.prev_tree
        tbl.delete(*tbl.get_children())
        tbl["columns"] = cols
        for c in cols:
            tbl.heading(c, text=c)
            tbl.column(c, width=max(70, min(140, len(c)*9)), anchor="center")
        for _, row in self.model.df.head(20).iterrows():
            tbl.insert("", "end", values=list(row))

        n = len(self.model.df)
        self._set_status(f"✓  Loaded {n} rows — configure columns, then Next ▶")

    def _preview_crops(self, _=None):
        col = self.var_crop.get()
        if not col or self.model.df is None:
            return
        for w in self.frm_crop_chips.winfo_children():
            w.destroy()
        tk.Label(self.frm_crop_chips, text="Crops found:",
                 bg=C["panel"], fg=C["sub"], font=FONT_SMALL).pack(side="left", padx=(0, 8))
        vals = sorted(self.model.df[col].dropna().unique())
        for v in vals[:15]:
            n = (self.model.df[col] == v).sum()
            tk.Label(self.frm_crop_chips, text=f"{v} ({n})",
                     bg=C["acc_light"], fg=C["acc"],
                     font=("Segoe UI", 8, "bold"),
                     padx=8, pady=3).pack(side="left", padx=2)
        if len(vals) > 15:
            tk.Label(self.frm_crop_chips, text=f"…+{len(vals)-15} more",
                     bg=C["panel"], fg=C["sub"], font=FONT_SMALL).pack(side="left")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 — Raster Folder & NDVI Extraction
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step2(self, parent):
        f = tk.Frame(parent, bg=C["bg"])

        top = card(f, "Step 2 — Raster Folder & Band Configuration", fill="x")

        # folder row
        r0 = tk.Frame(top, bg=C["panel"]); r0.pack(fill="x", pady=(0, 10))
        self.var_rdir = tk.StringVar(value="No folder selected")
        tk.Label(r0, textvariable=self.var_rdir, bg=C["panel"], fg=C["sub"],
                 font=FONT_SMALL, anchor="w", width=60).pack(side="left")
        accent_btn(r0, "📂  Browse Raster Folder", self._browse_raster_dir).pack(side="left", padx=(10, 0))

        # band indices
        band_row = tk.Frame(top, bg=C["panel"]); band_row.pack(fill="x", pady=(0, 8))
        tk.Label(band_row, text="Red band index:", bg=C["panel"], font=FONT_BODY).pack(side="left")
        self.var_red = tk.IntVar(value=3)
        ttk.Spinbox(band_row, from_=1, to=30, textvariable=self.var_red, width=5).pack(side="left", padx=(4, 20))
        tk.Label(band_row, text="NIR band index:", bg=C["panel"], font=FONT_BODY).pack(side="left")
        self.var_nir = tk.IntVar(value=4)
        ttk.Spinbox(band_row, from_=1, to=30, textvariable=self.var_nir, width=5).pack(side="left", padx=(4, 20))
        tk.Label(band_row,
                 text="(1-based.  Sentinel-2 10m: Red=1, NIR=2  |  20m SR: Red=3, NIR=7  — check your stack)",
                 bg=C["panel"], fg=C["sub"], font=FONT_SMALL).pack(side="left")

        # extraction mode
        mode_row = tk.Frame(top, bg=C["panel"]); mode_row.pack(fill="x", pady=(0, 8))
        tk.Label(mode_row, text="Extraction mode:", bg=C["panel"], font=FONT_BODY).pack(side="left")
        self.var_extr_mode = tk.StringVar(value="auto")
        for val, lbl in [("auto", "Auto (WKT polygon → centroid fallback)"),
                         ("polygon", "WKT polygon only"),
                         ("centroid", "Lat/Lon centroid only")]:
            ttk.Radiobutton(mode_row, text=lbl, variable=self.var_extr_mode,
                            value=val).pack(side="left", padx=(10, 0))

        # action buttons
        act_row = tk.Frame(top, bg=C["panel"]); act_row.pack(fill="x", pady=(4, 0))
        accent_btn(act_row, "🔍  Scan Folder", self._scan_raster_folder).pack(side="left", padx=(0, 10))
        self.btn_extract = accent_btn(act_row, "⚙  Extract NDVI Time-Series", self._extract_ndvi)
        self.btn_extract.pack(side="left")

        # raster list
        rlist = card(f, "Detected Raster Files", fill="x", pady_outer=(6, 0))
        rl = tk.Frame(rlist, bg=C["panel"]); rl.pack(fill="x")
        self.raster_listbox = tk.Listbox(rl, height=6, font=FONT_MONO,
                                          bg="white", selectbackground=C["acc_light"])
        rlsb = ttk.Scrollbar(rl, command=self.raster_listbox.yview)
        self.raster_listbox.config(yscrollcommand=rlsb.set)
        self.raster_listbox.pack(side="left", fill="x", expand=True)
        rlsb.pack(side="right", fill="y")

        # progress / log
        log_card = card(f, "Extraction Log", fill="both", expand=True, pady_outer=(6, 0))
        self.log_text = tk.Text(log_card, height=8, font=FONT_MONO,
                                bg="#1C2B3A", fg="#B0E0FF", wrap="word")
        lsb = ttk.Scrollbar(log_card, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=lsb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        lsb.pack(side="right", fill="y")
        self._log("Ready. Scan a raster folder to begin.")
        return f

    def _refresh_raster_step(self):
        """Called when moving from step 1 → step 2; no mandatory action."""
        pass

    def _browse_raster_dir(self):
        d = filedialog.askdirectory(title="Select folder containing GeoTIFF files")
        if d:
            self.var_rdir.set(d)
            self._scan_raster_folder()

    def _scan_raster_folder(self):
        path = self.var_rdir.get()
        if not path or path == "No folder selected":
            messagebox.showwarning("No folder", "Select a raster folder first.")
            return
        try:
            files = self.model.scan_raster_dir(path)
        except Exception as e:
            messagebox.showerror("Scan Error", str(e)); return
        self.raster_listbox.delete(0, "end")
        for p, d in zip(self.model.raster_files, self.model.dates):
            self.raster_listbox.insert("end", f"  {d}   {os.path.basename(p)}")
        self._log(f"Found {len(files)} raster file(s) in:\n  {path}")

    def _extract_ndvi(self):
        if not RASTERIO_OK:
            messagebox.showerror("Missing library",
                "rasterio is required.\n\npip install rasterio"); return
        if not self.model.raster_files:
            messagebox.showwarning("No rasters", "Scan a folder first."); return
        if self.model.df is None:
            messagebox.showwarning("No data", "Load GT file first."); return

        self.model.red_band = self.var_red.get()
        self.model.nir_band = self.var_nir.get()

        self._log(f"\nStarting NDVI extraction — {len(self.model.raster_files)} dates × {len(self.model.df)} rows")
        self.btn_extract.config(text="⏳  Extracting…", state="disabled")
        self.update_idletasks()

        def progress(t_idx, n_dates, fname):
            self._log(f"  [{t_idx+1}/{n_dates}]  {fname}")
            self.update_idletasks()

        try:
            self.model.extract_ndvi_series(progress_cb=progress)
            self._log(f"\n✓  Done.  Added {len(self.model.ndvi_cols)} NDVI columns to DataFrame.")
            self._set_status("✓  NDVI extracted — click Next ▶ to review auto-flagging")
        except Exception as e:
            self._log(f"\n✗  ERROR: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Extraction Error", str(e))
        finally:
            self.btn_extract.config(text="⚙  Extract NDVI Time-Series", state="normal")

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3 — Auto-Flag Bad Rows
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step3(self, parent):
        f = tk.Frame(parent, bg=C["bg"])

        top = card(f, "Step 3 — Auto-Flag Non-Crop / Mislabeled Rows", fill="x")

        tk.Label(top,
                 text="Rows are flagged 'auto_rejected' if their NDVI time-series fails either threshold below.\n"
                      "Flagged rows are excluded from clustering and exported separately as 'gt_rejected'.",
                 bg=C["panel"], fg=C["sub"], font=FONT_SMALL, justify="left").pack(anchor="w", pady=(0, 8))

        thr_frame = tk.Frame(top, bg=C["panel"]); thr_frame.pack(fill="x", pady=(0, 8))

        tk.Label(thr_frame, text="Max NDVI threshold:", bg=C["panel"], font=FONT_BODY).grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        self.var_max_thr = tk.DoubleVar(value=0.3)
        ttk.Spinbox(thr_frame, from_=0.0, to=1.0, increment=0.05, format="%.2f",
                    textvariable=self.var_max_thr, width=7).grid(row=0, column=1, sticky="w")
        tk.Label(thr_frame,
                 text="  ← rows with peak NDVI below this → rejected  (no crop green-up)",
                 bg=C["panel"], fg=C["sub"], font=FONT_SMALL).grid(row=0, column=2, sticky="w", padx=(8, 0))

        tk.Label(thr_frame, text="NDVI variance threshold:", bg=C["panel"], font=FONT_BODY).grid(
            row=1, column=0, sticky="w", pady=(6, 0), padx=(0, 8))
        self.var_var_thr = tk.DoubleVar(value=0.005)
        ttk.Spinbox(thr_frame, from_=0.0, to=0.5, increment=0.001, format="%.4f",
                    textvariable=self.var_var_thr, width=9).grid(row=1, column=1, sticky="w", pady=(6, 0))
        tk.Label(thr_frame,
                 text="  ← rows with flat NDVI curve (variance below this) → rejected",
                 bg=C["panel"], fg=C["sub"], font=FONT_SMALL).grid(row=1, column=2, sticky="w",
                                                                    padx=(8, 0), pady=(6, 0))

        btn_row = tk.Frame(top, bg=C["panel"]); btn_row.pack(fill="x", pady=(8, 0))
        accent_btn(btn_row, "▶  Run Auto-Flag", self._do_auto_flag).pack(side="left")
        self.lbl_flag_result = tk.Label(btn_row, text="", bg=C["panel"],
                                        fg=C["warn"], font=FONT_SMALL)
        self.lbl_flag_result.pack(side="left", padx=16)

        # NDVI plot for preview
        pw = tk.PanedWindow(f, orient="horizontal", sashwidth=5, bg=C["border"])
        pw.pack(fill="both", expand=True, pady=(10, 0))

        # rejected list
        p1 = tk.Frame(pw, bg=C["bg"]); pw.add(p1, minsize=320, width=360)
        rc = card(p1, "Auto-Rejected Rows (sample)", fill="both", expand=True, pady_outer=(0, 0))
        rf = tk.Frame(rc, bg=C["panel"]); rf.pack(fill="both", expand=True)
        rej_cols = ("idx", "crop", "max_ndvi", "variance", "reason")
        self.rej_tree = ttk.Treeview(rf, columns=rej_cols, show="headings", height=20)
        col_w = {"idx": 50, "crop": 90, "max_ndvi": 80, "variance": 80, "reason": 160}
        for c in rej_cols:
            self.rej_tree.heading(c, text=c)
            self.rej_tree.column(c, width=col_w.get(c, 80), anchor="center")
        rej_vsb = ttk.Scrollbar(rf, orient="vertical", command=self.rej_tree.yview)
        self.rej_tree.configure(yscrollcommand=rej_vsb.set)
        self.rej_tree.grid(row=0, column=0, sticky="nsew")
        rej_vsb.grid(row=0, column=1, sticky="ns")
        rf.rowconfigure(0, weight=1); rf.columnconfigure(0, weight=1)
        self.rej_tree.bind("<<TreeviewSelect>>", self._on_rej_select)

        # plot pane
        p2 = tk.Frame(pw, bg=C["bg"]); pw.add(p2, minsize=400)
        pc = card(p2, "NDVI Curve Preview", padx=6, pady=6,
                  fill="both", expand=True, pady_outer=(0, 0))
        self.fig3 = Figure(figsize=(5, 3.5), dpi=90, facecolor=C["panel"])
        self.ax3  = self.fig3.add_subplot(111)
        self._style_ax(self.ax3, self.fig3)
        self.cv3  = FigureCanvasTkAgg(self.fig3, master=pc)
        self.cv3.get_tk_widget().pack(fill="both", expand=True)
        return f

    def _do_auto_flag(self):
        if not self.model.ndvi_cols:
            self._set_status("⚠  Extract NDVI first (Step 2).")
            return
        try:
            self.model.max_ndvi_thresh = self.var_max_thr.get()
            self.model.var_ndvi_thresh = self.var_var_thr.get()
            self.model.auto_flag()
        except Exception as e:
            messagebox.showerror("Flag Error", str(e)); return

        df = self.model.df
        n_bad  = (df["ndvi_flag"] == "auto_rejected").sum()
        n_good = (df["ndvi_flag"] == "ok").sum()
        self.lbl_flag_result.config(
            text=f"✓  {n_good} OK  |  {n_bad} auto-rejected",
            fg=C["success"] if n_bad == 0 else C["warn"])
        self._fill_rejected_table()
        self.model.init_clusters()
        self._set_status(f"✓  Flagged {n_bad} rows — check the list, then Next ▶ to cluster")

    def _fill_rejected_table(self):
        self.rej_tree.delete(*self.rej_tree.get_children())
        df = self.model.df
        rej = df[df["ndvi_flag"] == "auto_rejected"]
        for idx, row in rej.iterrows():
            series  = pd.to_numeric(row[self.model.ndvi_cols], errors="coerce").values
            max_ndvi = f"{np.nanmax(series):.3f}" if not np.all(np.isnan(series)) else "nan"
            var_ndvi = f"{np.nanvar(series):.4f}" if not np.all(np.isnan(series)) else "nan"
            crop_val = row.get(self.model.crop_col, "")
            self.rej_tree.insert("", "end", iid=str(idx),
                                 values=(idx, crop_val, max_ndvi, var_ndvi,
                                         row.get("ndvi_flag_reason", "")))

    def _on_rej_select(self, _=None):
        sel = self.rej_tree.selection()
        if not sel:
            return
        row_idx = int(sel[0])
        self._plot_single_curve(row_idx, self.ax3, self.cv3, self.fig3,
                                title=f"Row {row_idx} — NDVI curve (rejected)")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4 — Cluster
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step4(self, parent):
        f = tk.Frame(parent, bg=C["bg"])

        pw = tk.PanedWindow(f, orient="horizontal", sashwidth=5, bg=C["border"])
        pw.pack(fill="both", expand=True)

        # ── Pane 1: crop list ─────────────────────────────────────────────────
        p1 = tk.Frame(pw, bg=C["bg"]); pw.add(p1, minsize=180, width=210)
        lc = card(p1, "Crops", fill="both", expand=True, pady_outer=(0, 0))
        lf = tk.Frame(lc, bg=C["panel"]); lf.pack(fill="both", expand=True)
        self.cl_lb = tk.Listbox(lf, font=("Segoe UI", 9), bg="white",
                                selectbackground=C["acc_light"],
                                selectforeground=C["acc"],
                                activestyle="none", height=30)
        cl_vsb = ttk.Scrollbar(lf, command=self.cl_lb.yview)
        self.cl_lb.config(yscrollcommand=cl_vsb.set)
        self.cl_lb.pack(side="left", fill="both", expand=True)
        cl_vsb.pack(side="right", fill="y")
        self.cl_lb.bind("<<ListboxSelect>>", self._on_cl4_crop_sel)

        self.lbl_cl4_summary = tk.Label(lc, text="", bg=C["panel"], fg=C["sub"],
                                        font=FONT_SMALL, wraplength=200, justify="left")
        self.lbl_cl4_summary.pack(anchor="w", pady=(4, 0))

        # ── Pane 2: cluster cards ─────────────────────────────────────────────
        p2 = tk.Frame(pw, bg=C["bg"]); pw.add(p2, minsize=380, width=460)
        rc = card(p2, "", padx=8, fill="both", expand=True, pady_outer=(0, 0))

        ctrl = tk.Frame(rc, bg=C["panel"]); ctrl.pack(fill="x", pady=(0, 6))
        self.lbl_cl4_crop = tk.Label(ctrl, text="← Select a crop",
                                     bg=C["panel"], fg=C["acc"], font=("Segoe UI", 10, "bold"))
        self.lbl_cl4_crop.pack(side="left", padx=(0, 10))
        tk.Label(ctrl, text="N clusters:", bg=C["panel"], font=FONT_BODY).pack(side="left")
        self.var_cl4_n = tk.IntVar(value=3)
        ttk.Spinbox(ctrl, from_=1, to=20, textvariable=self.var_cl4_n, width=4).pack(side="left", padx=(4, 8))
        accent_btn(ctrl, "▶  Run K-means", self._cl4_run).pack(side="left", padx=(0, 6))
        ghost_btn(ctrl, "↺  Reset", self._cl4_reset).pack(side="left", padx=(0, 8))
 
         # New: Show/Hide All buttons
        ghost_btn(ctrl, "Show All", lambda: self._set_all_visibility(True)).pack(side="left", padx=2)
        ghost_btn(ctrl, "Hide All", lambda: self._set_all_visibility(False)).pack(side="left", padx=2)
        self.lbl_cl4_status = tk.Label(ctrl, text="", bg=C["panel"],
                                       fg=C["success"], font=FONT_SMALL)
        self.lbl_cl4_status.pack(side="left", padx=(10, 0))

        tk.Label(rc,
                 text="Clusters auto-labelled by peak NDVI date: Early / Normal / Late.\n"
                      "▼ Show rows → click a row to highlight its NDVI curve on the plot.",
                 bg=C["panel"], fg=C["sub"], font=FONT_SMALL, justify="left").pack(anchor="w", pady=(0, 4))

        # scrollable card area
        co = tk.Frame(rc, bg=C["border"]); co.pack(fill="both", expand=True)
        self.cl4_canvas = tk.Canvas(co, bg=C["panel"], bd=0, highlightthickness=0)
        cl4_vsb = ttk.Scrollbar(co, orient="vertical", command=self.cl4_canvas.yview)
        self.cl4_canvas.configure(yscrollcommand=cl4_vsb.set)
        cl4_vsb.pack(side="right", fill="y")
        self.cl4_canvas.pack(side="left", fill="both", expand=True)
        self.cl4_inner = tk.Frame(self.cl4_canvas, bg=C["panel"])
        self.cl4_win   = self.cl4_canvas.create_window((0, 0), window=self.cl4_inner, anchor="nw")
        self.cl4_inner.bind("<Configure>",
            lambda _: self.cl4_canvas.configure(scrollregion=self.cl4_canvas.bbox("all")))
        self.cl4_canvas.bind("<Configure>",
            lambda e: self.cl4_canvas.itemconfig(self.cl4_win, width=e.width))

        # ── Pane 3: NDVI plot ─────────────────────────────────────────────────
        p3 = tk.Frame(pw, bg=C["bg"]); pw.add(p3, minsize=340)
        pc = card(p3, "NDVI Time-Series Plot", padx=6, pady=6,
                  fill="both", expand=True, pady_outer=(0, 0))

        # legend strip
        lg = tk.Frame(pc, bg=C["panel"]); lg.pack(fill="x", pady=(0, 4))
        for swatch, lbl in [("#AAAAAA", "All rows in cluster (grey)"),
                             (C["highlight"], "Selected row"),
                             (PALETTE[0], "Cluster mean curve")]:
            tk.Frame(lg, bg=swatch, width=14, height=14).pack(side="left", padx=(0, 3))
            tk.Label(lg, text=lbl, bg=C["panel"], fg=C["sub"],
                     font=("Segoe UI", 7)).pack(side="left", padx=(0, 10))

        self.fig4 = Figure(figsize=(4, 4), dpi=90, facecolor=C["panel"])
        self.ax4  = self.fig4.add_subplot(111)
        self._style_ax(self.ax4, self.fig4)
        self.cv4  = FigureCanvasTkAgg(self.fig4, master=pc)
        self.cv4.get_tk_widget().pack(fill="both", expand=True)
        self.fig4.canvas.mpl_connect("pick_event", self._on_pick4)

        self.lbl_row_hint = tk.Label(pc, text="", bg=C["panel"], fg=C["highlight"],
                                     font=("Segoe UI", 8, "italic"), wraplength=320)
        self.lbl_row_hint.pack(anchor="w", pady=(3, 0))
        return f

    # ── Step-4 helpers ────────────────────────────────────────────────────────
    def _populate_cluster_list(self):
        self.cl_lb.delete(0, "end")
        for crop, cv in self.model.clusters.items():
            n_rows = len(cv["all_idx"])
            n_cl   = cv["n_done"]
            lbl    = (f"✓ {crop}  ({n_rows}r, {n_cl}cl)"
                      if n_cl > 0 else f"{crop}  ({n_rows} rows)")
            self.cl_lb.insert("end", lbl)
        if self.model.clusters:
            self.cl_lb.selection_set(0)
            self._on_cl4_crop_sel()

    def _current_crop(self):
        sel = self.cl_lb.curselection()
        if not sel:
            return None
        raw = self.cl_lb.get(sel[0])
        return raw.lstrip("✓ ").split("  (")[0].strip()

    def _on_cl4_crop_sel(self, _=None):
        crop = self._current_crop()
        if crop is None:
            return
        self.lbl_cl4_crop.config(text=f"Crop:  {crop}")
        self._cl4_redraw(crop)
        self._cl4_draw_plot(crop)

    def _cl4_run(self):
        crop = self._current_crop()
        if crop is None:
            messagebox.showwarning("No crop", "Select a crop first."); return
        try:
            self.model.split_crop(crop, self.var_cl4_n.get())
        except Exception as e:
            messagebox.showerror("Clustering Error", str(e)); return
        self._cl4_redraw(crop)
        self._cl4_update_listbox(crop)
        self._cl4_draw_plot(crop)
        n = self.var_cl4_n.get()
        self.lbl_cl4_status.config(text=f"✓  {n} clusters (Early/Normal/Late order)")
        self._cl4_update_summary()

    def _cl4_reset(self):
        crop = self._current_crop()
        if crop is None:
            return
        self.model.clusters[crop].update({"subclusters": {}, "deleted": [], "n_done": 0})
        self._cl4_redraw(crop)
        self._cl4_update_listbox(crop)
        self._cl4_draw_plot(crop)
        self.lbl_cl4_status.config(text="↺  Reset", fg=C["warn"])
        self._cl4_update_summary()

    def _cl4_delete_sub(self, crop, sub):
        n = len(self.model.clusters[crop]["subclusters"].get(sub, []))
        if not messagebox.askyesno("Delete Cluster",
                f"Delete '{sub}'?\n{n} rows will be marked '{crop}_Deleted' on export."):
            return
        self.model.delete_subcluster(crop, sub)
        self._cl4_redraw(crop)
        self._cl4_update_listbox(crop)
        self._cl4_draw_plot(crop)
        self.lbl_cl4_status.config(text=f"✕  Deleted '{sub}'", fg=C["danger"])
        self._cl4_update_summary()

    def _cl4_rename_sub(self, crop, sub):
        new_name = simpledialog.askstring(
            "Rename Cluster",
            f"New name for '{sub}':\n(e.g. {crop}_EarlyRabi, {crop}_Late_Irrigated)",
            parent=self)
        if not new_name or new_name.strip() == sub:
            return
        new_name = new_name.strip()
        self.model.rename_cluster(sub, new_name)
        self._cl4_redraw(crop)
        self._cl4_update_listbox(crop)
        self._cl4_draw_plot(crop)
        self.lbl_cl4_status.config(text=f"✎  Renamed → '{new_name}'", fg=C["acc"])

    def _cl4_delete_gt_rows(self, crop, sub, tv):
        selected = tv.selection()
        if not selected:
            messagebox.showwarning("None selected", "Select row(s) first."); return
        row_indices = [int(tv.item(s, "values")[0]) for s in selected]
        for idx in row_indices:
            self.model.remove_gt_row(crop, sub, idx)
        for idx in row_indices:
            try:
                tv.delete(str(idx))
            except Exception:
                pass
        self._cl4_draw_plot(crop)
        self._cl4_update_listbox(crop)
        self._cl4_update_summary()
        self.lbl_cl4_status.config(
            text=f"✕  Removed {len(row_indices)} row(s)", fg=C["danger"])

    # ── Cluster card builder ──────────────────────────────────────────────────

    # ── Cluster card builder ──────────────────────────────────────────────────
    def _cl4_redraw(self, crop):
        """Redraw all cluster cards with visibility checkboxes"""
        # Clear previous mappings and old widgets
        self._pick_map = {}
        if not hasattr(self, '_cluster_vis_vars'):
            self._cluster_vis_vars = {}

        for k in list(self._sub_tv_map.keys()):
            if k[0] == crop:
                del self._sub_tv_map[k]
        for k in list(self._sub_show_fns.keys()):
            if k[0] == crop:
                del self._sub_show_fns[k]

        for w in self.cl4_inner.winfo_children():
            w.destroy()

        if crop not in self.model.clusters:
            return

        subs    = self.model.clusters[crop]["subclusters"]
        deleted = self.model.clusters[crop].get("deleted", [])
        n_done  = self.model.clusters[crop]["n_done"]

        if not subs and n_done == 0:
            tk.Label(self.cl4_inner,
                     text="No clustering yet.\nSet N and click  ▶ Run K-means",
                     bg=C["panel"], fg=C["sub"],
                     font=("Segoe UI", 10, "italic")).pack(pady=30)
            return

        df_cols = list(self.model.df.columns)
        preview_cols = [c for c in df_cols if c not in self.model.ndvi_cols][:6]

        for i, (sub, idxs) in enumerate(subs.items()):
            clr = PALETTE[i % len(PALETTE)]
            card_f = tk.Frame(self.cl4_inner, bg=C["panel"], relief="solid", bd=1)
            card_f.pack(fill="x", padx=4, pady=4, ipadx=6, ipady=4)

            # ==================== UPDATED HEADER WITH CHECKBOX ====================
            hrow = tk.Frame(card_f, bg=C["panel"])
            hrow.pack(fill="x", pady=(4, 6))

            # Visibility Checkbox
            vis_var = tk.BooleanVar(value=self.model.cluster_visibility.get(sub, True))
            self.model.cluster_visibility[sub] = vis_var.get()   # sync

            chk = ttk.Checkbutton(
                hrow, 
                variable=vis_var,
                text="", 
                command=lambda s=sub, v=vis_var: self._toggle_cluster_visibility(s, v)
            )
            chk.pack(side="left", padx=(6, 8))

            # Color indicator + Cluster name
            tk.Label(hrow, text="  ", bg=clr, width=2, relief="solid", bd=1).pack(side="left")
            tk.Label(hrow, text=f"  {sub}", bg=C["panel"], fg=clr,
                     font=("Segoe UI", 10, "bold")).pack(side="left")
            tk.Label(hrow, text=f"  ({len(idxs)} rows)",
                     bg=C["panel"], fg=C["sub"], font=FONT_SMALL).pack(side="left", padx=(6, 0))

            # Peak NDVI info
            if idxs and self.model.ndvi_cols:
                mc = self.model.mean_curve(crop, sub)
                if len(mc):
                    peak_val  = float(np.nanmax(mc))
                    peak_date = self.model.dates[int(np.nanargmax(mc))] if self.model.dates else ""
                    tk.Label(hrow,
                             text=f"   peak NDVI {peak_val:.2f} @ {peak_date}",
                             bg=C["panel"], fg=C["sub"], font=FONT_SMALL).pack(side="left", padx=(12, 0))

            # Action buttons on the right
            ghost_btn(hrow, "✎ Rename",
                      lambda c=crop, s=sub: self._cl4_rename_sub(c, s)).pack(side="right", padx=4)
            danger_btn(hrow, "✕ Delete",
                       lambda c=crop, s=sub: self._cl4_delete_sub(c, s)).pack(side="right", padx=4)

            # Store reference to checkbox variable
            self._cluster_vis_vars[(crop, sub)] = vis_var

            # ==================== GT Table (unchanged) ====================
            tbl_frame = tk.Frame(card_f, bg=C["panel"])
            tbl_frame.pack(fill="x", padx=4, pady=(3, 0))
            th = tk.Frame(tbl_frame, bg=C["panel"]); th.pack(fill="x")

            cols_show = ["_idx"] + preview_cols
            tv = ttk.Treeview(tbl_frame, columns=cols_show,
                              show="headings", height=5, selectmode="extended")
            tv.heading("_idx", text="Row #")
            tv.column("_idx", width=52, anchor="center", stretch=False)
            for c in preview_cols:
                tv.heading(c, text=c)
                tv.column(c, width=max(55, min(120, len(c)*8)), anchor="center")
            tv_vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=tv.yview)
            tv.configure(yscrollcommand=tv_vsb.set)

            for j, row_idx in enumerate(idxs):
                vals = [row_idx] + [str(self.model.df.loc[row_idx, c])
                                    for c in preview_cols]
                tv.insert("", "end", iid=str(row_idx), values=vals,
                          tags=("even" if j % 2 == 0 else "odd",))
            tv.tag_configure("even", background=C["row_even"])
            tv.tag_configure("odd",  background=C["row_odd"])

            self._sub_tv_map[(crop, sub)] = tv
            tv.bind("<<TreeviewSelect>>",
                    lambda e, c=crop, s=sub, t=tv: self._on_row_sel(c, s, t))

            # toggle show/hide rows
            toggle_var = tk.BooleanVar(value=False)
            brl = [None]

            def _toggle(tf=tbl_frame, tv_=tv, vsb=tv_vsb, bv=toggle_var, br=brl):
                bv.set(not bv.get())
                if bv.get():
                    tv_.pack(side="left", fill="x", expand=True, pady=(2, 0))
                    vsb.pack(side="right", fill="y", pady=(2, 0))
                    if br[0]: br[0].config(text="▲  Hide rows")
                else:
                    tv_.pack_forget(); vsb.pack_forget()
                    if br[0]: br[0].config(text="▼  Show rows")

            tbtn = ghost_btn(th, "▼  Show rows", _toggle)
            tbtn.pack(side="left", padx=(0, 6)); brl[0] = tbtn

            def _force_show(tf=tbl_frame, tv_=tv, vsb=tv_vsb, bv=toggle_var, br=brl):
                if not bv.get():
                    bv.set(True)
                    tv_.pack(side="left", fill="x", expand=True, pady=(2, 0))
                    vsb.pack(side="right", fill="y", pady=(2, 0))
                    if br[0]: br[0].config(text="▲  Hide rows")
            self._sub_show_fns[(crop, sub)] = _force_show

            danger_btn(th, "✕  Delete selected rows",
                       lambda c=crop, s=sub, tv_=tv:
                           self._cl4_delete_gt_rows(c, s, tv_)).pack(side="left", padx=4)

        # deleted pool
        if deleted:
            df_card = tk.Frame(self.cl4_inner, bg=C["del_bg"], relief="solid", bd=1)
            df_card.pack(fill="x", padx=4, pady=4, ipadx=6, ipady=4)
            tk.Label(df_card,
                     text=f"🗑  {crop}_Deleted — {len(deleted)} row(s)",
                     bg=C["del_bg"], fg=C["danger"],
                     font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8)

    # ── Row selection → plot overlay ──────────────────────────────────────────
    def _on_row_sel(self, crop, sub, tv):
        sel = tv.selection()
        if not sel:
            self._cl4_draw_plot(crop)
            self.lbl_row_hint.config(text="")
            return
        idxs = [int(tv.item(s, "values")[0]) for s in sel]
        self._cl4_draw_plot(crop, highlight_idxs=idxs, highlight_sub=sub)
        self.lbl_row_hint.config(
            text=f"Showing {len(idxs)} selected row(s) — outliers deviate from mean")

    # ── Pick event: click a line → jump to table ──────────────────────────────
    def _on_pick4(self, event):
        artist = event.artist
        if artist not in self._pick_map:
            return
        crop, sub, row_idx = self._pick_map[artist]
        tv = self._sub_tv_map.get((crop, sub))
        if tv is None:
            return
        iid = str(row_idx)
        try:
            show_fn = self._sub_show_fns.get((crop, sub))
            if show_fn:
                show_fn()
            tv.selection_set(iid); tv.see(iid); tv.focus(iid)
            self.cl4_canvas.update_idletasks()
            y    = tv.winfo_rooty() - self.cl4_canvas.winfo_rooty()
            frac = max(0, y / max(self.cl4_inner.winfo_height(), 1))
            self.cl4_canvas.yview_moveto(frac)
        except Exception:
            pass
        self.lbl_row_hint.config(text=f"↑ Row {row_idx} selected in {sub}")

    # ── Main NDVI cluster plot ────────────────────────────────────────────────
    def _cl4_draw_plot(self, crop, highlight_idxs=None, highlight_sub=None):
        self.ax4.clear()
        self._style_ax(self.ax4, self.fig4)
        self._pick_map = {}

        if crop not in self.model.clusters or not self.model.ndvi_cols:
            self.cv4.draw(); return

        subs  = self.model.clusters[crop]["subclusters"]
        x     = list(range(len(self.model.ndvi_cols)))
        xlbls = self.model.dates if self.model.dates else [str(i) for i in x]

        for i, sub in enumerate(subs):
            if not self.model.cluster_visibility.get(sub, True):
                continue   # Skip hidden clusters

            mc  = self.model.mean_curve(crop, sub)
            clr = PALETTE[i % len(PALETTE)]
            if len(mc) == 0:
                continue

            self.ax4.plot(x, mc, "o-", color=clr, lw=2, markersize=5,
                          label=sub, zorder=3, alpha=0.9)

        # row overlays
        if highlight_idxs is not None and highlight_sub is not None:
            all_idxs = self.model.clusters[crop]["subclusters"].get(highlight_sub, [])
            for row_idx in all_idxs:
                if row_idx in highlight_idxs:
                    continue
                vals = self.model.row_curve(row_idx)
                if np.all(np.isnan(vals)):
                    continue
                line, = self.ax4.plot(x, vals, "-", color="#BBBBBB",
                                      lw=0.8, alpha=0.5, zorder=2, picker=5)
                self._pick_map[line] = (crop, highlight_sub, row_idx)
            for k, row_idx in enumerate(highlight_idxs):
                vals = self.model.row_curve(row_idx)
                if np.all(np.isnan(vals)):
                    continue
                line, = self.ax4.plot(x, vals, "D-", color=C["highlight"],
                                      lw=2.2, markersize=6, alpha=0.92,
                                      label=f"Row {row_idx}", zorder=6, picker=5)
                self._pick_map[line] = (crop, highlight_sub, row_idx)

        self.ax4.set_xticks(x)
        self.ax4.set_xticklabels(xlbls, rotation=45, ha="right", fontsize=6)
        self.ax4.set_title(f"{crop} — NDVI time-series", fontsize=9,
                           color=C["text"], pad=5)
        self.ax4.set_xlabel("Date", fontsize=8, color=C["sub"])
        self.ax4.set_ylabel("NDVI", fontsize=8, color=C["sub"])
        self.ax4.legend(fontsize=7, loc="best", framealpha=0.9)
        self.ax4.grid(True, color=C["border"], lw=0.5, ls="--", zorder=1)
        self.fig4.tight_layout(pad=1.2)
        self.cv4.draw()

    def _cl4_update_listbox(self, crop):
        crops = list(self.model.clusters.keys())
        if crop not in crops:
            return
        idx    = crops.index(crop)
        n_rows = len(self.model.clusters[crop]["all_idx"])
        n_cl   = self.model.clusters[crop]["n_done"]
        lbl    = (f"✓ {crop}  ({n_rows}r, {n_cl}cl)"
                  if n_cl > 0 else f"{crop}  ({n_rows} rows)")
        self.cl_lb.delete(idx); self.cl_lb.insert(idx, lbl)
        self.cl_lb.selection_set(idx)

    def _cl4_update_summary(self):
        total   = len(self.model.clusters)
        done    = sum(1 for v in self.model.clusters.values() if v["n_done"] > 0)
        deleted = sum(len(v.get("deleted", [])) for v in self.model.clusters.values())
        self.lbl_cl4_summary.config(
            text=f"{done}/{total} crops clustered\n{deleted} rows deleted")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5 — Review & Export
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step5(self, parent):
        f = tk.Frame(parent, bg=C["bg"])

        top = card(f, "Step 5 — Review All Clusters & Export", fill="x")

        # summary chips
        self.frm_s5_chips = tk.Frame(top, bg=C["panel"])
        self.frm_s5_chips.pack(fill="x", pady=(0, 8))

        tk.Label(top,
                 text="Export saves three files:\n"
                      "  gt_cleaned.csv / .xlsx  — all rows with a valid Crop_Cluster label\n"
                      "  gt_rejected.csv / .xlsx — auto-rejected + manually deleted rows\n"
                      "  gt_full.csv / .xlsx      — complete dataset with Crop_Cluster column\n"
                      "  cluster_summary.json    — cluster sizes",
                 bg=C["panel"], fg=C["sub"], font=FONT_SMALL, justify="left").pack(anchor="w")

        # output folder
        out_row = tk.Frame(top, bg=C["panel"]); out_row.pack(fill="x", pady=(8, 0))
        tk.Label(out_row, text="Output folder:", bg=C["panel"], font=FONT_BODY).pack(side="left")
        self.var_out = tk.StringVar()
        tk.Entry(out_row, textvariable=self.var_out, width=48,
                 font=FONT_SMALL).pack(side="left", padx=(8, 8))
        ghost_btn(out_row, "📁  Browse", self._browse_out).pack(side="left")

        self.lbl_exp = tk.Label(top, text="", bg=C["panel"],
                                fg=C["success"], font=FONT_SMALL)
        self.lbl_exp.pack(anchor="w", pady=(4, 0))

        # All-crop NDVI overview plot
        plot_card = card(f, "NDVI Mean Curves — All Clusters",
                         padx=6, pady=6, fill="both", expand=True, pady_outer=(8, 0))
        btn_row = tk.Frame(plot_card, bg=C["panel"]); btn_row.pack(fill="x", pady=(0, 4))
        ghost_btn(btn_row, "📈  Refresh Overview Plot", self._draw_overview_plot).pack(side="left")

        self.fig5 = Figure(figsize=(10, 4), dpi=90, facecolor=C["panel"])
        self.ax5  = self.fig5.add_subplot(111)
        self._style_ax(self.ax5, self.fig5)
        self.cv5  = FigureCanvasTkAgg(self.fig5, master=plot_card)
        self.cv5.get_tk_widget().pack(fill="both", expand=True)
        return f

    def _refresh_review(self):
        self._update_s5_chips()
        self._draw_overview_plot()

    def _update_s5_chips(self):
        for w in self.frm_s5_chips.winfo_children():
            w.destroy()
        m = self.model
        df = m.df if m.df is not None else pd.DataFrame()
        n_total = len(df)
        n_rej   = (df.get("ndvi_flag", pd.Series(dtype=str)) == "auto_rejected").sum()
        n_del   = sum(len(v.get("deleted", [])) for v in m.clusters.values())
        n_clean = n_total - n_rej - n_del
        for txt, clr in [(f"Total rows: {n_total}", C["text"]),
                         (f"Clean: {n_clean}", C["success"]),
                         (f"Auto-rejected: {n_rej}", C["warn"]),
                         (f"Manually deleted: {n_del}", C["danger"])]:
            tk.Label(self.frm_s5_chips, text=txt, bg=C["acc_light"], fg=clr,
                     font=("Segoe UI", 9, "bold"), padx=10, pady=4).pack(side="left", padx=4)

    def _draw_overview_plot(self):
        self.ax5.clear()
        self._style_ax(self.ax5, self.fig5)
        m = self.model
        if not m.ndvi_cols or not m.clusters:
            self.cv5.draw(); return
        x     = list(range(len(m.ndvi_cols)))
        xlbls = m.dates if m.dates else [str(i) for i in x]
        k = 0
        for crop, cv in m.clusters.items():
            for sub, idxs in cv["subclusters"].items():
                if not m.cluster_visibility.get(sub, True):
                    continue   # Hide unchecked clusters

                if not idxs:
                    continue
                mc  = m.mean_curve(crop, sub)
                clr = PALETTE[k % len(PALETTE)]
                k  += 1
                self.ax5.plot(x, mc, "o-", color=clr, lw=1.8,
                              markersize=4, label=sub, zorder=3)
        self.ax5.set_xticks(x)
        self.ax5.set_xticklabels(xlbls, rotation=45, ha="right", fontsize=7)
        self.ax5.set_title("Mean NDVI Time-Series — All Clusters",
                           fontsize=10, color=C["text"])
        self.ax5.set_xlabel("Date", fontsize=9, color=C["sub"])
        self.ax5.set_ylabel("Mean NDVI", fontsize=9, color=C["sub"])
        self.ax5.legend(fontsize=7, loc="best", framealpha=0.9,
                        ncol=max(1, k // 8))
        self.ax5.grid(True, color=C["border"], lw=0.5, ls="--")
        self.fig5.tight_layout(pad=1.4)
        self.cv5.draw()

    def _browse_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.var_out.set(d)

    def _do_export(self):
        out = self.var_out.get().strip()
        if not out:
            out = filedialog.askdirectory(title="Select output folder")
            if not out:
                return
            self.var_out.set(out)
        try:
            saved = self.model.export(out)
        except Exception as e:
            messagebox.showerror("Export Error", str(e)); return
        names = "\n".join(f"✓ {os.path.basename(p)}" for p in saved)
        self.lbl_exp.config(text=f"Saved to: {out}\n{names}")
        messagebox.showinfo("Export Complete",
                            f"{len(saved)} file(s) saved:\n{names}\n\nFolder: {out}")

    # ── Shared ax styling ─────────────────────────────────────────────────────
    @staticmethod
    def _style_ax(ax, fig):
        ax.set_facecolor(C["panel"])
        ax.tick_params(labelsize=7, colors=C["sub"])
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        fig.tight_layout(pad=1.4)

    def _plot_single_curve(self, row_idx, ax, cv, fig, title=""):
        ax.clear()
        self._style_ax(ax, fig)
        if not self.model.ndvi_cols:
            cv.draw(); return
        vals  = self.model.row_curve(row_idx)
        x     = list(range(len(vals)))
        xlbls = self.model.dates if self.model.dates else [str(i) for i in x]
        ax.plot(x, vals, "o-", color=C["highlight"], lw=2, markersize=5)
        ax.axhline(self.model.max_ndvi_thresh, color=C["warn"], lw=1, ls="--",
                   label=f"max_thresh={self.model.max_ndvi_thresh}")
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(title, fontsize=9, color=C["text"])
        ax.set_ylabel("NDVI", fontsize=8, color=C["sub"])
        ax.legend(fontsize=7)
        ax.grid(True, color=C["border"], lw=0.5, ls="--")
        fig.tight_layout(pad=1.2)
        cv.draw()


# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()