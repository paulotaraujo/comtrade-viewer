#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QScrollArea, QMessageBox,
    QDoubleSpinBox, QSplitter, QGroupBox, QTabWidget, QStackedWidget, QFrame,
    QDialog, QListWidget, QListWidgetItem, QComboBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# =============================
# Modelos COMTRADE
# =============================
@dataclass
class AnaDef:
    idx: int
    name: str
    phase: str
    units: str
    a: float
    b: float
    skew: float
    min_level: int
    max_level: int
    pri: float
    sec: float
    pri_sec: str  # 'P' ou 'S'


@dataclass
class Cfg:
    station: str
    revision: str
    version: str
    total: int
    na: int
    nd: int
    freq: float
    fs: float
    samples: int
    start_dt: str
    end_dt: str
    file_type: str  # ASCII/BINARY
    time_mult: float
    analogs: List[AnaDef]
    digitals: List[str]


# =============================
# Utilidades parsing
# =============================
def _digits_only_int(s: str) -> int:
    s = (s or "").strip()
    out = "".join([c for c in s if c.isdigit() or c == "-"])
    return int(out) if out else 0


def _safe_float(s: str, default: float) -> float:
    try:
        if s is None:
            return default
        ss = str(s).strip()
        if not ss or ss.lower() == "nan":
            return default
        return float(ss)
    except Exception:
        return default


def _safe_int(s: str, default: int) -> int:
    try:
        if s is None:
            return default
        ss = str(s).strip()
        if not ss:
            return default
        return int(float(ss))
    except Exception:
        return default


def parse_cfg(cfg_path: Path) -> Cfg:
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = [[x.strip() for x in ln.split(",")] for ln in lines]

    if len(rows) < 6:
        raise ValueError("CFG muito curto/ inválido.")

    station = rows[0][0] if rows[0] else ""
    revision = rows[0][1] if len(rows[0]) > 1 else ""
    version = rows[0][2] if len(rows[0]) > 2 and rows[0][2] else "1999"

    total = _digits_only_int(rows[1][0]) if len(rows) > 1 else 0
    na = _digits_only_int(rows[1][1]) if len(rows) > 1 and len(rows[1]) > 1 else 0
    nd = _digits_only_int(rows[1][2]) if len(rows) > 1 and len(rows[1]) > 2 else 0

    chan_start = 2
    analog_rows = rows[chan_start:chan_start + na]
    digital_rows = rows[chan_start + na:chan_start + na + nd]

    analogs: List[AnaDef] = []
    for r in analog_rows:
        idx = _safe_int(r[0] if len(r) > 0 else "", 0)
        name = r[1] if len(r) > 1 else f"AN{idx}"
        phase = r[2] if len(r) > 2 else ""
        units = r[4] if len(r) > 4 else ""

        a = _safe_float(r[5] if len(r) > 5 else "", float("nan"))
        b = _safe_float(r[6] if len(r) > 6 else "", 0.0)
        skew = _safe_float(r[7] if len(r) > 7 else "", 0.0)
        min_level = _safe_int(r[8] if len(r) > 8 else "", -32768)
        max_level = _safe_int(r[9] if len(r) > 9 else "", 32767)
        pri = _safe_float(r[10] if len(r) > 10 else "", 1.0)
        sec = _safe_float(r[11] if len(r) > 11 else "", 1.0)
        pri_sec = (r[12].strip() if len(r) > 12 and r[12] else "S").upper()

        analogs.append(AnaDef(idx, name, phase, units, a, b, skew, min_level, max_level, pri, sec, pri_sec))

    digitals: List[str] = []
    for r in digital_rows:
        digitals.append(r[1] if len(r) > 1 else "")

    p = chan_start + na + nd

    freq = 60.0
    if p < len(rows) and len(rows[p]) >= 1:
        freq = _safe_float(rows[p][0], 60.0)
    p += 1

    nrates = 0
    if p < len(rows) and len(rows[p]) >= 1:
        nrates = _safe_int(rows[p][0], 0)
    p += 1

    fs = 0.0
    samples = 0
    sample_rates: List[Tuple[float, int]] = []
    if nrates > 0:
        for _ in range(nrates):
            if p >= len(rows):
                break
            fs_k = _safe_float(rows[p][0] if len(rows[p]) > 0 else "", 0.0)
            end_k = _safe_int(rows[p][1] if len(rows[p]) > 1 else "", 0)
            if fs_k > 0 and end_k > 0:
                sample_rates.append((fs_k, end_k))
            p += 1

    if sample_rates:
        fs = sample_rates[0][0]
        samples = sample_rates[-1][1]

    start_dt = ""
    end_dt = ""
    if p < len(rows) and len(rows[p]) >= 2:
        start_dt = f"{rows[p][0]} {rows[p][1]}"
    p += 1
    if p < len(rows) and len(rows[p]) >= 2:
        end_dt = f"{rows[p][0]} {rows[p][1]}"
    p += 1

    file_type = ""
    if p < len(rows) and len(rows[p]) >= 1:
        ft = rows[p][0].upper()
        if ft in ("ASCII", "BINARY"):
            file_type = ft
    p += 1
    if not file_type:
        for r in rows:
            if r and r[0].upper() in ("ASCII", "BINARY"):
                file_type = r[0].upper()
                break
    if not file_type:
        raise ValueError("CFG inválido: não encontrei 'ASCII' ou 'BINARY'.")

    time_mult = 1.0
    if p < len(rows) and len(rows[p]) >= 1:
        time_mult = _safe_float(rows[p][0], 1.0)
    else:
        for i, r in enumerate(rows):
            if r and r[0].upper() in ("ASCII", "BINARY"):
                if i + 1 < len(rows) and len(rows[i + 1]) >= 1:
                    time_mult = _safe_float(rows[i + 1][0], 1.0)
                break

    return Cfg(
        station=station,
        revision=revision,
        version=version,
        total=total,
        na=na,
        nd=nd,
        freq=freq,
        fs=fs,
        samples=samples,
        start_dt=start_dt,
        end_dt=end_dt,
        file_type=file_type,
        time_mult=time_mult,
        analogs=analogs,
        digitals=digitals,
    )


# =============================
# Escalonamento
# =============================
def scale_analogs(cfg: Cfg, analog_raw: np.ndarray, apply_pri_sec: bool = True) -> Dict[str, np.ndarray]:
    """
    1) clamp
    2) y = aX + b
    3) opcional PRI/SEC (converte tudo para PRIMÁRIO):
       - se PS == 'S' -> y *= (PRI/SEC)
       - se PS == 'P' -> mantém
    """
    out: Dict[str, np.ndarray] = {}
    for i, adef in enumerate(cfg.analogs):
        x = analog_raw[:, i].astype(np.float64)
        x = np.clip(x, adef.min_level, adef.max_level)

        y = x * adef.a + adef.b

        if apply_pri_sec:
            ps = (adef.pri_sec or "S").upper()
            pri = float(adef.pri if adef.pri is not None else 1.0)
            sec = float(adef.sec if adef.sec is not None else 1.0)
            if abs(sec) < 1e-18:
                sec = 1.0
            if ps == "S":
                y = y * (pri / sec)

        out[adef.name] = y
    return out


# =============================
# Leitura ASCII / BINARY
# =============================
def read_dat_ascii(dat_path: Path, cfg: Cfg) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(str(dat_path), delimiter=",", dtype=float, ndmin=2)
    expected = 2 + cfg.na + cfg.nd
    if data.shape[1] != expected:
        raise ValueError(f"ASCII .dat com colunas inesperadas: {data.shape[1]} (esperado {expected}).")

    t = (data[:, 1].astype(np.float64) * cfg.time_mult) * 1e-6
    t -= t[0]
    analog_raw = data[:, 2:2 + cfg.na].astype(np.int32)
    return t, analog_raw


def _try_read_binary(blob: bytes, na: int, nd: int, ts_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    nwords = int(math.ceil(nd / 16))
    rec_fmt = "<i" + ts_mode + ("h" * na) + ("H" * nwords)
    rec_size = struct.calcsize(rec_fmt)

    nrec = len(blob) // rec_size
    if nrec <= 0 or (nrec * rec_size) != len(blob):
        raise ValueError("Tamanho não compatível com este layout.")

    t_raw = np.empty(nrec, dtype=np.int64 if ts_mode == "q" else np.int32)
    analog_raw = np.empty((nrec, na), dtype=np.int16)

    off = 0
    for i in range(nrec):
        rec = struct.unpack_from(rec_fmt, blob, off)
        off += rec_size
        t_raw[i] = rec[1]
        analog_raw[i, :] = rec[2:2 + na]

    return t_raw.astype(np.int64), analog_raw.astype(np.int32)


def read_bdat_binary(bdat_path: Path, cfg: Cfg) -> Tuple[np.ndarray, np.ndarray]:
    blob = bdat_path.read_bytes()

    try:
        t_raw, analog_raw = _try_read_binary(blob, cfg.na, cfg.nd, ts_mode="i")
    except Exception:
        t_raw, analog_raw = _try_read_binary(blob, cfg.na, cfg.nd, ts_mode="q")

    t_ts = (t_raw.astype(np.float64) * cfg.time_mult) * 1e-6
    t_ts -= t_ts[0]

    span = float(np.max(t_ts) - np.min(t_ts)) if len(t_ts) else 0.0
    deltas = np.diff(t_ts) if len(t_ts) > 1 else np.array([], dtype=np.float64)
    zeros_ratio = float(np.mean(np.isclose(deltas, 0.0))) if deltas.size else 1.0

    if span <= 1e-9 or zeros_ratio > 0.95:
        fs = cfg.fs if cfg.fs and cfg.fs > 0 else 0.0
        if fs <= 0.0:
            raise ValueError("Não foi possível determinar fs no CFG para reconstruir o tempo.")
        t = np.arange(len(t_raw), dtype=np.float64) / fs
    else:
        t = t_ts

    return t, analog_raw


def classify_va(cfg: Cfg) -> Tuple[List[str], List[str], List[str]]:
    voltage, current, others = [], [], []
    for adef in cfg.analogs:
        unit = (adef.units or "").strip().upper()
        if "V" in unit and "A" not in unit:
            voltage.append(adef.name)
        elif "A" in unit:
            current.append(adef.name)
        else:
            others.append(adef.name)

    if not voltage and not current:
        voltage, current, others = [], [], []
        for adef in cfg.analogs:
            nm = (adef.name or "").upper()
            if nm.startswith("V") or nm.startswith("U"):
                voltage.append(adef.name)
            elif nm.startswith("I"):
                current.append(adef.name)
            else:
                others.append(adef.name)

    return voltage, current, others


# =============================
# Unidades -> ylabel
# =============================
def _norm_unit(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    u = u.replace(" ", "")
    u = u.replace("VOLTS", "V").replace("VOLT", "V")
    u = u.replace("AMPS", "A").replace("AMP", "A")
    return u


def units_for_names(cfg: Cfg, names: List[str]) -> List[str]:
    if cfg is None:
        return []
    m = {a.name: _norm_unit(a.units) for a in cfg.analogs}
    units: List[str] = []
    for n in names:
        u = m.get(n, "")
        if u and u not in units:
            units.append(u)
    return units


def format_ylabel(base: str, units: List[str]) -> str:
    units = [u for u in units if u]
    if not units:
        return base
    if len(units) == 1:
        return f"{base} ({units[0]})"
    return f"{base} ({' / '.join(units)})"


# =============================
# Export helpers (CFG/DAT/BDAT)
# =============================
def _split_dt(dt: str) -> Tuple[str, str]:
    dt = (dt or "").strip()
    if not dt:
        return "", ""
    parts = dt.split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], ""


def _fmt_float(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "0.0"
    return f"{float(x):.12g}"


def write_cfg(cfg: Cfg, out_cfg: Path):
    lines: List[str] = []
    lines.append(f"{cfg.station},{cfg.revision},{cfg.version}")
    lines.append(f"{cfg.total},{cfg.na}A,{cfg.nd}D")

    for a in cfg.analogs:
        row = [
            str(int(a.idx)),
            a.name or "",
            a.phase or "",
            "",  # ccbm
            a.units or "",
            _fmt_float(a.a),
            _fmt_float(a.b),
            _fmt_float(a.skew),
            str(int(a.min_level)),
            str(int(a.max_level)),
            _fmt_float(a.pri),
            _fmt_float(a.sec),
            (a.pri_sec or "P").upper(),
        ]
        lines.append(",".join(row))

    for i, name in enumerate(cfg.digitals, start=1):
        lines.append(f"{i},{name}")

    lines.append(_fmt_float(cfg.freq))

    fs = float(cfg.fs) if cfg.fs and cfg.fs > 0 else 0.0
    if fs <= 0.0:
        fs = 1.0
    lines.append("1")
    lines.append(f"{_fmt_float(fs)},{int(cfg.samples)}")

    sd, st = _split_dt(cfg.start_dt)
    ed, et = _split_dt(cfg.end_dt)
    lines.append(f"{sd},{st}")
    lines.append(f"{ed},{et}")

    lines.append((cfg.file_type or "ASCII").upper())
    lines.append(_fmt_float(cfg.time_mult if cfg.time_mult else 1.0))

    out_cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dat_ascii(out_dat: Path, t: np.ndarray, analog_raw: np.ndarray, time_mult: float):
    if t.ndim != 1:
        t = np.asarray(t).reshape(-1)
    n = int(t.shape[0])
    if n <= 0:
        raise ValueError("Sem amostras para exportar.")

    tm = float(time_mult) if time_mult and abs(time_mult) > 1e-18 else 1.0
    ts = np.rint((t * 1e6) / tm).astype(np.int64)
    smp = (np.arange(n, dtype=np.int64) + 1)

    analog_raw = np.asarray(analog_raw)
    if analog_raw.shape[0] != n:
        raise ValueError("t e analog_raw com tamanhos diferentes.")
    if analog_raw.ndim != 2:
        analog_raw = analog_raw.reshape(n, -1)

    mat = np.column_stack([smp, ts, analog_raw.astype(np.int64)])

    np.savetxt(str(out_dat), mat, fmt="%d", delimiter=",", newline="\n")


def write_bdat_binary(out_bdat: Path, t: np.ndarray, analog_raw: np.ndarray, cfg: Cfg, ts_mode: str = "i"):
    """
    Layout compatível com o seu reader:
      <i (sample#) + <ts_mode (timestamp) + na*h (analogs) + nwords*H (digitals)
    timestamp base:
      t(s) = timestamp * time_mult * 1e-6  =>  timestamp = round(t * 1e6 / time_mult)
    """
    if t.ndim != 1:
        t = np.asarray(t).reshape(-1)
    n = int(t.shape[0])
    if n <= 0:
        raise ValueError("Sem amostras para exportar.")

    analog_raw = np.asarray(analog_raw)
    if analog_raw.ndim != 2:
        analog_raw = analog_raw.reshape(n, -1)
    if analog_raw.shape[0] != n:
        raise ValueError("t e analog_raw com tamanhos diferentes.")
    if analog_raw.shape[1] != cfg.na:
        raise ValueError(f"analog_raw tem {analog_raw.shape[1]} canais, cfg.na={cfg.na}.")

    nd = int(cfg.nd) if cfg.nd else 0
    nwords = int(math.ceil(nd / 16)) if nd > 0 else 0

    tm = float(cfg.time_mult) if cfg.time_mult and abs(cfg.time_mult) > 1e-18 else 1.0
    ts = np.rint((t * 1e6) / tm)

    if ts_mode == "i":
        ts = np.clip(ts, -2**31, 2**31 - 1).astype(np.int32)
    elif ts_mode == "q":
        ts = ts.astype(np.int64)
    else:
        raise ValueError("ts_mode deve ser 'i' ou 'q'.")

    smp = (np.arange(n, dtype=np.int32) + 1)

    dig_words = np.zeros((n, nwords), dtype=np.uint16) if nwords > 0 else None

    rec_fmt = "<i" + ts_mode + ("h" * cfg.na) + ("H" * nwords)
    with out_bdat.open("wb") as f:
        for i in range(n):
            anal = np.clip(analog_raw[i, :], -32768, 32767).astype(np.int16)
            if nwords > 0:
                rec = (int(smp[i]), int(ts[i]), *anal.tolist(), *dig_words[i, :].tolist())
            else:
                rec = (int(smp[i]), int(ts[i]), *anal.tolist())
            f.write(struct.pack(rec_fmt, *rec))


# =============================
# UI helpers
# =============================
class ChannelRow(QWidget):
    """Linha: swatch (linha colorida) + checkbox; desmarcado fica 'apagado'."""
    def __init__(self, name: str, on_toggle: Callable[[], None]):
        super().__init__()
        self.name = name
        self.on_toggle = on_toggle

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(8)

        self.swatch = QFrame()
        self.swatch.setFixedSize(26, 3)
        self.swatch.setFrameShape(QFrame.NoFrame)

        self.cb = QCheckBox(name)
        self.cb.setChecked(True)
        self.cb.stateChanged.connect(self._changed)

        lay.addWidget(self.swatch, 0, Qt.AlignVCenter)
        lay.addWidget(self.cb, 1)

        self._rgb = (0, 0, 0)
        self.apply_style()

    def set_color(self, color: str):
        if isinstance(color, str) and color.startswith("#") and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            self._rgb = (r, g, b)
        self.apply_style()

    def is_checked(self) -> bool:
        return self.cb.isChecked()

    def set_checked(self, state: bool):
        self.cb.setChecked(state)
        self.apply_style()

    def _changed(self, _):
        self.apply_style()
        self.on_toggle()

    def apply_style(self):
        r, g, b = self._rgb
        if self.cb.isChecked():
            self.cb.setStyleSheet("")
            self.swatch.setStyleSheet(f"background-color: rgb({r},{g},{b});")
        else:
            self.cb.setStyleSheet("color: rgba(0,0,0,130);")
            self.swatch.setStyleSheet(f"background-color: rgba({r},{g},{b},120);")


class ChannelBox(QGroupBox):
    def __init__(self, title: str, empty_msg: str, on_toggle: Callable[[], None]):
        super().__init__(title)
        self._on_toggle = on_toggle
        self._empty_msg = empty_msg
        self.rows: Dict[str, ChannelRow] = {}

        lay = QVBoxLayout(self)

        row_btn = QHBoxLayout()
        self.btn_all = QPushButton("Marcar todos")
        self.btn_none = QPushButton("Desmarcar todos")
        self.btn_all.setEnabled(False)
        self.btn_none.setEnabled(False)
        row_btn.addWidget(self.btn_all)
        row_btn.addWidget(self.btn_none)
        lay.addLayout(row_btn)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(4, 4, 4, 4)
        self.scroll_layout.setSpacing(2)

        self.empty_label = QLabel(self._empty_msg)
        self.empty_label.setStyleSheet("color: rgba(0,0,0,140);")
        self.empty_label.setWordWrap(True)
        self.scroll_layout.addWidget(self.empty_label)

        self.scroll_layout.addStretch(1)
        self.scroll.setWidget(self.scroll_content)
        lay.addWidget(self.scroll, 1)

        self.btn_all.clicked.connect(lambda: self.set_all(True))
        self.btn_none.clicked.connect(lambda: self.set_all(False))

    def set_channels(self, names: List[str]):
        for rw in self.rows.values():
            rw.deleteLater()
        self.rows.clear()

        while self.scroll_layout.count() > 1:
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not names:
            self.empty_label = QLabel(self._empty_msg)
            self.empty_label.setStyleSheet("color: rgba(0,0,0,140);")
            self.empty_label.setWordWrap(True)
            self.scroll_layout.insertWidget(0, self.empty_label)
            self.btn_all.setEnabled(False)
            self.btn_none.setEnabled(False)
            return

        self.btn_all.setEnabled(True)
        self.btn_none.setEnabled(True)

        for n in names:
            rw = ChannelRow(n, self._on_toggle)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, rw)
            self.rows[n] = rw

    def set_color_map(self, colors: Dict[str, str]):
        for n, rw in self.rows.items():
            c = colors.get(n)
            if c:
                rw.set_color(c)

    def set_all(self, state: bool):
        for rw in self.rows.values():
            rw.cb.blockSignals(True)
            rw.set_checked(state)
            rw.cb.blockSignals(False)
        self._on_toggle()

    def is_visible(self, name: str) -> bool:
        rw = self.rows.get(name)
        return bool(rw.is_checked()) if rw else False


# =============================
# Toolbar custom (FIX real do Home)
# =============================
class CustomToolbar(NavigationToolbar):
    """
    O QAction do 'Home' fica ligado ao método da toolbar criado no __init__ dela.
    Trocar `toolbar.home = ...` NÃO muda o slot do botão.
    Então a correção correta é sobrescrever home() via subclasse.
    """

    def __init__(self, canvas, parent, home_cb: Callable[..., None]):
        super().__init__(canvas, parent)
        self._home_cb = home_cb

    def home(self, *args, **kwargs):
        # 1) comportamento nativo (sai de pan/zoom + mexe no nav stack)
        try:
            super().home(*args, **kwargs)
        except Exception:
            pass

        # 2) garante layout correto do app (foco/restaurar) + limites corretos
        if callable(self._home_cb):
            self._home_cb(from_toolbar=True)


# =============================
# Main Window
# =============================
class ComtradeViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COMTRADE Viewer")

        self.cfg: Optional[Cfg] = None
        self.t: Optional[np.ndarray] = None
        self.analog_raw: Optional[np.ndarray] = None
        self.analog_scaled: Dict[str, np.ndarray] = {}

        self.voltage_names: List[str] = []
        self.current_names: List[str] = []

        self.t0: float = 0.0
        self.t1: float = 0.0

        self.lines_v: Dict[str, object] = {}
        self.lines_i: Dict[str, object] = {}
        self._colors: Dict[str, str] = {}

        self.vline_v = None
        self.vline_i = None
        self._cursor_syncing = False

        # ---- Foco/restore (estado + posições default)
        self._focus_mode: str = "both"  # "both" | "v" | "i"
        self._pos_v_default = None
        self._pos_i_default = None
        self._pos_union = None
        self._captured_positions = False

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # Top bar
        top = QHBoxLayout()
        self.btn_open = QPushButton("Abrir")
        self.btn_open.clicked.connect(self.open_files)
        self.info = QLabel("Nenhum arquivo carregado.")
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.btn_open)
        top.addWidget(self.info, 1)
        main.addLayout(top)

        # Time controls + PRI/SEC
        row = QHBoxLayout()
        row.addWidget(QLabel("tmin:"))
        self.sp_tmin = QDoubleSpinBox()
        self.sp_tmin.setDecimals(6)
        self.sp_tmin.setRange(-1e12, 1e12)
        self.sp_tmin.setSingleStep(0.001)
        self.sp_tmin.setEnabled(False)
        row.addWidget(self.sp_tmin)

        row.addWidget(QLabel("tmax:"))
        self.sp_tmax = QDoubleSpinBox()
        self.sp_tmax.setDecimals(6)
        self.sp_tmax.setRange(-1e12, 1e12)
        self.sp_tmax.setSingleStep(0.001)
        self.sp_tmax.setEnabled(False)
        row.addWidget(self.sp_tmax)

        self.btn_apply = QPushButton("Aplicar")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.apply_time_window)
        row.addWidget(self.btn_apply)

        self.btn_reset_time = QPushButton("Reset tempo")
        self.btn_reset_time.setEnabled(False)
        self.btn_reset_time.clicked.connect(self.reset_time_range)
        row.addWidget(self.btn_reset_time)

        self.cb_pri_sec = QCheckBox("PRI/SEC")
        self.cb_pri_sec.setChecked(True)
        self.cb_pri_sec.setEnabled(False)
        self.cb_pri_sec.stateChanged.connect(self.on_pri_sec_changed)
        row.addWidget(self.cb_pri_sec)

        row.addStretch(1)
        main.addLayout(row)

        # Splitter: left tabs / right plot
        splitter = QSplitter(Qt.Horizontal)
        main.addWidget(splitter, 1)

        # LEFT
        left = QWidget()
        left_lay = QVBoxLayout(left)

        self.tabs = QTabWidget()
        self.box_v = ChannelBox(
            "Canais de Tensão",
            "Nenhuma variável de tensão foi encontrada.",
            self.on_toggles_changed
        )
        self.box_i = ChannelBox(
            "Canais de Corrente",
            "Nenhuma variável de corrente foi encontrada.",
            self.on_toggles_changed
        )

        tab_v = QWidget()
        tab_v_lay = QVBoxLayout(tab_v)
        tab_v_lay.setContentsMargins(0, 0, 0, 0)
        tab_v_lay.addWidget(self.box_v, 1)

        tab_i = QWidget()
        tab_i_lay = QVBoxLayout(tab_i)
        tab_i_lay.setContentsMargins(0, 0, 0, 0)
        tab_i_lay.addWidget(self.box_i, 1)

        self.tabs.addTab(tab_v, "Tensão")
        self.tabs.addTab(tab_i, "Corrente")
        left_lay.addWidget(self.tabs, 1)

        # BOTÃO EXPORTAR COMO...
        self.btn_export_as = QPushButton("Exportar como...")
        self.btn_export_as.setEnabled(False)
        self.btn_export_as.clicked.connect(self.export_as_dialog)
        left_lay.addWidget(self.btn_export_as)

        splitter.addWidget(left)

        # RIGHT: placeholder + plot page
        right = QWidget()
        right_lay = QVBoxLayout(right)

        self.plot_stack = QStackedWidget()
        self.page_empty = QLabel("Carregue um arquivo COMTRADE (.cfg + .dat/.bdat) para visualizar os gráficos.")
        self.page_empty.setAlignment(Qt.AlignCenter)
        self.page_empty.setStyleSheet("color: rgba(0,0,0,140);")

        self.page_plot = QWidget()
        page_plot_lay = QVBoxLayout(self.page_plot)
        page_plot_lay.setContentsMargins(0, 0, 0, 0)
        page_plot_lay.setSpacing(4)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.08, hspace=0.28)

        self.ax_v = self.fig.add_subplot(211)
        self.ax_i = self.fig.add_subplot(212, sharex=self.ax_v)

        # --- toolbar + botões foco
        self.toolbar = CustomToolbar(self.canvas, self, self.on_home_clicked)

        bar = QWidget()
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(0, 0, 0, 0)
        bar_lay.setSpacing(8)
        bar_lay.addWidget(self.toolbar, 0)

        bar_lay.addStretch(1)

        self.lbl_focus_mode = QLabel("Visualizar")
        self.lbl_focus_mode.setEnabled(False)  # acompanha o estado do combo

        self.cb_focus_mode = QComboBox()
        self.cb_focus_mode.addItems(["Todos", "Tensão analógico", "Corrente analógico"])
        self.cb_focus_mode.setEnabled(False)
        self.cb_focus_mode.currentIndexChanged.connect(self.on_focus_mode_changed)

        bar_lay.addWidget(self.lbl_focus_mode, 0, Qt.AlignVCenter)
        bar_lay.addWidget(self.cb_focus_mode, 0)

        page_plot_lay.addWidget(bar)
        page_plot_lay.addWidget(self.canvas, 1)

        self.plot_stack.addWidget(self.page_empty)
        self.plot_stack.addWidget(self.page_plot)
        self.plot_stack.setCurrentWidget(self.page_empty)

        right_lay.addWidget(self.plot_stack, 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Cursor
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("figure_leave_event", lambda ev: self.clear_cursor())

        # X sincronizado
        self._syncing_xlim = False
        self.ax_v.callbacks.connect("xlim_changed", lambda ax: self._sync_x_from(self.ax_v))
        self.ax_i.callbacks.connect("xlim_changed", lambda ax: self._sync_x_from(self.ax_i))

    # ---------- helpers seleção ----------
    def selected_names(self) -> List[str]:
        names: List[str] = []
        for n in self.voltage_names:
            if self.box_v.is_visible(n):
                names.append(n)
        for n in self.current_names:
            if self.box_i.is_visible(n):
                names.append(n)

        out: List[str] = []
        seen = set()
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out

    def _current_time_window(self) -> Tuple[float, float]:
        tmin = float(self.sp_tmin.value())
        tmax = float(self.sp_tmax.value())
        if tmax < tmin:
            tmin, tmax = tmax, tmin
        if self.t is None:
            return tmin, tmax
        tmin = max(tmin, self.t0)
        tmax = min(tmax, self.t1)
        return tmin, tmax

    # ---------- Foco/Restore (layout robusto) ----------
    def _capture_default_positions_if_needed(self):
        if self._captured_positions:
            return
        try:
            pv = self.ax_v.get_position()
            pi = self.ax_i.get_position()
            self._pos_v_default = pv.frozen()
            self._pos_i_default = pi.frozen()
            x0 = min(pv.x0, pi.x0)
            y0 = min(pv.y0, pi.y0)
            x1 = max(pv.x1, pi.x1)
            y1 = max(pv.y1, pi.y1)
            from matplotlib.transforms import Bbox
            self._pos_union = Bbox.from_extents(x0, y0, x1, y1).frozen()
            self._captured_positions = True
        except Exception:
            pass

    def _apply_focus_layout(self):
        self._capture_default_positions_if_needed()
        if not self._captured_positions:
            return

        if self._focus_mode == "both":
            self.ax_v.set_visible(True)
            self.ax_i.set_visible(True)
            self.ax_v.set_position(self._pos_v_default)
            self.ax_i.set_position(self._pos_i_default)

            self.ax_v.set_xlabel("")
            self.ax_i.set_xlabel("t (s)")

        elif self._focus_mode == "v":
            self.ax_i.set_visible(False)
            self.ax_v.set_visible(True)
            self.ax_v.set_position(self._pos_union)
            self.ax_v.set_xlabel("t (s)")

        elif self._focus_mode == "i":
            self.ax_v.set_visible(False)
            self.ax_i.set_visible(True)
            self.ax_i.set_position(self._pos_union)
            self.ax_i.set_xlabel("t (s)")

        self.canvas.draw_idle()

    def focus_voltage(self):
        if self.t is None:
            return
        self._focus_mode = "v"
        self._apply_focus_layout()
        self.autoscale_y_visible(self.ax_v, self.lines_v)
        self.update_ylabels_from_visible()
        self.canvas.draw_idle()

    def focus_current(self):
        if self.t is None:
            return
        self._focus_mode = "i"
        self._apply_focus_layout()
        self.autoscale_y_visible(self.ax_i, self.lines_i)
        self.update_ylabels_from_visible()
        self.canvas.draw_idle()

    def restore_both(self):
        if self.t is None:
            return
        self._focus_mode = "both"
        self._apply_focus_layout()
        xmin, xmax = self.ax_v.get_xlim()
        self._set_xlim_visible(xmin, xmax)
        self.autoscale_y_visible(self.ax_v, self.lines_v)
        self.autoscale_y_visible(self.ax_i, self.lines_i)
        self.update_ylabels_from_visible()
        self.canvas.draw_idle()

    def on_focus_mode_changed(self, idx: int):
        if self.t is None:
            return

        if idx == 0:
            self._focus_mode = "both"
            self._apply_focus_layout()
            # mantém X e ajusta Y de ambos
            xmin, xmax = self.ax_v.get_xlim()
            self._set_xlim_visible(xmin, xmax)
            if self.ax_v.get_visible():
                self.autoscale_y_visible(self.ax_v, self.lines_v)
            if self.ax_i.get_visible():
                self.autoscale_y_visible(self.ax_i, self.lines_i)

        elif idx == 1:
            self._focus_mode = "v"
            self._apply_focus_layout()
            if self.ax_v.get_visible():
                self.autoscale_y_visible(self.ax_v, self.lines_v)

        elif idx == 2:
            self._focus_mode = "i"
            self._apply_focus_layout()
            if self.ax_i.get_visible():
                self.autoscale_y_visible(self.ax_i, self.lines_i)

        self.update_ylabels_from_visible()
        self.canvas.draw_idle()

    # ---------- Sincronismo X ----------
    def _sync_x_from(self, src_ax):
        if self._syncing_xlim:
            return
        if self.t is None:
            return

        if self._focus_mode == "v" and src_ax is self.ax_i:
            return
        if self._focus_mode == "i" and src_ax is self.ax_v:
            return

        self._syncing_xlim = True
        try:
            xmin, xmax = src_ax.get_xlim()
            self._set_xlim_visible(xmin, xmax)

            if self.ax_v.get_visible():
                self.autoscale_y_visible(self.ax_v, self.lines_v)
            if self.ax_i.get_visible():
                self.autoscale_y_visible(self.ax_i, self.lines_i)

            self.update_ylabels_from_visible()
            self.canvas.draw_idle()
        finally:
            self._syncing_xlim = False

    # ---------- IO ----------
    def open_files(self):
        cfg_path, _ = QFileDialog.getOpenFileName(self, "Selecione o arquivo .cfg", "", "CFG (*.cfg);;All (*.*)")
        if not cfg_path:
            return

        data_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione o arquivo .dat/.bdat",
            str(Path(cfg_path).parent),
            "DAT/BDAT (*.dat *.bdat);;All (*.*)"
        )
        if not data_path:
            return

        try:
            self.load_comtrade(Path(cfg_path), Path(data_path))
        except Exception as e:
            QMessageBox.critical(self, "Erro ao carregar COMTRADE", str(e))

    def load_comtrade(self, cfg_path: Path, data_path: Path):
        self.cfg = parse_cfg(cfg_path)

        if self.cfg.file_type == "ASCII":
            t, analog_raw = read_dat_ascii(data_path, self.cfg)
            layout = "ASCII"
        else:
            t, analog_raw = read_bdat_binary(data_path, self.cfg)
            layout = "BINARY"

        self.t = t
        self.analog_raw = analog_raw

        self.t0 = float(self.t[0]) if len(self.t) else 0.0
        self.t1 = float(self.t[-1]) if len(self.t) else 0.0

        self.voltage_names, self.current_names, _ = classify_va(self.cfg)

        self.info.setText(
            f"{cfg_path.name} | {layout} | na={self.cfg.na} fs={self.cfg.fs}Hz "
            f"| t=[{self.t0:.6f}..{self.t1:.6f}]s | V={len(self.voltage_names)} I={len(self.current_names)}"
        )

        self.sp_tmin.setEnabled(True)
        self.sp_tmax.setEnabled(True)
        self.btn_apply.setEnabled(True)
        self.btn_reset_time.setEnabled(True)
        self.cb_pri_sec.setEnabled(True)

        self.sp_tmin.setValue(self.t0)
        self.sp_tmax.setValue(self.t1)

        self.box_v.set_channels(self.voltage_names)
        self.box_i.set_channels(self.current_names)

        self.plot_stack.setCurrentWidget(self.page_plot)

        self.recompute_scaled()
        self.build_plots()

        self.cb_focus_mode.setEnabled(True)
        self.cb_focus_mode.blockSignals(True)
        self.cb_focus_mode.setCurrentIndex(0)  # "Todos"
        self.cb_focus_mode.blockSignals(False)


        self.btn_export_as.setEnabled(True)

        self._focus_mode = "both"
        self._captured_positions = False
        self.on_home_clicked()

    # ---------- Escala (PRI/SEC) ----------
    def recompute_scaled(self):
        if self.cfg is None or self.analog_raw is None:
            return
        self.analog_scaled = scale_analogs(self.cfg, self.analog_raw, apply_pri_sec=bool(self.cb_pri_sec.isChecked()))

    # ---------- Plot ----------
    def build_plots(self):
        if self.cfg is None or self.t is None:
            return

        self.ax_v.clear()
        self.ax_i.clear()
        self.lines_v.clear()
        self.lines_i.clear()

        self.ax_v.set_title("Canais Analógicos de Tensão")
        self.ax_i.set_title("Canais Analógicos de Corrente")
        self.ax_i.set_xlabel("t (s)")

        self._colors.clear()

        # plota tensão
        for nm in self.voltage_names:
            y = self.analog_scaled.get(nm)
            if y is None:
                continue
            ln, = self.ax_v.plot(self.t, y, label=nm)
            self.lines_v[nm] = ln
            self._colors[nm] = ln.get_color()

        # plota corrente
        for nm in self.current_names:
            y = self.analog_scaled.get(nm)
            if y is None:
                continue
            ln, = self.ax_i.plot(self.t, y, label=nm)
            self.lines_i[nm] = ln
            self._colors[nm] = ln.get_color()

        self.ax_v.grid(True)
        self.ax_i.grid(True)

        self.box_v.set_color_map(self._colors)
        self.box_i.set_color_map(self._colors)

        self._captured_positions = False
        self._apply_focus_layout()

        self._set_xlim_visible(self.t0, self.t1)
        self.on_toggles_changed()

        self.canvas.draw_idle()

    def on_toggles_changed(self):
        # tensão
        for nm, ln in self.lines_v.items():
            show = self.box_v.is_visible(nm)
            ln.set_visible(show)

        # corrente
        for nm, ln in self.lines_i.items():
            show = self.box_i.is_visible(nm)
            ln.set_visible(show)

        # autoscale só do visível (por eixo)
        if self.ax_v.get_visible():
            self.autoscale_y_visible(self.ax_v, self.lines_v)
        if self.ax_i.get_visible():
            self.autoscale_y_visible(self.ax_i, self.lines_i)

        self.update_ylabels_from_visible()
        self.canvas.draw_idle()

    def autoscale_y_visible(self, ax, lines: Dict[str, object]):
        if self.t is None:
            return
        xmin, xmax = ax.get_xlim()
        xs = self.t

        i0 = int(np.searchsorted(xs, xmin, side="left"))
        i1 = int(np.searchsorted(xs, xmax, side="right"))
        i0 = max(0, min(i0, len(xs)-1)) if len(xs) else 0
        i1 = max(i0+1, min(i1, len(xs))) if len(xs) else 0

        ys = []
        for ln in lines.values():
            if not ln.get_visible():
                continue
            y = ln.get_ydata()
            if y is None or len(y) != len(xs):
                continue
            seg = y[i0:i1]
            if seg.size:
                seg = seg[np.isfinite(seg)]
                if seg.size:
                    ys.append(seg)

        if not ys:
            return

        ycat = np.concatenate(ys)
        y_min = float(np.min(ycat))
        y_max = float(np.max(ycat))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return
        if abs(y_max - y_min) < 1e-18:
            pad = 1.0
        else:
            pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)

    def update_ylabels_from_visible(self):
        if self.cfg is None:
            return

        v_vis = [n for n in self.voltage_names if self.box_v.is_visible(n)]
        i_vis = [n for n in self.current_names if self.box_i.is_visible(n)]

        v_units = units_for_names(self.cfg, v_vis)
        i_units = units_for_names(self.cfg, i_vis)

        if self.ax_v.get_visible():
            self.ax_v.set_ylabel(format_ylabel("Tensão", v_units))
        if self.ax_i.get_visible():
            self.ax_i.set_ylabel(format_ylabel("Corrente", i_units))

    def _set_xlim_visible(self, xmin: float, xmax: float):
        # aplica em ambos, mesmo se oculto (para manter consistência do sharex)
        try:
            self.ax_v.set_xlim(xmin, xmax)
        except Exception:
            pass
        try:
            self.ax_i.set_xlim(xmin, xmax)
        except Exception:
            pass

    # ---------- Home custom (chamado pela toolbar e pelo app) ----------
    def on_home_clicked(self, from_toolbar: bool = False):
        if self.t is None:
            return

        self.sp_tmin.blockSignals(True)
        self.sp_tmax.blockSignals(True)
        self.sp_tmin.setValue(self.t0)
        self.sp_tmax.setValue(self.t1)
        self.sp_tmin.blockSignals(False)
        self.sp_tmax.blockSignals(False)

        # 1) reseta xlim em ambos (mesmo se invisível)
        self._set_xlim_visible(self.t0, self.t1)

        # 2) reaplica o layout do foco/restaura
        self._apply_focus_layout()

        # 3) autoscale somente do que estiver visível agora
        if self.ax_v.get_visible():
            self.autoscale_y_visible(self.ax_v, self.lines_v)
        if self.ax_i.get_visible():
            self.autoscale_y_visible(self.ax_i, self.lines_i)

        self.update_ylabels_from_visible()
        self.canvas.draw_idle()

    # ---------- PRI/SEC --------
    def on_pri_sec_changed(self, _):
        if self.cfg is None:
            return
        self.recompute_scaled()
        # atualiza dados das linhas sem recriar tudo
        for nm, ln in self.lines_v.items():
            y = self.analog_scaled.get(nm)
            if y is not None:
                ln.set_ydata(y)
        for nm, ln in self.lines_i.items():
            y = self.analog_scaled.get(nm)
            if y is not None:
                ln.set_ydata(y)

        # respeita o xlim atual e recalcula y
        if self.ax_v.get_visible():
            self.autoscale_y_visible(self.ax_v, self.lines_v)
        if self.ax_i.get_visible():
            self.autoscale_y_visible(self.ax_i, self.lines_i)

        self.update_ylabels_from_visible()
        self.canvas.draw_idle()

    # ---------- Time window ----
    def apply_time_window(self):
        if self.t is None:
            return
        tmin, tmax = self._current_time_window()
        self._set_xlim_visible(tmin, tmax)

        if self.ax_v.get_visible():
            self.autoscale_y_visible(self.ax_v, self.lines_v)
        if self.ax_i.get_visible():
            self.autoscale_y_visible(self.ax_i, self.lines_i)

        self.update_ylabels_from_visible()
        self.canvas.draw_idle()

    def reset_time_range(self):
        if self.t is None:
            return
        self.sp_tmin.blockSignals(True)
        self.sp_tmax.blockSignals(True)
        self.sp_tmin.setValue(self.t0)
        self.sp_tmax.setValue(self.t1)
        self.sp_tmin.blockSignals(False)
        self.sp_tmax.blockSignals(False)
        self.on_home_clicked()

    # ---------- Cursor ----------
    def clear_cursor(self):
        if self.vline_v is not None:
            try:
                self.vline_v.remove()
            except Exception:
                pass
            self.vline_v = None
        if self.vline_i is not None:
            try:
                self.vline_i.remove()
            except Exception:
                pass
            self.vline_i = None
        self.canvas.draw_idle()

    def on_mouse_move(self, event):
        if self.t is None:
            return
        if event.inaxes not in (self.ax_v, self.ax_i):
            return

        if self._cursor_syncing:
            return
        self._cursor_syncing = True
        try:
            x = event.xdata
            if x is None:
                return

            if self.vline_v is None and self.ax_v.get_visible():
                self.vline_v = self.ax_v.axvline(x, linestyle="--", linewidth=1)
            if self.vline_i is None and self.ax_i.get_visible():
                self.vline_i = self.ax_i.axvline(x, linestyle="--", linewidth=1)

            if self.vline_v is not None and self.ax_v.get_visible():
                self.vline_v.set_xdata([x, x])
            if self.vline_i is not None and self.ax_i.get_visible():
                self.vline_i.set_xdata([x, x])

            self.canvas.draw_idle()
        finally:
            self._cursor_syncing = False

    # ---------- Export UI ----------
    def export_as_dialog(self):
        if self.cfg is None or self.t is None or self.analog_raw is None:
            QMessageBox.information(self, "Exportar", "Carregue um COMTRADE antes de exportar.")
            return

        dlg = ExportDialog(self, self)
        dlg.exec()

    # ---------- Export (core) ----------
    def _build_export_cfg_and_raw(self, names: List[str], t_sel: np.ndarray, raw_sel: np.ndarray) -> Cfg:
        if self.cfg is None:
            raise ValueError("CFG original não carregado.")

        name_to_col = {a.name: i for i, a in enumerate(self.cfg.analogs)}
        kept_adefs: List[AnaDef] = []
        for nm in names:
            if nm in name_to_col:
                kept_adefs.append(self.cfg.analogs[name_to_col[nm]])

        apply_ps = bool(self.cb_pri_sec.isChecked())

        new_analogs: List[AnaDef] = []
        for new_idx, a in enumerate(kept_adefs, start=1):
            k = 1.0
            if apply_ps:
                ps = (a.pri_sec or "S").upper()
                pri = float(a.pri if a.pri is not None else 1.0)
                sec = float(a.sec if a.sec is not None else 1.0)
                if abs(sec) < 1e-18:
                    sec = 1.0
                if ps == "S":
                    k = pri / sec

            a_eff = float(a.a) * k
            b_eff = float(a.b) * k

            new_analogs.append(AnaDef(
                idx=new_idx,
                name=a.name,
                phase=a.phase,
                units=a.units,
                a=a_eff,
                b=b_eff,
                skew=a.skew,
                min_level=a.min_level,
                max_level=a.max_level,
                pri=1.0,
                sec=1.0,
                pri_sec="P",
            ))

        fs = float(self.cfg.fs) if self.cfg.fs and self.cfg.fs > 0 else 0.0
        if fs <= 0.0 and t_sel.size > 1:
            d = np.diff(t_sel)
            d = d[np.isfinite(d) & (d > 0)]
            if d.size:
                fs = float(1.0 / np.median(d))
        if fs <= 0.0:
            fs = 1.0

        return Cfg(
            station=self.cfg.station,
            revision=self.cfg.revision,
            version=self.cfg.version,
            total=len(new_analogs),
            na=len(new_analogs),
            nd=0,
            freq=float(self.cfg.freq if self.cfg.freq else 60.0),
            fs=fs,
            samples=int(t_sel.size),
            start_dt=self.cfg.start_dt,
            end_dt=self.cfg.end_dt,
            file_type="ASCII",
            time_mult=float(self.cfg.time_mult if self.cfg.time_mult else 1.0),
            analogs=new_analogs,
            digitals=[],
        )

    # ---------- Export ASCII (.cfg+.dat) ----------
    def export_selected(self):
        if self.cfg is None or self.t is None or self.analog_raw is None:
            return

        names = self.selected_names()
        if not names:
            QMessageBox.information(self, "Exportar", "Nenhum canal selecionado.")
            return

        tmin, tmax = self._current_time_window()
        i0 = int(np.searchsorted(self.t, tmin, side="left"))
        i1 = int(np.searchsorted(self.t, tmax, side="right"))
        t_sel = self.t[i0:i1]
        raw_sel = self.analog_raw[i0:i1, :]

        name_to_col = {a.name: i for i, a in enumerate(self.cfg.analogs)}
        cols = [name_to_col[n] for n in names if n in name_to_col]
        if not cols:
            QMessageBox.information(self, "Exportar", "Seleção inválida.")
            return

        raw_sel = raw_sel[:, cols].astype(np.int32)

        out_dir = self._choose_and_make_export_folder(prefix="export_ascii")
        if out_dir is None:
            return

        out_cfg = out_dir / "export.cfg"
        out_dat = out_dir / "export.dat"

        new_cfg = self._build_export_cfg_and_raw(names, t_sel, raw_sel)
        try:
            write_cfg(new_cfg, out_cfg)
            write_dat_ascii(out_dat, t_sel, raw_sel, new_cfg.time_mult)
        except Exception as e:
            QMessageBox.critical(self, "Exportar", f"Falha ao exportar:\n{e}")
            return

        QMessageBox.information(
            self,
            "Exportar",
            f"Exportação concluída!\n\n"
            f"Pasta criada:\n{out_dir}\n\n"
            f"Arquivos gerados:\n" +
            "\n".join(p.name for p in out_dir.iterdir())
        )

    # ---------- Export BINARY (.cfg+.bdat) ----------
    def export_selected_binary(self, ts_mode: str = "i"):
        if self.cfg is None or self.t is None or self.analog_raw is None:
            return

        names = self.selected_names()
        if not names:
            QMessageBox.information(self, "Exportar", "Nenhum canal selecionado.")
            return

        tmin, tmax = self._current_time_window()
        i0 = int(np.searchsorted(self.t, tmin, side="left"))
        i1 = int(np.searchsorted(self.t, tmax, side="right"))
        t_sel = self.t[i0:i1]
        raw_sel = self.analog_raw[i0:i1, :]

        name_to_col = {a.name: i for i, a in enumerate(self.cfg.analogs)}
        cols = [name_to_col[n] for n in names if n in name_to_col]
        if not cols:
            QMessageBox.information(self, "Exportar", "Seleção inválida.")
            return

        raw_sel = raw_sel[:, cols].astype(np.int32)

        out_dir = self._choose_and_make_export_folder(prefix="export_binary")
        if out_dir is None:
            return

        out_cfg = out_dir / "export.cfg"
        out_bdat = out_dir / "export.bdat"

        new_cfg = self._build_export_cfg_and_raw(names, t_sel, raw_sel)
        new_cfg.file_type = "BINARY"
        try:
            write_cfg(new_cfg, out_cfg)
            write_bdat_binary(out_bdat, t_sel, raw_sel, new_cfg, ts_mode=ts_mode)
        except Exception as e:
            QMessageBox.critical(self, "Exportar", f"Falha ao exportar:\n{e}")
            return

        QMessageBox.information(
            self,
            "Exportar",
            f"Exportação concluída!\n\n"
            f"Pasta criada:\n{out_dir}\n\n"
            f"Arquivos gerados:\n" +
            "\n".join(p.name for p in out_dir.iterdir())
        )

    # ---------- Export JSON ----------
    def export_json(self):
        if self.cfg is None or self.t is None or self.analog_raw is None:
            return

        names = self.selected_names()
        if not names:
            QMessageBox.information(self, "Exportar JSON", "Nenhum canal selecionado.")
            return

        tmin, tmax = self._current_time_window()
        i0 = int(np.searchsorted(self.t, tmin, side="left"))
        i1 = int(np.searchsorted(self.t, tmax, side="right"))
        t_sel = self.t[i0:i1]

        # dados escalonados (com PRI/SEC aplicado conforme checkbox)
        scaled = {n: self.analog_scaled.get(n)[i0:i1].tolist() for n in names if n in self.analog_scaled}

        out_dir = self._choose_and_make_export_folder(prefix="export_json")
        if out_dir is None:
            return

        out_json = out_dir / "export.json"

        channels = []
        for a in self.cfg.analogs:
            if a.name not in names:
                continue
            channels.append({
                "name": a.name,
                "phase": a.phase,
                "units": a.units,
                "data": scaled.get(a.name, [])
            })

        payload = {
            "meta": {
                "station": self.cfg.station,
                "revision": self.cfg.revision,
                "version": self.cfg.version,
                "apply_pri_sec": bool(self.cb_pri_sec.isChecked()),
                "freq": float(self.cfg.freq) if self.cfg.freq else None,
                "fs": float(self.cfg.fs) if self.cfg.fs else None,
                "time_mult": float(self.cfg.time_mult) if self.cfg.time_mult else 1.0,
                "tmin": float(tmin),
                "tmax": float(tmax),
                "samples": int(t_sel.size),
            },
            "time": {"unit": "s", "t": t_sel.tolist()},
            "channels": channels
        }

        try:
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Exportar JSON", f"Falha ao exportar JSON:\n{e}")
            return

        QMessageBox.information(
            self,
            "Exportar",
            f"Exportação concluída!\n\n"
            f"Pasta criada:\n{out_dir}\n\n"
            f"Arquivos gerados:\n" +
            "\n".join(p.name for p in out_dir.iterdir())
        )

    def _choose_and_make_export_folder(self, prefix: str = "COMTRADE_export") -> Optional[Path]:
        base_dir = QFileDialog.getExistingDirectory(self, "Escolha onde criar a pasta de exportação")
        if not base_dir:
            return None

        base = Path(base_dir)

        # subpasta automática (timestamp evita conflito)
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base / f"{prefix}_{stamp}"

        try:
            out_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # muito raro, mas garante fallback
            out_dir = base / f"{prefix}_{stamp}_1"
            out_dir.mkdir(parents=True, exist_ok=True)

        return out_dir


class ExportDialog(QDialog):
    def __init__(self, parent, viewer: ComtradeViewer):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Exportar como...")

        lay = QVBoxLayout(self)

        self.list = QListWidget()
        for txt in [
            "COMTRADE ASCII (.cfg + .dat)",
            "COMTRADE BINARY 32-bit ts (.cfg + .bdat)",
            "COMTRADE BINARY 64-bit ts (.cfg + .bdat)",
            "JSON (tempo + canais selecionados)"
        ]:
            it = QListWidgetItem(txt)
            self.list.addItem(it)
        lay.addWidget(self.list)

        btns = QHBoxLayout()
        self.btn_ok = QPushButton("Exportar")
        self.btn_cancel = QPushButton("Cancelar")
        btns.addStretch(1)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._do_export)

    def _do_export(self):
        row = self.list.currentRow()
        if row < 0:
            QMessageBox.information(self, "Exportar", "Selecione um formato.")
            return
        if row == 0:
            self.viewer.export_selected()
        elif row == 1:
            self.viewer.export_selected_binary(ts_mode="i")
        elif row == 2:
            self.viewer.export_selected_binary(ts_mode="q")
        elif row == 3:
            self.viewer.export_json()
        self.accept()