#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

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
    QDoubleSpinBox, QSplitter, QGroupBox, QTabWidget, QStackedWidget, QFrame
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
        self.btn_all = QPushButton("Mostrar todos")
        self.btn_none = QPushButton("Ocultar todos")
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

        self.cb_pri_sec = QCheckBox("Aplicar PRI/SEC (P/S)")
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
            "Canais de Tensão (marque/desmarque)",
            "Nenhuma variável de tensão foi encontrada.",
            self.on_toggles_changed
        )
        self.box_i = ChannelBox(
            "Canais de Corrente (marque/desmarque)",
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

        # FIG: sem constrained_layout (evita “home” restaurar lims estranhos em alguns backends)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)

        # Ajuste manual para reduzir “área branca” e aumentar os plots
        self.fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.08, hspace=0.28)

        self.ax_v = self.fig.add_subplot(211)
        self.ax_i = self.fig.add_subplot(212, sharex=self.ax_v)

        self.toolbar = NavigationToolbar(self.canvas, self)
        page_plot_lay.addWidget(self.toolbar)
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

        # X sincronizado: sempre que qualquer eixo mudar xlim, forçamos ambos iguais
        self._syncing_xlim = False
        self.ax_v.callbacks.connect("xlim_changed", lambda ax: self._sync_x_from(self.ax_v))
        self.ax_i.callbacks.connect("xlim_changed", lambda ax: self._sync_x_from(self.ax_i))

        # Home personalizado: NÃO usa a “nav stack” do matplotlib (que estava restaurando ylims ruins)
        self.toolbar.home = self.on_home_clicked

    # ---------- Sincronismo X ----------
    def _sync_x_from(self, src_ax):
        if self._syncing_xlim:
            return
        if self.t is None:
            return
        self._syncing_xlim = True
        try:
            xmin, xmax = src_ax.get_xlim()
            # garante limites exatos, sem “margens” diferentes por eixo
            self.ax_v.set_xlim(xmin, xmax)
            self.ax_i.set_xlim(xmin, xmax)
            # após qualquer mudança em X, recomputa Y por eixo
            self.autoscale_y_visible(self.ax_v, self.lines_v)
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

        # default: range total (isso evita “desalinhamento” visual de início/fim)
        self.sp_tmin.setValue(self.t0)
        self.sp_tmax.setValue(self.t1)

        self.box_v.set_channels(self.voltage_names)
        self.box_i.set_channels(self.current_names)

        self.plot_stack.setCurrentWidget(self.page_plot)

        self.recompute_scaled()
        self.build_plots()
        self.on_home_clicked()  # home consistente ao carregar

    # ---------- Dados/plot ----------
    def recompute_scaled(self):
        if self.cfg is None or self.analog_raw is None:
            return
        self.analog_scaled = scale_analogs(self.cfg, self.analog_raw, apply_pri_sec=self.cb_pri_sec.isChecked())

    def build_plots(self):
        if self.t is None:
            return

        self.ax_v.clear()
        self.ax_i.clear()
        self.lines_v.clear()
        self.lines_i.clear()
        self._colors.clear()

        self.ax_v.set_title("Canais Analógicos de Tensão", pad=6)
        self.ax_i.set_title("Canais Analógicos de Corrente", pad=6)

        # Plot tensão
        for n in self.voltage_names:
            y = self.analog_scaled.get(n)
            if y is None:
                continue
            line, = self.ax_v.plot(self.t, y, label=n)
            self.lines_v[n] = line
            c = line.get_color()
            if isinstance(c, str):
                self._colors[n] = c

        # Plot corrente
        for n in self.current_names:
            y = self.analog_scaled.get(n)
            if y is None:
                continue
            line, = self.ax_i.plot(self.t, y, label=n)
            self.lines_i[n] = line
            c = line.get_color()
            if isinstance(c, str):
                self._colors[n] = c

        self.ax_i.set_xlabel("t (s)")
        self.ax_v.grid(True)
        self.ax_i.grid(True)

        self.vline_v = self.ax_v.axvline(x=np.nan, linestyle="--", linewidth=1)
        self.vline_i = self.ax_i.axvline(x=np.nan, linestyle="--", linewidth=1)

        self.box_v.set_color_map({n: self._colors.get(n, "#000000") for n in self.voltage_names})
        self.box_i.set_color_map({n: self._colors.get(n, "#000000") for n in self.current_names})

        self.on_toggles_changed()
        self.canvas.draw_idle()

    def on_toggles_changed(self):
        for n, line in self.lines_v.items():
            line.set_visible(self.box_v.is_visible(n))
        for n, line in self.lines_i.items():
            line.set_visible(self.box_i.is_visible(n))

        # Recalcula Y com base no X atual (sempre)
        self.autoscale_y_visible(self.ax_v, self.lines_v)
        self.autoscale_y_visible(self.ax_i, self.lines_i)
        self.update_ylabels_from_visible()

        self.canvas.draw_idle()

    def update_ylabels_from_visible(self):
        if self.cfg is None:
            return
        v_visible = [n for n, ln in self.lines_v.items() if ln.get_visible()]
        i_visible = [n for n, ln in self.lines_i.items() if ln.get_visible()]
        v_units = units_for_names(self.cfg, v_visible)
        i_units = units_for_names(self.cfg, i_visible)
        self.ax_v.set_ylabel(format_ylabel("Tensão", v_units))
        self.ax_i.set_ylabel(format_ylabel("Corrente", i_units))

    def autoscale_y_visible(self, ax, lines: Dict[str, object], pad_frac: float = 0.05):
        try:
            xmin, xmax = ax.get_xlim()
        except Exception:
            return

        y_min = None
        y_max = None

        for line in lines.values():
            if not line.get_visible():
                continue
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if x.size == 0 or y.size == 0:
                continue

            m = (x >= xmin) & (x <= xmax)
            if not np.any(m):
                continue

            yy = y[m]
            lo = float(np.nanmin(yy))
            hi = float(np.nanmax(yy))

            if y_min is None:
                y_min, y_max = lo, hi
            else:
                y_min = min(y_min, lo)
                y_max = max(y_max, hi)

        # se não há nenhuma linha visível nesse eixo, mantém e sai
        if y_min is None or y_max is None:
            return
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return

        if abs(y_max - y_min) < 1e-12:
            eps = 1.0 if abs(y_max) < 1e-6 else abs(y_max) * 0.05
            ax.set_ylim(y_min - eps, y_max + eps)
        else:
            pad = (y_max - y_min) * pad_frac
            ax.set_ylim(y_min - pad, y_max + pad)

    # ---------- Cursor ----------
    def on_mouse_move(self, event):
        if self._cursor_syncing:
            return
        if event.inaxes is None or event.xdata is None:
            return
        x = float(event.xdata)
        self._cursor_syncing = True
        try:
            if self.vline_v is not None:
                self.vline_v.set_xdata([x, x])
            if self.vline_i is not None:
                self.vline_i.set_xdata([x, x])
            self.canvas.draw_idle()
        finally:
            self._cursor_syncing = False

    def clear_cursor(self):
        if self.vline_v is not None:
            self.vline_v.set_xdata([np.nan, np.nan])
        if self.vline_i is not None:
            self.vline_i.set_xdata([np.nan, np.nan])
        self.canvas.draw_idle()

    # ---------- Janela de tempo ----------
    def _set_xlim_both(self, tmin: float, tmax: float):
        # força X idêntico nos dois eixos (resolve “desalinhamento”)
        self.ax_v.set_xlim(tmin, tmax)
        self.ax_i.set_xlim(tmin, tmax)

    def apply_time_window(self):
        if self.t is None:
            return

        tmin = float(self.sp_tmin.value())
        tmax = float(self.sp_tmax.value())
        if tmax < tmin:
            tmin, tmax = tmax, tmin
        if (tmax - tmin) <= 0:
            return

        tmin = max(tmin, self.t0)
        tmax = min(tmax, self.t1)

        self._set_xlim_both(tmin, tmax)
        self.autoscale_y_visible(self.ax_v, self.lines_v)
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
        self.apply_time_window()

    # ---------- HOME ----------
    def on_home_clicked(self):
        """
        Home consistente:
        - volta para o range TOTAL (t0..t1) (igual 'reset tempo' visual)
        - força xlim nos 2 eixos
        - recalcula Y para cada eixo (evita escala “desproporcional”)
        """
        if self.t is None:
            return

        # garante que os spinboxes representam o home esperado
        self.sp_tmin.blockSignals(True)
        self.sp_tmax.blockSignals(True)
        self.sp_tmin.setValue(self.t0)
        self.sp_tmax.setValue(self.t1)
        self.sp_tmin.blockSignals(False)
        self.sp_tmax.blockSignals(False)

        self._set_xlim_both(self.t0, self.t1)
        self.autoscale_y_visible(self.ax_v, self.lines_v)
        self.autoscale_y_visible(self.ax_i, self.lines_i)
        self.update_ylabels_from_visible()
        self.canvas.draw_idle()

    # ---------- PRI/SEC ----------
    def on_pri_sec_changed(self):
        if self.t is None or self.cfg is None or self.analog_raw is None:
            return

        self.recompute_scaled()

        for n, line in self.lines_v.items():
            y = self.analog_scaled.get(n)
            if y is not None:
                line.set_ydata(y)
        for n, line in self.lines_i.items():
            y = self.analog_scaled.get(n)
            if y is not None:
                line.set_ydata(y)

        # mantém o X atual, mas recalcula o Y corretamente
        xmin, xmax = self.ax_v.get_xlim()
        self._set_xlim_both(xmin, xmax)
        self.autoscale_y_visible(self.ax_v, self.lines_v)
        self.autoscale_y_visible(self.ax_i, self.lines_i)
        self.update_ylabels_from_visible()

        self.canvas.draw_idle()


def main():
    app = QApplication([])
    w = ComtradeViewer()
    w.resize(1600, 900)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
