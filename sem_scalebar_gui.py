"""
SEM Scalebar GUI
────────────────
Combines multi-folder selection with automatic scalebar rendering.
  • Remembers the last-used folder across runs
  • Processes only .tif files
  • Logs errors per file without stopping the batch
  • Double-click a queued file to render it individually
"""

import sys
import os
import re
import traceback
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QTextEdit, QSplitter, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# ── paths ─────────────────────────────────────────────────────────────────────
SETTINGS_FILE = Path.home() / ".sem_scalebar_last_dir.txt"


def load_last_dir() -> str:
    try:
        if SETTINGS_FILE.exists():
            return SETTINGS_FILE.read_text().strip()
    except Exception:
        pass
    return str(Path(__file__).parent)


def save_last_dir(path: str):
    try:
        SETTINGS_FILE.write_text(path)
    except Exception:
        pass


# ── SEM image class ───────────────────────────────────────────────────────────
class SEM_Image:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.image = Image.open(file_path)
        self.pixel_size, self.image_tilt = self._get_metadata()
        self.imgarray = np.asarray(self.image)
        self.shape = self.imgarray.shape

    def _get_metadata(self):
        try:
            raw = None
            try:
                raw = self.image.getexif()[34118]
            except (KeyError, TypeError):
                pass

            if raw is None:
                try:
                    raw = self.image.tag_v2[34118]
                except (KeyError, AttributeError):
                    pass

            if isinstance(raw, bytes):
                raw = raw.decode("latin-1", errors="ignore")

            if raw is None:
                raise ValueError("EXIF tag 34118 not found")

            match_pixel = re.search(r'Pixel Size =(\s+[\d.]+) (µm|nm)', raw)
            if not match_pixel:
                raise ValueError("Pixel Size not found in metadata")
            pixel_size_value = float(match_pixel.group(1))
            if match_pixel.group(2) == "µm":
                pixel_size_value *= 1000

            match_tilt = re.search(r'Stage at T =(\s+[\d.]+) °', raw)
            if not match_tilt:
                raise ValueError("Stage Tilt not found in metadata")

            return pixel_size_value, float(match_tilt.group(1))

        except Exception as exc:
            raise RuntimeError(f"Could not read metadata: {exc}")

    def scalebar_calc(self):
        ps = self.pixel_size
        if ps < 3:
            length = 200
        elif ps < 6:
            length = 500
        elif ps < 10:
            length = 1000
        elif ps < 50:
            length = 5000
        elif ps < 100:
            length = 10000
        else:
            length = 200000
        self.scalebar_length = length
        self.scalebar_pixels = length // ps
        return self.scalebar_pixels, length

    def plot_scalebar(self, ax):
        image_cropped = self.imgarray[:685, :]
        self.scalebar_pixels, self.scalebar_length = self.scalebar_calc()

        height, width = image_cropped.shape[:2]
        scalebar_center_x = 100
        scalebar_y = height - 80
        scalebar_start_x = int(scalebar_center_x - self.scalebar_pixels / 2)
        scalebar_end_x = scalebar_start_x + self.scalebar_pixels
        text_center_y = scalebar_y + 50

        ax.imshow(image_cropped, cmap="gray")
        ax.hlines(scalebar_y, scalebar_start_x, scalebar_end_x,
                  color="white", linewidth=3)
        ax.vlines([scalebar_start_x, scalebar_end_x],
                  ymin=scalebar_y - 20, ymax=scalebar_y + 20,
                  color="white", linewidth=2)

        label = (
            f"{self.scalebar_length * 0.001:.0f} µm"
            if self.scalebar_length >= 1000
            else f"{self.scalebar_length} nm"
        )
        ax.text(scalebar_center_x, text_center_y, label,
                color="white", ha="center", va="center",
                fontweight="bold", fontsize=16)
        ax.axis("off")


# ── shared save helper ────────────────────────────────────────────────────────
def save_scalebar(sem: SEM_Image, out_path: str):
    """Render scalebar and save PNG with no padding or white border."""
    image_cropped = sem.imgarray[:685, :]
    h, w = image_cropped.shape[:2]
    dpi = 150
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # axes fills entire figure canvas
    sem.plot_scalebar(ax)
    fig.savefig(out_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


# ── worker thread ─────────────────────────────────────────────────────────────
class ProcessWorker(QThread):
    progress = pyqtSignal(int)       # 0-100
    log      = pyqtSignal(str, str)  # message, level ("info"|"error"|"ok")
    finished = pyqtSignal(int, int)  # ok_count, error_count

    def __init__(self, file_list: List[str], output_dir: str):
        super().__init__()
        self.file_list  = file_list
        self.output_dir = output_dir

    def run(self):
        total     = len(self.file_list)
        ok_count  = 0
        err_count = 0

        for i, fpath in enumerate(self.file_list, 1):
            self.progress.emit(int(i / total * 100))
            fname = Path(fpath).name
            try:
                sem      = SEM_Image(fpath)
                out_name = Path(fpath).stem + "_scalebar.png"
                out_path = os.path.join(self.output_dir, out_name)
                save_scalebar(sem, out_path)
                self.log.emit(f"✓  {fname}  →  {out_name}", "ok")
                ok_count += 1
            except Exception:
                detail = traceback.format_exc().splitlines()[-1]
                self.log.emit(f"✗  {fname}  —  {detail}", "error")
                err_count += 1

        self.finished.emit(ok_count, err_count)


# ── main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM Scalebar Renderer")
        self.setMinimumSize(820, 580)
        self.file_list: List[str] = []
        self.output_dir: str = ""
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # title
        title = QLabel("SEM Scalebar Renderer")
        title.setFont(QFont("Courier New", 15, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        # folder + output buttons
        btn_row = QHBoxLayout()
        self.btn_select = QPushButton("📂  Select Source Folder(s)")
        self.btn_select.setFixedHeight(36)
        self.btn_select.clicked.connect(self._select_folders)

        self.btn_output = QPushButton("💾  Set Output Folder")
        self.btn_output.setFixedHeight(36)
        self.btn_output.clicked.connect(self._select_output)

        btn_row.addWidget(self.btn_select)
        btn_row.addWidget(self.btn_output)
        root.addLayout(btn_row)

        self.lbl_output = QLabel("Output: (not set)")
        self.lbl_output.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_output.setWordWrap(True)
        root.addWidget(self.lbl_output)

        # splitter: file list | log
        splitter = QSplitter(Qt.Horizontal)

        left = QFrame()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        lbl_files = QLabel("Queued .tif files  (double-click to render one)")
        lbl_files.setFont(QFont("Courier New", 10, QFont.Bold))
        self.file_list_widget = QListWidget()
        self.file_list_widget.setFont(QFont("Courier New", 9))
        self.file_list_widget.itemDoubleClicked.connect(self._render_single)
        self.lbl_count = QLabel("0 files found")
        self.lbl_count.setStyleSheet("color: #555; font-size: 10px;")
        left_lay.addWidget(lbl_files)
        left_lay.addWidget(self.file_list_widget)
        left_lay.addWidget(self.lbl_count)

        right = QFrame()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        lbl_log = QLabel("Processing log")
        lbl_log.setFont(QFont("Courier New", 10, QFont.Bold))
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Courier New", 9))
        right_lay.addWidget(lbl_log)
        right_lay.addWidget(self.log_widget)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([360, 420])
        root.addWidget(splitter, stretch=1)

        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

        # run button
        self.btn_run = QPushButton("▶  Render All Scalebars")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setEnabled(False)
        self.btn_run.setFont(QFont("Courier New", 11, QFont.Bold))
        self.btn_run.clicked.connect(self._run)
        root.addWidget(self.btn_run)

    # ── folder selection ──────────────────────────────────────────────────────
    def _select_folders(self):
        from PyQt5.QtWidgets import QListView, QTreeView

        last_dir = load_last_dir()
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Source Folder(s)")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setDirectory(last_dir)

        for view in dialog.findChildren((QListView, QTreeView)):
            view.setSelectionMode(view.ExtendedSelection)

        if not dialog.exec_():
            return

        folders = dialog.selectedFiles()
        if folders:
            save_last_dir(os.path.dirname(folders[0]))

        found: List[str] = []
        for folder in folders:
            for dirpath, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(".tif"):
                        found.append(os.path.join(dirpath, f))

        self.file_list = found
        self._refresh_file_list()

    def _refresh_file_list(self):
        self.file_list_widget.clear()
        for fp in self.file_list:
            item = QListWidgetItem(Path(fp).name)
            item.setToolTip(fp)   # full path retrieved on double-click
            self.file_list_widget.addItem(item)
        n = len(self.file_list)
        self.lbl_count.setText(f"{n} .tif file{'s' if n != 1 else ''} found")
        self._update_run_btn()

    # ── output folder ─────────────────────────────────────────────────────────
    def _select_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", load_last_dir(),
            QFileDialog.DontUseNativeDialog
        )
        if folder:
            self.output_dir = folder
            self.lbl_output.setText(f"Output: {folder}")
            self._update_run_btn()

    # ── single-file double-click ──────────────────────────────────────────────
    def _render_single(self, item: QListWidgetItem):
        if not self.output_dir:
            self._append_log("⚠  Set an output folder before rendering.", "error")
            return
        fpath = item.toolTip()
        fname = Path(fpath).name
        try:
            sem      = SEM_Image(fpath)
            out_name = Path(fpath).stem + "_scalebar.png"
            out_path = os.path.join(self.output_dir, out_name)
            save_scalebar(sem, out_path)
            self._append_log(f"✓  {fname}  →  {out_name}", "ok")
        except Exception:
            detail = traceback.format_exc().splitlines()[-1]
            self._append_log(f"✗  {fname}  —  {detail}", "error")

    # ── batch run ─────────────────────────────────────────────────────────────
    def _update_run_btn(self):
        self.btn_run.setEnabled(bool(self.file_list) and bool(self.output_dir))

    def _run(self):
        self.log_widget.clear()
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_output.setEnabled(False)

        self.worker = ProcessWorker(self.file_list, self.output_dir)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _append_log(self, msg: str, level: str):
        colors = {"ok": "#2ecc71", "error": "#e74c3c", "info": "#aaaaaa"}
        color  = colors.get(level, "#cccccc")
        self.log_widget.append(
            f'<span style="color:{color}; font-family:Courier New;">{msg}</span>'
        )

    def _on_finished(self, ok: int, errors: int):
        self._append_log(
            f"\n─── Done: {ok} succeeded, {errors} failed ───", "info"
        )
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_output.setEnabled(True)
        self.progress_bar.setValue(100)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
