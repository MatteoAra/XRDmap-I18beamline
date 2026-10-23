import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt 
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QSplitter, QDoubleSpinBox, QComboBox)
from PyQt5.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.patches import (Rectangle, Ellipse)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)


# ==========================================================
# SETTINGS
# ==========================================================

# This only limits the number of curves DISPLAYED.
# ALL curves are retained and saved.
MAX_DISPLAY_CURVES = 300


# ==========================================================
# DATA EXPLORER
# ==========================================================

class H5ThetaExplorer:

    def __init__(self, filename, x_path, y_path, intensity_map_path, theta_path, theta_intensity_path,):
        self.filename = filename
        self.x = None
        self.y = None
        self.intensity_map = None
        self.theta = None
        self.theta_intensity = None
        self._load_data(x_path, y_path, intensity_map_path, theta_path, theta_intensity_path,)

        # --------------------------------------------------
        # Last extraction
        # --------------------------------------------------
        self.last_theta = None
        self.last_all_curves = None
        self.last_avg_intensity = None

        # Spatial coordinates of last extracted profiles
        self.last_x_coords = None
        self.last_y_coords = None

    # ------------------------------------------------------

    def _load_data(self, x_path, y_path, intensity_map_path, theta_path, theta_intensity_path,):
        with h5py.File(self.filename, "r") as f:
            self.x = np.asarray( f[x_path][:] )
            self.y = np.asarray( f[y_path][:] )
            self.intensity_map = np.asarray( f[intensity_map_path][:] )
            self.theta = np.asarray( f[theta_path][:] )
            self.theta_intensity = np.asarray( f[theta_intensity_path][:] )

        # --------------------------------------------------
        # Basic consistency checks
        # --------------------------------------------------

        if self.x.ndim != 1:
            raise ValueError("X coordinate data must be 1-dimensional." )
        if self.y.ndim != 1:
            raise ValueError("Y coordinate data must be 1-dimensional.")

        if self.intensity_map.ndim != 2:
            raise ValueError("Map intensity data must be 2-dimensional.")
        if self.theta.ndim != 1:
            raise ValueError("Theta data must be 1-dimensional.")
        if self.theta_intensity.ndim != 3:
            raise ValueError(
                "XRD intensity data must be 3-dimensional "
                "(Y, X, theta)."
            )
        if (self.intensity_map.shape != self.theta_intensity.shape[:2]):
            raise ValueError(
                "The map and XRD spatial dimensions "
                "do not match."
            )
        if (self.theta_intensity.shape[2] != len(self.theta)):
            raise ValueError(
                "The theta dimension does not match "
                "the XRD intensity data."
            )
        if len(self.x) != self.intensity_map.shape[1]:
            raise ValueError(
                "X coordinate dimension does not match "
                "the map."
            )
        if len(self.y) != self.intensity_map.shape[0]:
            raise ValueError(
                "Y coordinate dimension does not match "
                "the map."
            )

    # ------------------------------------------------------

    def save_last_xrd(self, filename):
        if self.last_theta is None:
            raise ValueError("No XRD data available.")

        curves = self.last_all_curves
        data = np.column_stack([self.last_theta, self.last_avg_intensity, curves.T,])
        header_lines = ["2theta average_intensity individual_profiles"]

        # --------------------------------------------------
        # Coordinates of every individual profile
        # --------------------------------------------------

        if (self.last_x_coords is not None and self.last_y_coords is not None):
            header_lines.append("")
            header_lines.append("Individual profile coordinates:")

            for i, (x, y) in enumerate(zip(self.last_x_coords, self.last_y_coords,)):
                header_lines.append(
                    f"profile_{i}: "
                    f"X={x:.10g}, "
                    f"Y={y:.10g}"
                )

        np.savetxt(filename, data, header="\n".join(header_lines), )

# ==========================================================
# GUI
# ==========================================================

class H5ThetaExplorerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("I18 Map-XRD Datafile Explorer")
        self.resize(1200,1000,)
        self.filename = None
        self.explorer = None

        # ==================================================
        # Matplotlib callbacks
        # ==================================================

        self.line_cid = None
        self.pixel_cid = None
        self.iso_cid = None

        # ==================================================
        # Rectangle callbacks
        # ==================================================

        self.rect_press_cid = None
        self.rect_motion_cid = None
        self.rect_release_cid = None
        self.rect_start = None
        self.rect_patch = None

        # ==================================================
        # Ellipse callbacks
        # ==================================================
        
        self.ellipse_start = None
        self.ellipse_patch = None
        self.ellipse_press_cid = None
        self.ellipse_motion_cid = None
        self.ellipse_release_cid = None

        # ==================================================
        # Line ROI
        # ==================================================

        self.line_coords = []
        # Actual pixel indices selected by line ROI
        self.line_pixel_indices = []
        # Graphics for line ROI
        self.line_artist = None
        self.line_pixel_artist = None
        self.line_endpoint_artists = []

        # ==================================================
        # Map graphics
        # ==================================================

        self.map_image = None
        self.pixel_marker = None
        self.iso_overlay = None
        self.phase_match_artist = None
        self.map_colorbar = None

        # ==================================================
        # Original full-map limits
        # ==================================================

        self.original_xlim = None
        self.original_ylim = None

        # ==================================================
        # Selected pixel
        # ==================================================

        self.selected_pixel = None
        self._init_ui()

    # ======================================================
    # USER INTERFACE
    # ======================================================

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        grid = QGridLayout()

        # ==================================================
        # FILE
        # ==================================================

        self.open_button = QPushButton("Select HDF5 File")
        self.open_button.clicked.connect(self.open_file)
        grid.addWidget(self.open_button, 0, 0, )
        self.file_label = QLabel("No file selected")
        grid.addWidget(self.file_label, 0, 1, 1, 5,)

        # ==================================================
        # DATA PATHS
        # ==================================================

        self.entries = {}
        path_rows = [
            (1, [("X path", "processed/It/t1x_value_set"),
                 ("Y path", "processed/It/t1y_value_set")]),
            (2, [("Map Intensity path", "processed/It/data")]),
            (3, [("XRD 2Theta path", "processed/integrated/2th_deg"),
                 ("XRD Intensity path", "processed/integrated/data")]),
        ]
        for row, fields in path_rows:
            col = 0
            for label, default in fields:
                lbl = QLabel(label)
                entry = QLineEdit(default)
                entry.setMinimumWidth(100)
                grid.addWidget(lbl, row, col)
                grid.addWidget(entry, row, col + 1)
                self.entries[label] = entry
                col += 2

        # ==================================================
        # ANALYSIS BUTTONS
        # ==================================================


        button_row = 4
        button_row1 = QHBoxLayout()
        button_row1.setSpacing(20)
        
        self.load_btn = QPushButton("Load + Plot Map")
        self.load_btn.clicked.connect(self.load_and_plot)
        self.load_btn.setFixedWidth(170)
        button_row1.addWidget(self.load_btn)
        button_row1.addStretch()

        self.rect_btn = QPushButton("Rectangle ROI")
        self.rect_btn.clicked.connect(self._activate_rectangle)
        self.rect_btn.setEnabled(False)
        self.rect_btn.setFixedWidth(170)
        button_row1.addWidget(self.rect_btn)

        self.line_btn = QPushButton("Line ROI")
        self.line_btn.clicked.connect(self._activate_line)
        self.line_btn.setEnabled(False)
        self.line_btn.setFixedWidth(170)
        button_row1.addWidget(self.line_btn)

        self.ellipse_btn = QPushButton("Ellipse ROI")
        self.ellipse_btn.clicked.connect(self._activate_ellipse)
        self.ellipse_btn.setEnabled(False)
        self.ellipse_btn.setFixedWidth(170)
        button_row1.addWidget(self.ellipse_btn)

        self.pixel_btn = QPushButton("Select Pixel")
        self.pixel_btn.clicked.connect(self._activate_pixel_selection)
        self.pixel_btn.setEnabled(False)
        self.pixel_btn.setFixedWidth(170)
        button_row1.addWidget(self.pixel_btn)

        button_row1.addStretch()
        grid.addLayout(button_row1, button_row, 0, 1, 6)

        # ==================================================
        # ROW 2: TOLERANCE, ISOSURFACE, CLEAR AND SAVE
        # ==================================================

        button_row2 = QHBoxLayout()
        button_row2.setSpacing(20)
        
        #cmap_row = QHBoxLayout()
        #cmap_row.setSpacing(8)
        #cmap_row.addStretch()
        #cmap_row.addWidget(QLabel("Map colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(sorted(plt.colormaps()))
        self.cmap_combo.setCurrentText("viridis")
        self.cmap_combo.currentTextChanged.connect(self._change_colormap)
        self.cmap_combo.setFixedWidth(170)
        button_row2.addWidget(self.cmap_combo)
        #grid.addLayout(button_row2, button_row + 1, 0, 1, 6)
        
        
        button_row2.addStretch()
        button_row2.addStretch()

        # Intensity tolerance label
        tolerance_label = QLabel("Intensity tolerance:")
        button_row2.addWidget(tolerance_label)

        # Intensity tolerance widget
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.01, 100.0)
        self.tolerance_spin.setDecimals(1)
        self.tolerance_spin.setSingleStep(1.0)
        self.tolerance_spin.setValue(1.0)
        self.tolerance_spin.setSuffix(" %")
        self.tolerance_spin.setFixedWidth(100)
        button_row2.addWidget(self.tolerance_spin)

        # Isosurface ±Tolerance
        self.iso_btn = QPushButton("Isosurface ±Tolerance")
        self.iso_btn.clicked.connect(self._activate_isosurface_selection)
        self.iso_btn.setEnabled(False)
        self.iso_btn.setFixedWidth(170)
        button_row2.addWidget(self.iso_btn)
        
        # Clear Selection
        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.clicked.connect(self._clear_selection)
        self.clear_btn.setEnabled(False)
        self.clear_btn.setFixedWidth(170)
        button_row2.addWidget(self.clear_btn)

        # Save Extracted XRD
        self.save_btn = QPushButton("Save Extracted XRD")
        self.save_btn.clicked.connect(self.save_xrd)
        self.save_btn.setEnabled(False)
        self.save_btn.setFixedWidth(170)
        button_row2.addWidget(self.save_btn)

        button_row2.addStretch()
        grid.addLayout(button_row2, button_row + 1, 0, 1, 6)


        button_row3 = QHBoxLayout()
        button_row3.setSpacing(20)
        for index in range(10): 
            button_row3.addStretch(); 

        # XRD matching method
        button_row3.addWidget(QLabel("Method:"))
        self.phase_method_combo = QComboBox()
        self.phase_method_combo.addItems(["Pearson Correlation", "Peak-position Jaccard",])
        self.phase_method_combo.setFixedWidth(170)
        button_row3.addWidget(self.phase_method_combo)

        # Minimum score threshold, used by both matching methods
        button_row3.addWidget(QLabel("Minimum score:"))
        self.profile_similarity_spin = QDoubleSpinBox()
        self.profile_similarity_spin.setRange(0.50, 1.00)
        self.profile_similarity_spin.setDecimals(2)
        self.profile_similarity_spin.setSingleStep(0.05)
        self.profile_similarity_spin.setValue(0.90)
        self.profile_similarity_spin.setFixedWidth(70)
        self.profile_similarity_spin.setToolTip("Minimum Pearson correlation or Jaccard similarity score (0–1).")
        button_row3.addWidget(self.profile_similarity_spin)

        # Angular tolerance used only by peak-position Jaccard matching
        button_row3.addWidget(QLabel("Peak tolerance:"))
        self.peak_tolerance_spin = QDoubleSpinBox()
        self.peak_tolerance_spin.setRange(0.01, 2.00)
        self.peak_tolerance_spin.setDecimals(2)
        self.peak_tolerance_spin.setSingleStep(0.05)
        self.peak_tolerance_spin.setValue(0.10)
        self.peak_tolerance_spin.setSuffix("°")
        self.peak_tolerance_spin.setFixedWidth(70)
        self.peak_tolerance_spin.setToolTip("Maximum 2θ separation for two peaks to count as a match.")
        button_row3.addWidget(self.peak_tolerance_spin)

        # Find XRD Phase button
        self.find_phase_btn = QPushButton("Find XRD Phase")
        self.find_phase_btn.clicked.connect(self._find_xrd_phase)
        self.find_phase_btn.setEnabled(False)
        self.find_phase_btn.setFixedWidth(170)
        button_row3.addWidget(self.find_phase_btn)

        button_row3.addStretch()

        grid.addLayout(button_row3, button_row + 2, 0, 1, 6)



        # ==================================================
        # STATUS
        # ==================================================

        self.status_label = QLabel("Load an HDF5 file to begin.")
        grid.addWidget(self.status_label, button_row + 4, 0, 1, 6,)
        main_layout.addLayout(grid)

        # ==================================================
        # SPLITTER
        # ==================================================

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # ==================================================
        # MAP
        # ==================================================

        self.map_fig = Figure()
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_ax = self.map_fig.add_subplot(111)
        map_container = QWidget()
        map_layout = QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0,)
        self.map_toolbar = NavigationToolbar(self.map_canvas, self,)
        map_layout.addWidget(self.map_toolbar)
        map_layout.addWidget(self.map_canvas)
        splitter.addWidget(map_container)

        # ==================================================
        # XRD
        # ==================================================

        self.xrd_fig = Figure()
        self.xrd_canvas = FigureCanvas(self.xrd_fig)
        self.xrd_ax = self.xrd_fig.add_subplot(111)
        splitter.addWidget(self.xrd_canvas)
        splitter.setSizes([550,350])

    # ======================================================
    # FILE
    # ======================================================

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open HDF5 File", "", "HDF5 Files (*.h5 *.nxs *.hdf5)",)
        if filename:
            self.filename = filename
            self.file_label.setText(filename)

    # ======================================================
    # LOAD MAP
    # ======================================================

    def load_and_plot(self):
        if not self.filename:
            QMessageBox.warning(self, "No file", "Please select an HDF5 file first.",)
            return
        try:
            self.explorer = H5ThetaExplorer(
                self.filename,
                self.entries["X path"].text(),
                self.entries["Y path"].text(),
                self.entries["Map Intensity path"].text(),
                self.entries["XRD 2Theta path"].text(),
                self.entries["XRD Intensity path"].text(),
            )

        except Exception as exc:
            QMessageBox.critical(self, "Loading error", str(exc),)
            return

        # --------------------------------------------------
        # Reset selection state
        # --------------------------------------------------

        self._deactivate_selection_tools()
        self._remove_selection_graphics()
        self._remove_rectangle_patch()
        self._remove_ellipse_patch()
        self._remove_line_graphics()
        self.selected_pixel = None

        # --------------------------------------------------
        # Clear previous map
        # --------------------------------------------------

        if self.map_colorbar is not None:
            self.map_colorbar.remove()
            self.map_colorbar = None
        self.map_ax.clear()

        # --------------------------------------------------
        # Plot map
        # --------------------------------------------------

        self.map_image = self.map_ax.pcolormesh(
            self.explorer.x,
            self.explorer.y,
            self.explorer.intensity_map,
            shading="auto",
            cmap="viridis",
        )

        self.map_ax.invert_yaxis()

        self.map_ax.set_xlabel("X")
        self.map_ax.set_ylabel("Y")
        self.map_ax.set_title("Intensity Map")

        self.map_colorbar = self.map_fig.colorbar(self.map_image, ax=self.map_ax,)
        self.map_fig.tight_layout()
        self.map_canvas.draw()

        # --------------------------------------------------
        # Save full-map limits
        # --------------------------------------------------

        self.original_xlim = (self.map_ax.get_xlim())
        self.original_ylim = (self.map_ax.get_ylim())

        # --------------------------------------------------
        # Enable analysis
        # --------------------------------------------------

        self.rect_btn.setEnabled(True)
        self.ellipse_btn.setEnabled(True)
        self.line_btn.setEnabled(True)
        self.pixel_btn.setEnabled(True)
        self.iso_btn.setEnabled(True)
        self.find_phase_btn.setEnabled(self.explorer.last_avg_intensity is not None)
        self.clear_btn.setEnabled(True)

        self.status_label.setText(
            "Map loaded. Use the Matplotlib toolbar "
            "to zoom/pan, then select an analysis mode."
        )

    # ======================================================
    # TOOLBAR STATE
    # ======================================================

    def _toolbar_is_active(self):
        return getattr( self.map_toolbar, "_active", None, )

    # ------------------------------------------------------

    def _deactivate_toolbar(self):
        try:
            active = getattr(self.map_toolbar, "_active", None,)
            if active == "PAN":
                self.map_toolbar.pan()
            elif active == "ZOOM":
                self.map_toolbar.zoom()
        except Exception:
            pass

    # ======================================================
    # DEACTIVATE SELECTION TOOLS
    # ======================================================

    def _deactivate_selection_tools(self):
        self._deactivate_rectangle()
        self._deactivate_ellipse()

        if self.line_cid is not None:
            self.map_canvas.mpl_disconnect(self.line_cid)
            self.line_cid = None
        self.line_coords = []

        if self.pixel_cid is not None:
            self.map_canvas.mpl_disconnect(self.pixel_cid)
            self.pixel_cid = None

        # --------------------------------------------------
        # Connected region
        # --------------------------------------------------

        if self.iso_cid is not None:
            self.map_canvas.mpl_disconnect(self.iso_cid)
            self.iso_cid = None


    # ======================================================
    # COLORMAP CHANGING UTILITY 
    # ======================================================
    
    def _change_colormap(self, cmap_name):
        if self.map_image is None:
            return

        self.map_image.set_cmap(cmap_name)

        # Update the existing colorbar to match the map
        if self.map_colorbar is not None:
            self.map_colorbar.update_normal(self.map_image)

        self.map_canvas.draw_idle()


    # ======================================================
    # RECTANGLE ROI
    # ======================================================

    def _activate_rectangle(self):
        if self.explorer is None:
            return
        self._deactivate_selection_tools()
        self._deactivate_toolbar()
        self._remove_rectangle_patch()
        self.rect_start = None
        self.rect_press_cid = (self.map_canvas.mpl_connect("button_press_event",self._rectangle_press,))
        self.rect_motion_cid = (self.map_canvas.mpl_connect("motion_notify_event",self._rectangle_motion,))
        self.rect_release_cid = (self.map_canvas.mpl_connect("button_release_event", self._rectangle_release,))
        self.status_label.setText(
            "Rectangle ROI active: "
            "click and drag over the desired region."
        )

    # ------------------------------------------------------

    def _deactivate_rectangle(self):
        if self.rect_press_cid is not None:
            self.map_canvas.mpl_disconnect(self.rect_press_cid)
            self.rect_press_cid = None

        if self.rect_motion_cid is not None:
            self.map_canvas.mpl_disconnect(self.rect_motion_cid)
            self.rect_motion_cid = None

        if self.rect_release_cid is not None:
            self.map_canvas.mpl_disconnect(self.rect_release_cid)
            self.rect_release_cid = None

        self.rect_start = None

    # ------------------------------------------------------

    def _rectangle_press(self, event):
        if event.button != 1:
            return

        if event.inaxes != self.map_ax:
            return

        if (event.xdata is None or event.ydata is None):
            return

        # IMPORTANT:
        # Remove the old patch BEFORE setting rect_start.
        #
        # The old implementation did this in the opposite
        # order, while _remove_rectangle_patch() reset
        # rect_start to None.
        self._remove_rectangle_patch()
        self.rect_start = (float(event.xdata), float(event.ydata),)

        self.rect_patch = Rectangle(
            xy=self.rect_start,
            width=0.0,
            height=0.0,
            facecolor="black",
            edgecolor="white",
            alpha=0.25,
            linewidth=2,
            zorder=100,
        )

        self.map_ax.add_patch(self.rect_patch)
        self.map_canvas.draw_idle()

    # ------------------------------------------------------

    def _rectangle_motion(self, event):
        if self.rect_start is None:
            return

        if event.inaxes != self.map_ax:
            return

        if (event.xdata is None or event.ydata is None):
            return

        x0, y0 = self.rect_start

        x1 = float(event.xdata)
        y1 = float(event.ydata)

        x_min = min(x0, x1)
        y_min = min(y0, y1)

        width = abs(x1 - x0)
        height = abs(y1 - y0)

        if self.rect_patch is not None:

            self.rect_patch.set_x(x_min)
            self.rect_patch.set_y(y_min)
            self.rect_patch.set_width(width)
            self.rect_patch.set_height(height)

        self.map_canvas.draw_idle()

    # ------------------------------------------------------

    def _rectangle_release(self, event):
        if self.rect_start is None:
            return
        if event.button != 1:
            return
        if (event.inaxes != self.map_ax or event.xdata is None or event.ydata is None):
            self._remove_rectangle_patch()
            self.map_canvas.draw_idle()
            self.status_label.setText(
                "Rectangle ROI cancelled: "
                "mouse released outside the map."
            )
            return

        x0, y0 = self.rect_start
        x1 = float(event.xdata)
        y1 = float(event.ydata)

        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        if self.rect_patch is not None:
            self.rect_patch.set_x(x_min)
            self.rect_patch.set_y(y_min)
            self.rect_patch.set_width(x_max - x_min)
            self.rect_patch.set_height(y_max - y_min)

        self.map_canvas.draw_idle()
        self._extract_rectangle(x_min,x_max,y_min,y_max,)
        self._deactivate_rectangle()

    # ------------------------------------------------------

    def _extract_rectangle(self, x_min, x_max, y_min, y_max,):
        ix = np.where((self.explorer.x >= x_min) & (self.explorer.x <= x_max))[0]
        iy = np.where((self.explorer.y >= y_min) & (self.explorer.y <= y_max))[0]

        if (len(ix) == 0 or len(iy) == 0):
            self.status_label.setText(
                "Rectangle contains no map pixels. "
                "Try a larger rectangle."
            )
            return

        curves = (self.explorer.theta_intensity[np.ix_(iy,ix,)].reshape(-1,len(self.explorer.theta),))

        xx, yy = np.meshgrid(self.explorer.x[ix], self.explorer.y[iy],)
        x_coords = xx.ravel()
        y_coords = yy.ravel()

        self._plot_xrd(
            curves,
            title=(
                f"Rectangle ROI "
                f"({len(curves)} pixels)"
            ),
            x_coords=x_coords,
            y_coords=y_coords,
        )

        self.status_label.setText(
            f"Rectangle ROI extracted: "
            f"{len(curves)} pixels | "
            f"X={x_min:.6g}–{x_max:.6g}, "
            f"Y={y_min:.6g}–{y_max:.6g}"
        )


    # ======================================================
    # ELLIPTICAL ROI
    # ======================================================
    def _activate_ellipse(self):
        if self.explorer is None:
            return

        self._deactivate_selection_tools()
        self._deactivate_toolbar()
        self._remove_ellipse_patch()
        self.ellipse_start = None
        self.ellipse_press_cid = self.map_canvas.mpl_connect("button_press_event", self._ellipse_press,)
        self.ellipse_motion_cid = self.map_canvas.mpl_connect("motion_notify_event", self._ellipse_motion,)
        self.ellipse_release_cid = self.map_canvas.mpl_connect("button_release_event", self._ellipse_release,)
        self.status_label.setText(
            "Ellipse ROI active: "
            "click and drag to define the ellipse.") 

    def _deactivate_ellipse(self):
        if self.ellipse_press_cid is not None:
            self.map_canvas.mpl_disconnect(self.ellipse_press_cid)
            self.ellipse_press_cid = None

        if self.ellipse_motion_cid is not None:
            self.map_canvas.mpl_disconnect(self.ellipse_motion_cid)
            self.ellipse_motion_cid = None

        if self.ellipse_release_cid is not None:
            self.map_canvas.mpl_disconnect(self.ellipse_release_cid)
            self.ellipse_release_cid = None
    
        self.ellipse_start = None
        
    def _ellipse_press(self, event):
        if event.button != 1:
            return
        if event.inaxes != self.map_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._remove_ellipse_patch()

        self.ellipse_start = (float(event.xdata), float(event.ydata),)
        self.ellipse_patch = Ellipse(xy=self.ellipse_start, width=0.0, height=0.0, facecolor="black", edgecolor="white", alpha=0.25, linewidth=2, zorder=100,)
        self.map_ax.add_patch(self.ellipse_patch)
        self.map_canvas.draw_idle()

    def _ellipse_motion(self, event):
        if self.ellipse_start is None:
            return
        if event.inaxes != self.map_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0 = self.ellipse_start
        x1 = float(event.xdata)
        y1 = float(event.ydata)
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        width = abs(x1 - x0)
        height = abs(y1 - y0)

        if self.ellipse_patch is not None:
            self.ellipse_patch.center = (center_x,center_y,)
            self.ellipse_patch.width = width
            self.ellipse_patch.height = height
        self.map_canvas.draw_idle()

    
    def _ellipse_release(self, event):
        if self.ellipse_start is None:
            return
        if event.button != 1:
            return
        if (event.inaxes != self.map_ax or event.xdata is None or event.ydata is None):
            self._remove_ellipse_patch()
            self.map_canvas.draw_idle()
            self.status_label.setText(
                "Ellipse ROI cancelled: "
                "mouse released outside the map."
            )
            self._deactivate_ellipse()
            return

        x0, y0 = self.ellipse_start
        x1 = float(event.xdata)
        y1 = float(event.ydata)
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        width = abs(x1 - x0)
        height = abs(y1 - y0)

        if self.ellipse_patch is not None:
            self.ellipse_patch.center = (center_x,center_y,)
            self.ellipse_patch.width = width
            self.ellipse_patch.height = height
        self.map_canvas.draw_idle()

        if width <= 0 or height <= 0:
            self.status_label.setText(
                "Ellipse ROI is too small. "
                "Try dragging over a larger region."
            )
            self._deactivate_ellipse()
            return

        self._extract_ellipse( center_x, center_y, width / 2.0, height / 2.0,)
        self._deactivate_ellipse()
    
    
    def _extract_ellipse(self, center_x, center_y, radius_x, radius_y,):
        # Get the map coordinates.
        x = np.asarray(self.explorer.x)
        y = np.asarray(self.explorer.y)
        # Find pixels inside the ellipse.
        xx, yy = np.meshgrid(x, y)

        ellipse_mask = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0)
        # Boolean indexing selects only pixels inside the ellipse.
        iy, ix = np.where(ellipse_mask)

        if len(ix) == 0:
            self.status_label.setText(
                "Ellipse contains no map pixels. "
                "Try a larger ellipse."
            )
            return

        # Extract the XRD curve for every selected pixel.
        #
        # Expected data shape:
        # theta_intensity.shape == (len(y), len(x), len(theta))
        curves = self.explorer.theta_intensity[ iy, ix, : ]

        # Corresponding spatial coordinates.
        x_coords = x[ix]
        y_coords = y[iy]

        # Plot the extracted XRD data using your existing method.
        self._plot_xrd(curves, title=f"Ellipse ROI ({len(curves)} pixels)", x_coords=x_coords, y_coords=y_coords, )
        self.status_label.setText(
            f"Ellipse ROI extracted: {len(curves)} pixels | "
            f"Center=({center_x:.6g}, {center_y:.6g}) | "
            f"Radii=({radius_x:.6g}, {radius_y:.6g})"
        )
    
    
    def _remove_ellipse_patch(self):
        if self.ellipse_patch is not None:
            try: self.ellipse_patch.remove()
            except ValueError:
                pass

            self.ellipse_patch = None

        if hasattr(self, "map_canvas"):
            self.map_canvas.draw_idle()
    

    # ======================================================
    # LINE ROI
    # ======================================================

    def _activate_line(self):

        if self.explorer is None:
            return
        self._deactivate_selection_tools()
        self._deactivate_toolbar()
        self._remove_line_graphics()
        self.line_coords = []
        self.status_label.setText(
            "Line ROI active: "
            "click the start and end points."
        )

        self.line_cid = (self.map_canvas.mpl_connect("button_press_event", self._line_click,))

    # ------------------------------------------------------

    def _line_click(self, event):
        if event.button != 1:
            return
        if event.inaxes != self.map_ax:
            return
        if (event.xdata is None or event.ydata is None):
            return

        self.line_coords.append((float(event.xdata), float(event.ydata),))

        # --------------------------------------------------
        # First point
        # --------------------------------------------------

        if len(self.line_coords) == 1:

            self.line_endpoint_artists.append(
                self.map_ax.plot(
                    event.xdata,
                    event.ydata,
                    "ro",
                    markersize=6,
                    zorder=110,
                )[0]
            )

            self.map_canvas.draw_idle()

            self.status_label.setText(
                "Line ROI: start point selected. "
                "Click the end point."
            )
            return

        # --------------------------------------------------
        # Second point
        # --------------------------------------------------

        if len(self.line_coords) == 2:
            self.map_canvas.mpl_disconnect(self.line_cid)
            self.line_cid = None
            self._extract_line()

    # ======================================================
    # LINE -> ACTUAL MAP PIXELS
    # ======================================================

    def _coordinate_to_index(self, coordinate, axis_values,):
        """
        Convert a physical X/Y coordinate to a floating-point
        array index.

        Example:

            axis = [10, 20, 30]

            coordinate = 15

        returns approximately:

            0.5

        This is only used to determine which EXISTING map
        pixels are traversed by the line. No XRD intensity
        interpolation is performed.
        """

        axis_values = np.asarray(axis_values, dtype=float,)

        if axis_values.size < 2:
            return 0.0

        # np.interp requires ascending coordinates.
        if axis_values[0] > axis_values[-1]:
            reversed_axis = axis_values[::-1]

            index_reversed = np.interp(
                coordinate,
                reversed_axis,
                np.arange(
                    len(axis_values),
                    dtype=float,
                )[::-1],
            )

            return float(index_reversed)

        return float(
            np.interp(
                coordinate,
                axis_values,
                np.arange(
                    len(axis_values),
                    dtype=float,
                ),
            )
        )

    # ------------------------------------------------------

    @staticmethod
    def _bresenham_pixels(x0, y0,x1, y1,):
        """
        Return integer grid pixels along a line.
        This is a standard discrete grid traversal.
        The returned coordinates are ARRAY INDICES, not physical X/Y coordinates.
        """

        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))

        pixels = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        x = x0
        y = y0

        while True:
            pixels.append((y, x))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return pixels

    # ------------------------------------------------------

    def _get_line_pixels(self, x0, y0, x1, y1,):
        """
        Determine the actual map pixels traversed by the
        requested physical-coordinate line.

        Returns: y_indices, x_indices
        These are integer indices into: theta_intensity[y_index, x_index, :]
        """
        x_index0 = self._coordinate_to_index(x0, self.explorer.x,)
        y_index0 = self._coordinate_to_index(y0, self.explorer.y,)
        x_index1 = self._coordinate_to_index(x1, self.explorer.x,)
        y_index1 = self._coordinate_to_index(y1, self.explorer.y,)
        pixel_pairs = self._bresenham_pixels(x_index0, y_index0, x_index1, y_index1,)
        ny, nx = (self.explorer.intensity_map.shape)
        valid_pairs = []
        for iy, ix in pixel_pairs:
            if (0 <= iy < ny and 0 <= ix < nx):
                valid_pairs.append((iy, ix))

        if not valid_pairs:
            return (np.asarray([], dtype=int), np.asarray([], dtype=int),)

        y_indices = np.asarray([p[0] for p in valid_pairs], dtype=int,)
        x_indices = np.asarray([p[1] for p in valid_pairs], dtype=int,)
        return (y_indices, x_indices,)

    # ------------------------------------------------------

    def _extract_line(self):
        if len(self.line_coords) != 2:
            return
        (x0, y0), (x1, y1) = (self.line_coords)

        # --------------------------------------------------
        # Draw physical-coordinate line
        # --------------------------------------------------

        self.line_artist, = self.map_ax.plot(
            [x0, x1],
            [y0, y1],
            "r-",
            lw=2,
            zorder=105,
        )

        # --------------------------------------------------
        # Find ACTUAL pixels traversed by line
        # --------------------------------------------------

        y_indices, x_indices = (self._get_line_pixels(x0,y0,x1,y1,))
        n_pixels = len(x_indices)

        if n_pixels == 0:
            self.status_label.setText("Line does not intersect any map pixels.")
            self.map_canvas.draw_idle()
            return

        self.line_pixel_indices = list(zip(y_indices,x_indices,))

        # --------------------------------------------------
        # Physical coordinates of actual pixels
        # --------------------------------------------------

        xs = self.explorer.x[x_indices]
        ys = self.explorer.y[y_indices]

        # --------------------------------------------------
        # Mark the ACTUAL pixels used
        # --------------------------------------------------

        self.line_pixel_artist = (
            self.map_ax.plot(
                xs,
                ys,
                "yo",
                markersize=4,
                markeredgecolor="black",
                markeredgewidth=0.5,
                linestyle="None",
                zorder=106,
            )[0]
        )

        self.map_canvas.draw_idle()
        curves = (self.explorer.theta_intensity[y_indices,x_indices,:])
        curves = np.asarray(curves)
        self._plot_xrd(
            curves,
            title=(
                f"Line ROI XRD "
                f"({n_pixels} actual pixels)"
            ),
            x_coords=xs,
            y_coords=ys,
        )

        self.status_label.setText(
            f"Line ROI extracted: "
            f"{n_pixels} actual map pixels "
            f"(no interpolation)"
        )

    # ======================================================
    # SINGLE PIXEL
    # ======================================================

    def _activate_pixel_selection(self):
        if self.explorer is None:
            return
        self._deactivate_selection_tools()
        self._deactivate_toolbar()
        self.status_label.setText("Pixel selection active: click a pixel.")
        self.pixel_cid = (self.map_canvas.mpl_connect("button_press_event",self._pixel_callback,))

    # ------------------------------------------------------

    def _pixel_callback(self, event):
        if event.inaxes != self.map_ax:
            return
        if (event.xdata is None or event.ydata is None):
            return

        ix = np.argmin(np.abs(self.explorer.x - event.xdata))
        iy = np.argmin(np.abs(self.explorer.y - event.ydata))
        intensity = (self.explorer.intensity_map[iy,ix,])

        if not np.isfinite(intensity):
            self.status_label.setText(
                "Selected pixel contains "
                "NaN/invalid intensity."
            )
            return

        self._remove_selection_graphics()

        self.pixel_marker, = (
            self.map_ax.plot(
                self.explorer.x[ix],
                self.explorer.y[iy],
                marker="+",
                markersize=15,
                markeredgewidth=2.5,
                color="red",
                zorder=110,
            )
        )

        self.map_canvas.draw_idle()
        curve = (self.explorer.theta_intensity[iy,ix,:])
        self._plot_xrd(
            np.asarray([curve]),
            title=(
                f"Pixel XRD | "
                f"X={self.explorer.x[ix]:.6g}, "
                f"Y={self.explorer.y[iy]:.6g}"
            ),
            x_coords=np.asarray([self.explorer.x[ix]]),
            y_coords=np.asarray([self.explorer.y[iy]]),)

        self.selected_pixel = (iy,ix,)

        self.status_label.setText(
            f"Pixel selected: "
            f"X={self.explorer.x[ix]:.6g}, "
            f"Y={self.explorer.y[iy]:.6g}, "
            f"I={intensity:.6g}"
        )

        self.map_canvas.mpl_disconnect(
            self.pixel_cid
        )

        self.pixel_cid = None

    # ======================================================
    # CONNECTED ±TOLERANCE
    # ======================================================

    def _activate_isosurface_selection(self):
        if self.explorer is None:
            return
        self._deactivate_selection_tools()
        self._deactivate_toolbar()
        tolerance = (self.tolerance_spin.value())

        self.status_label.setText(
            f"Connected ±{tolerance:g}% active: "
            "click the seed pixel."
        )

        self.iso_cid = (self.map_canvas.mpl_connect("button_press_event",self._isosurface_callback,))

    # ------------------------------------------------------

    def _isosurface_callback(self, event):
        if event.inaxes != self.map_ax:
            return
        if (event.xdata is None or event.ydata is None):
            return

        ix0 = np.argmin(np.abs(self.explorer.x - event.xdata))
        iy0 = np.argmin(np.abs(self.explorer.y - event.ydata))
        intensity_map = (self.explorer.intensity_map)
        I0 = intensity_map[iy0,ix0]

        if not np.isfinite(I0):
            self.status_label.setText(
                "Seed pixel contains "
                "invalid/NaN intensity."
            )
            return

        tolerance_percent = (self.tolerance_spin.value())
        tolerance_fraction = (tolerance_percent / 100.0)
        delta = (abs(I0) * tolerance_fraction)

        lower = I0 - delta
        upper = I0 + delta
        finite = np.isfinite(intensity_map)

        candidate_mask = (
            finite
            &
            (
                intensity_map >= lower
            )
            &
            (
                intensity_map <= upper
            )
        )

        if not candidate_mask[iy0, ix0]:

            self.status_label.setText(
                "Seed pixel is not inside the "
                "calculated intensity range."
            )

            return

        # --------------------------------------------------
        # Connected component
        # --------------------------------------------------

        ny, nx = (candidate_mask.shape)
        connected = np.zeros_like(candidate_mask, dtype=bool,)
        queue = deque()
        queue.append((iy0, ix0,))
        connected[iy0, ix0] = True

        neighbours = [
            (-1, -1),
            (-1,  0),
            (-1,  1),
            ( 0, -1),
            ( 0,  1),
            ( 1, -1),
            ( 1,  0),
            ( 1,  1),
        ]

        while queue:
            iy, ix = queue.popleft()
            for dy, dx in neighbours:
                jy = iy + dy
                jx = ix + dx

                if (
                    jy < 0
                    or jy >= ny
                    or jx < 0
                    or jx >= nx
                ):
                    continue

                if connected[jy, jx ]:
                    continue

                if not candidate_mask[jy,jx]:
                    continue
                connected[jy, jx] = True
                queue.append(( jy, jx,))

        # --------------------------------------------------
        # Get selected pixels
        # --------------------------------------------------

        selected_indices = np.argwhere(connected)
        n_pixels = len(selected_indices)

        if n_pixels == 0:
            self.status_label.setText("No connected pixels found.")
            return

        y_indices = (selected_indices[:, 0])
        x_indices = (selected_indices[:, 1])

        x_coords = (self.explorer.x[x_indices])
        y_coords = (self.explorer.y[y_indices])
        self._remove_selection_graphics()

        # --------------------------------------------------
        # Draw connected region
        # --------------------------------------------------

        overlay = np.where(connected, 1.0, np.nan,)

        self.iso_overlay = (
            self.map_ax.pcolormesh(
                self.explorer.x,
                self.explorer.y,
                overlay,
                shading="auto",
                cmap="autumn",
                alpha=0.45,
                zorder=90,
            )
        )

        # --------------------------------------------------
        # Seed marker
        # --------------------------------------------------

        self.pixel_marker, = (
            self.map_ax.plot(
                self.explorer.x[ix0],
                self.explorer.y[iy0],
                marker="+",
                markersize=15,
                markeredgewidth=2.5,
                color="cyan",
                zorder=110,
            )
        )

        self.map_canvas.draw_idle()

        # --------------------------------------------------
        # Extract ALL XRD profiles
        # --------------------------------------------------

        curves = (self.explorer.theta_intensity[y_indices, x_indices, :])

        self._plot_xrd(
            curves,
            title=(
                f"Connected ±"
                f"{tolerance_percent:g}% "
                f"Region "
                f"({n_pixels} pixels)"
            ),
            x_coords=x_coords,
            y_coords=y_coords,
        )

        self.selected_pixel = (iy0, ix0,)

        self.status_label.setText(
            f"Connected region: {n_pixels} pixels | "
            f"Seed X={self.explorer.x[ix0]:.6g}, "
            f"Y={self.explorer.y[iy0]:.6g} | "
            f"I₀={I0:.6g} | "
            f"Allowed range: "
            f"{lower:.6g} – {upper:.6g}"
        )

        self.map_canvas.mpl_disconnect(self.iso_cid)
        self.iso_cid = None

    # ======================================================
    # RECTANGLE GRAPHICS
    # ======================================================

    def _remove_rectangle_patch(self):
        if self.rect_patch is not None:
            try:
                self.rect_patch.remove()
            except (ValueError, AttributeError,):
                pass

            self.rect_patch = None

        # IMPORTANT:
        #
        # Do NOT reset rect_start here.
        #
        # This function removes GRAPHICS only.
        # Selection state is managed separately.
        #
        # The old implementation reset rect_start here,
        # which caused Rectangle(None, ...) in _rectangle_press.

        if hasattr(self, "map_canvas",):
            self.map_canvas.draw_idle()

    # ======================================================
    # LINE GRAPHICS
    # ======================================================

    def _remove_line_graphics(self):
        if self.line_artist is not None:
            try:
                self.line_artist.remove()
            except (ValueError, AttributeError,):
                pass
            self.line_artist = None

        if self.line_pixel_artist is not None:
            try:
                self.line_pixel_artist.remove()
            except (ValueError, AttributeError,):
                pass
            self.line_pixel_artist = None

        for artist in self.line_endpoint_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError,):
                pass

        self.line_endpoint_artists = []
        self.line_pixel_indices = []
        if hasattr(self, "map_canvas", ):

            self.map_canvas.draw_idle()

    # ======================================================
    # GRAPHICS
    # ======================================================

    def _remove_selection_graphics(self):
        if self.pixel_marker is not None:
            try:
                self.pixel_marker.remove()
            except (ValueError, AttributeError,):
                pass
            self.pixel_marker = None

        if self.phase_match_artist is not None:
            try:
                self.phase_match_artist.remove()
            except (ValueError, AttributeError):
                pass
            self.phase_match_artist = None

        if self.iso_overlay is not None:
            try:
                self.iso_overlay.remove()
            except (ValueError, AttributeError,):
                pass

            self.iso_overlay = None

        if hasattr(self, "map_canvas",):
            self.map_canvas.draw_idle()

    # ======================================================
    # CLEAR SELECTION
    # ======================================================

    def _clear_selection(self):
        self._deactivate_selection_tools()
        self._remove_selection_graphics()
        self._remove_rectangle_patch()
        self._remove_ellipse_patch()
        self._remove_line_graphics()
        self.selected_pixel = None
        self.status_label.setText(
            "Selection cleared. "
            "Use the Matplotlib toolbar to navigate "
            "or select another analysis mode."
        )
        self.map_canvas.draw_idle()

    # ======================================================
    # XRD PLOT
    # ======================================================
    def _plot_xrd(self, curves, title="Extracted XRD", x_coords=None, y_coords=None,):
        curves = np.asarray(curves)
        if curves.ndim == 1:
            curves = curves[np.newaxis, :]
        if curves.shape[0] == 0:
            return
        self.xrd_ax.clear()

        # --------------------------------------------------
        # Determine vertical offset
        # --------------------------------------------------
        finite_values = curves[np.isfinite(curves)]
        if finite_values.size == 0:
            offset = 1.0
        else:
            max_value = np.nanmax(np.abs(finite_values))
            offset = (
                0.03 * max_value
                if max_value > 0
                else 1.0
            )

        # --------------------------------------------------
        # Display XRD profiles
        # --------------------------------------------------
        n_curves = curves.shape[0]

        if n_curves <= MAX_DISPLAY_CURVES:
            display_indices = np.arange(n_curves)
        else:
            display_indices = np.linspace(0, n_curves - 1, MAX_DISPLAY_CURVES, dtype=int,)

        for plot_number, i in enumerate(display_indices):
            curve = curves[i]
            self.xrd_ax.plot(self.explorer.theta, curve + plot_number * offset, lw=0.7, alpha=0.55, color="tab:blue",)

        # --------------------------------------------------
        # Average XRD profile
        # --------------------------------------------------

        average = np.nanmean(curves, axis=0,)
        self.xrd_ax.plot(self.explorer.theta, average, color="black", lw=2.5, label="Average",)
        self.xrd_ax.set_xlabel("2θ (deg)")
        self.xrd_ax.set_ylabel("Intensity")
        self.xrd_ax.set_title(title)
        self.xrd_ax.grid(True, alpha=0.3,)
        self.xrd_ax.legend()
        self.xrd_fig.tight_layout()
        self.xrd_canvas.draw_idle()

        # ==================================================
        # Save extraction
        # ==================================================
        self.explorer.last_theta = (self.explorer.theta)
        self.explorer.last_all_curves = (curves)
        self.explorer.last_avg_intensity = (average)

        self.explorer.last_x_coords = (
            None
            if x_coords is None
            else np.asarray(x_coords)
        )

        self.explorer.last_y_coords = (
            None
            if y_coords is None
            else np.asarray(y_coords)
        )

        self.save_btn.setEnabled(True)
        self.find_phase_btn.setEnabled(True)

    # ======================================================
    # XRD PHASE PROFILE MATCHING
    # ======================================================

    @staticmethod
    def _peak_positions(theta, intensity):
        """Return 2θ positions of simple local maxima in a 1D profile.

        A point is treated as a peak when it is higher than its left
        neighbour and at least as high as its right neighbour. Endpoints,
        non-finite samples, and maxima adjacent to non-finite samples are
        excluded. This intentionally avoids adding a SciPy dependency.
        """
        theta = np.asarray(theta, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        if theta.ndim != 1 or intensity.ndim != 1 or theta.size != intensity.size or theta.size < 3:
            return np.asarray([], dtype=float)
        valid = (
            np.isfinite(theta[1:-1])
            & np.isfinite(theta[:-2])
            & np.isfinite(theta[2:])
            & np.isfinite(intensity[1:-1])
            & np.isfinite(intensity[:-2])
            & np.isfinite(intensity[2:])
        )
        peak_mask = valid & (intensity[1:-1] > intensity[:-2]) & (intensity[1:-1] >= intensity[2:])
        return theta[1:-1][peak_mask]

    @staticmethod
    def _jaccard_peak_score(reference_peaks, profile_peaks, tolerance):
        """Fast one-to-one peak matching in O(n+m), returning Jaccard score."""
        reference_peaks = np.asarray(reference_peaks, dtype=float)
        profile_peaks = np.asarray(profile_peaks, dtype=float)
        n_ref = reference_peaks.size
        n_profile = profile_peaks.size
        if n_ref == 0 and n_profile == 0:
            return np.nan
        if n_ref == 0 or n_profile == 0:
            return 0.0

        # Peak positions are ordered along 2-theta. A two-pointer scan avoids
        # creating/sorting every possible pair, which can explode for noisy data.
        i = j = n_matched = 0
        while i < n_ref and j < n_profile:
            delta = reference_peaks[i] - profile_peaks[j]
            if abs(delta) <= tolerance:
                n_matched += 1
                i += 1
                j += 1
            elif delta < -tolerance:
                i += 1
            else:
                j += 1

        union = n_ref + n_profile - n_matched
        return float(n_matched / union) if union else np.nan

    def _find_xrd_phase(self):
        """Match map pixels to the current black average XRD reference profile."""
        if self.explorer is None or self.explorer.last_avg_intensity is None:
            QMessageBox.information(
                self, "Reference profile required",
                "First extract an ROI or select a pixel so the black average XRD profile is available.",
            )
            return

        reference = np.asarray(self.explorer.last_avg_intensity, dtype=float)
        profiles = np.asarray(self.explorer.theta_intensity, dtype=float)
        theta = np.asarray(self.explorer.theta, dtype=float)
        if profiles.ndim != 3 or profiles.shape[-1] != reference.size or theta.size != reference.size:
            QMessageBox.warning(self, "Profile mismatch", "The reference profile does not match the map XRD data.")
            return

        method = self.phase_method_combo.currentText()
        threshold = self.profile_similarity_spin.value()
        scores = np.full(profiles.shape[:2], np.nan, dtype=float)

        if method == "Pearson correlation":
            # Preserve the existing whole-profile Pearson-style shape correlation.
            ref_valid = np.isfinite(reference)
            if np.count_nonzero(ref_valid) < 3:
                QMessageBox.warning(self, "Invalid reference", "The average reference profile has too few valid points.")
                return
            ref = reference[ref_valid]
            ref = ref - np.mean(ref)
            ref_norm = np.linalg.norm(ref)
            if not np.isfinite(ref_norm) or ref_norm == 0:
                QMessageBox.warning(self, "Invalid reference", "The average reference profile is flat or invalid.")
                return
            ref = ref / ref_norm

            flat = profiles.reshape(-1, profiles.shape[-1])[:, ref_valid]
            finite_rows = np.all(np.isfinite(flat), axis=1)
            flat_scores = np.full(flat.shape[0], np.nan, dtype=float)
            if np.any(finite_rows):
                candidates = flat[finite_rows]
                candidates = candidates - np.mean(candidates, axis=1, keepdims=True)
                norms = np.linalg.norm(candidates, axis=1)
                valid_norms = norms > 0
                valid_rows = np.flatnonzero(finite_rows)[valid_norms]
                if valid_rows.size:
                    flat_scores[valid_rows] = (candidates[valid_norms] / norms[valid_norms, None]) @ ref
            scores = flat_scores.reshape(profiles.shape[:2])
            score_description = "Pearson correlation"

        else:
            reference_peaks = self._peak_positions(theta, reference)
            if reference_peaks.size == 0:
                QMessageBox.warning(
                    self, "No reference peaks",
                    "No local maxima could be detected in the average reference profile. "
                    "Check the profile or use Pearson correlation.",
                )
                return
            angular_tolerance = self.peak_tolerance_spin.value()
            for iy in range(profiles.shape[0]):
                for ix in range(profiles.shape[1]):
                    profile_peaks = self._peak_positions(theta, profiles[iy, ix, :])
                    scores[iy, ix] = self._jaccard_peak_score(
                        reference_peaks, profile_peaks, angular_tolerance
                    )
            score_description = "peak-position Jaccard similarity"

        matched = np.isfinite(scores) & (scores >= threshold)
        if self.phase_match_artist is not None:
            try:
                self.phase_match_artist.remove()
            except (ValueError, AttributeError):
                pass
            self.phase_match_artist = None

        iy, ix = np.where(matched)
        if ix.size:
            self.phase_match_artist = self.map_ax.scatter(
                self.explorer.x[ix], self.explorer.y[iy],
                s=28, facecolors="none", edgecolors="magenta", linewidths=1.2,
                label="XRD profile match", zorder=115,
            )
        self.map_canvas.draw_idle()

        if method == "Peak-position Jaccard":
            self.status_label.setText(
                f"XRD matching ({score_description}): {len(ix)} pixels with score ≥ {threshold:.2f}; "
                f"peak tolerance ±{self.peak_tolerance_spin.value():.2f}° 2θ. "
                "Magenta circles mark matches to the current black average profile."
            )
        else:
            self.status_label.setText(
                f"XRD matching ({score_description}): {len(ix)} pixels with score ≥ {threshold:.2f}. "
                "Magenta circles mark matches to the current black average profile."
            )

    # ======================================================
    # SAVE XRD
    # ======================================================

    def save_xrd(self):
        if (self.explorer is None or self.explorer.last_theta is None):
            QMessageBox.warning(self, "No XRD", "No extracted XRD data available.",)
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save XRD Data", "", "DAT files (*.dat)",)

        if not filename:
            return

        try:
            self.explorer.save_last_xrd(filename)
            QMessageBox.information(self,"Saved","XRD data saved successfully.",)

        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc),)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = H5ThetaExplorerGUI()
    window.show()
    sys.exit(app.exec_())
