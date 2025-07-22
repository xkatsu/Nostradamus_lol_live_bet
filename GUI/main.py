import pandas as pd
import numpy as np
import random
import os
import joblib

from gensim.models import Word2Vec

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QHBoxLayout,
    QFormLayout, QLineEdit, QTextEdit, QGridLayout, QMessageBox, QFileDialog, QFrame, QSizePolicy, QTableWidget, QTableWidgetItem
)
from PySide6.QtGui import QFont, QPalette, QColor, QIcon
from PySide6.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

random.seed(77)
np.random.seed(77)

CHECKPOINTS = {
    "draft": [],
    "10min": ['golddiffat10', 'void_grubs', 'opp_void_grubs', 'dragons', 'opp_dragons'],
    "15min": ['golddiffat10', 'golddiffat15', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons'],
    "20min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'void_grubs', 'opp_void_grubs'],
    "25min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'golddiffat25', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'atakhans', 'opp_atakhans'],
}

class LoLPredictorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔮 Nostradamus v1.0.0")
        self.setWindowIcon(QIcon("icons/lol_icon.png"))
        self.setGeometry(150, 100, 1250, 900)
        self.setStyleSheet("background-color: #121212; color: #ffffff;")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        title = QLabel("🔮 Nostradamus v1.0.0 - League Live Betting AI")
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00ffcc; margin: 10px;")
        layout.addWidget(title)

        grid = QGridLayout()
        font_input = QFont("Segoe UI", 10)
        field_style = "background-color: #222; color: #eee; padding: 8px; border-radius: 6px; border: 1px solid #444;"

        self.team_blue_input = QLineEdit("GIANTX PRIDE")
        self.team_blue_input.setFont(font_input)
        self.team_blue_input.setStyleSheet(field_style)
        blue_label = QLabel("🔷 Time Azul:")
        blue_label.setStyleSheet("color: #1e90ff; font-weight: bold;")
        grid.addWidget(blue_label, 0, 0)
        grid.addWidget(self.team_blue_input, 0, 1)

        self.team_red_input = QLineEdit("Los Heretics")
        self.team_red_input.setFont(font_input)
        self.team_red_input.setStyleSheet(field_style)
        red_label = QLabel("🔺 Time Vermelho:")
        red_label.setStyleSheet("color: #ff5555; font-weight: bold;")
        grid.addWidget(red_label, 1, 0)
        grid.addWidget(self.team_red_input, 1, 1)

        layout.addLayout(grid)

        self.predict_button = QPushButton("📊 Gerar Previsões")
        self.predict_button.setStyleSheet(
            "QPushButton { background-color: #00aaff; color: white; font-size: 14px; padding: 8px 16px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #33ccff; }"
        )
        self.predict_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.predict_button, alignment=Qt.AlignCenter)

        result_container = QVBoxLayout()
        result_frame = QFrame()
        result_frame.setStyleSheet("background-color: #1e1e2f; border: 1px solid #2c2c3c; border-radius: 8px; padding: 8px;")
        result_frame.setLayout(result_container)

        self.result_display = QTextEdit()
        self.result_display.setFont(QFont("Courier New", 11))
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("background-color: #1b1b1b; border: none; padding: 8px; color: #b8faff;")
        result_container.addWidget(self.result_display)

        layout.addWidget(result_frame)

        self.graph_canvas = FigureCanvas(plt.figure(figsize=(6, 3)))
        self.graph_canvas.setStyleSheet("background-color: #1c1c1c; border: 1px solid #2a2a2a; border-radius: 8px;")
        layout.addWidget(self.graph_canvas)

        self.odds_table = QTableWidget(0, 4)
        self.odds_table.setHorizontalHeaderLabels(["Checkpoint", "🔷 Azul", "🔺 Vermelho", "Odds (Azul / Verm.)"])
        self.odds_table.setStyleSheet("background-color: #1d1d2b; color: #ffffff; gridline-color: #2c2c3c;")
        self.odds_table.setFont(QFont("Segoe UI", 9))
        self.odds_table.horizontalHeader().setStyleSheet("color: #00e6e6; font-weight: bold;")
        self.odds_table.verticalHeader().setVisible(False)
        self.odds_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.odds_table)

        self.predict_button.clicked.connect(self.generate_predictions)
        self.setLayout(layout)

    def generate_predictions(self):
        self.result_display.setText("⏳ Carregando previsões... Aguarde...")
        self.result_display.repaint()
        try:
            import RF as model_script
            result_text, odds_data, x_vals, y_vals = model_script.run_prediction_interface(
                self.team_blue_input.text(),
                self.team_red_input.text()
            )
            self.result_display.setText(result_text)

            self.odds_table.setRowCount(len(odds_data))
            for i, (cp, pb, pr, (ob, or_)) in enumerate(odds_data):
                self.odds_table.setItem(i, 0, QTableWidgetItem(cp))
                self.odds_table.setItem(i, 1, QTableWidgetItem(f"{pb:.1%}"))
                self.odds_table.setItem(i, 2, QTableWidgetItem(f"{pr:.1%}"))
                self.odds_table.setItem(i, 3, QTableWidgetItem(f"{ob:.2f} / {or_:.2f}"))

            self.graph_canvas.figure.clear()
            ax = self.graph_canvas.figure.add_subplot(111)
            ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='#00ccff')
            ax.set_title("Prob. de Vitória do Time Azul")
            ax.set_xlabel("Checkpoint")
            ax.set_ylabel("Prob. Azul")
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.3)
            self.graph_canvas.draw()

        except Exception as e:
            self.result_display.setText(f"❌ Erro ao gerar previsões:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication([])
    window = LoLPredictorUI()
    window.show()
    app.exec()
