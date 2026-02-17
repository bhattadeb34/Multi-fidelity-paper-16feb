
# ===================== CONFIGURATION =====================
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'figures')

os.makedirs(OUT_DIR, exist_ok=True)

NIST_NEU_PATH = os.path.join(DATA_DIR, 'FINAL_PM7_ALL_neutral_cleaned.csv')
NIST_PROT_PATH = os.path.join(DATA_DIR, 'FINAL_PM7_ALL_protonated_cleaned.csv')
NIST_EXP_PATH = os.path.join(DATA_DIR, 'Dataset.csv')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Plotting specific configuration
FIGURES_DPI = 300
FONT_SIZE = 10
AXES_LABEL_SIZE = 11
AXES_TITLE_SIZE = 12
LEGEND_FONT_SIZE = 9
XTICK_LABEL_SIZE = 9
YTICK_LABEL_SIZE = 9

import matplotlib
try:
    if os.environ.get('COLAB_GPU') or os.environ.get('TRAMPOLINE_CI'):
         matplotlib.use('Agg')
except:
    pass
