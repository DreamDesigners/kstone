from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent.parent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


labels = ["Normal", "Stone"]
base_dir = os.path.join(BASE_DIR, 'static/data/')

print(base_dir)