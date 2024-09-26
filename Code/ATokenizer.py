import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

# Possible sound things could be:
# Frequency (in Hz, determines wavelength)
# Intensite (db/power, also describable as the amplitude)
# Sample rate (resolution of audio)