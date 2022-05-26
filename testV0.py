from algorithmV1 import Encoding

import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.signal import spectrogram
import scipy.signal
#rom skimage.feature import peak_local_max

fs, s = read("samples\Cash Machine - Anno Domini Beats.wav")
size = 128
noverlap = 32
# nperseg si la fenêtre n'a pas directement la bonne taille
window = scipy.signal.get_window("boxcar", size, fftbins=True)

fingerprint = Encoding(window, size)
spectro, peak = fingerprint.process(fs, s)

fingerprint.display_spectrogram()

plt.scatter(peak[:, 0], peak[:, 1], s = 5)
plt.show()