from algorithmV1 import Encoding

import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.signal import spectrogram
import scipy.signal
#rom skimage.feature import peak_local_max

fs, s = read("samples\Frisk - Au.Ra.wav")
size = 128
noverlap = 32
# nperseg si la fenêtre n'a pas directement la bonne taille
window = scipy.signal.get_window("boxcar", size, fftbins=True)

fingerprint = Encoding(window, size)
hashes = fingerprint.process(fs, s)

print(hashes)
# fingerprint.display_spectrogram(fs ,s)

# plt.scatter(peak[:, 0], peak[:, 1], s = 5)
# plt.show()
