from algorithm import Encoding

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

fingerprint = Encoding()
hashes = fingerprint.process(fs, s)

fingerprint.display_spectrogram()
#fingerprint.display_spectrogram(fs ,s)

#plt.scatter(fingerprint.anchors[:, 0], fingerprint.anchors[:, 1], s = 5)
plt.show()
