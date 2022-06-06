from algorithm import Encoding, Matching

import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.signal import spectrogram
import scipy.signal
#rom skimage.feature import peak_local_max

fs, s = read("samples\Cash Machine - Anno Domini Beats.wav")
fs2,s2 = read("samples\Dark Alley Deals - Aaron Kenny.wav")
size = 128
noverlap = 32
# nperseg si la fenêtre n'a pas directement la bonne taille

fingerprint1 = Encoding()
fingerprint2 = Encoding()


hashes1 = fingerprint1.process(fs, s)
hashes2 = fingerprint2.process(fs2,s2)

# match = Matching(hashes1,hashes2)

# print(match.compare())

fingerprint1.display_spectrogram()
# fingerprint.display_spectrogram(fs ,s)

#plt.scatter(fingerprint.anchors[:, 0], fingerprint.anchors[:, 1], s = 5)
