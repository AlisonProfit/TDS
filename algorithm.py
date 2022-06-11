"""
Algorithm implementation
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from operator import itemgetter

from scipy.io.wavfile import read
from scipy.signal import spectrogram
from skimage.feature import peak_local_max
import scipy.signal

# ----------------------------------------------------------------------------
# Create a fingerprint for an audio file based on a set of hashes
# ----------------------------------------------------------------------------


class Encoding:

    """
    Class implementing the procedure for creating a fingerprint 
    for the audio files

    The fingerprint is created through the following steps
    - compute the spectrogram of the audio signal
    - extract local maxima of the spectrogram
    - create hashes using these maxima

    """

    def __init__(self, window = scipy.signal.get_window("boxcar", 128, fftbins=True), window_size = 128):

        """
        Class constructor

        To Do
        -----

        Initialize in the constructor all the parameters required for
        creating the signature of the audio files. These parameters include for
        instance:
        - the window selected for computing the spectrogram
        - the size of the temporal window 
        - the size of the overlap between subsequent windows
        - etc.

        All these parameters should be kept as attributes of the class.
        """

        # Insert code here
        self.window = window
        self.window_size = window_size


    def process(self, fs, s, deltaT = 1000, deltaF = 10000):

        """

        To Do
        -----

        This function takes as input a sampled signal s and the sampling
        frequency fs and returns the fingerprint (the hashcodes) of the signal.
        The fingerprint is created through the following steps
        - spectrogram computation
        - local maxima extraction
        - hashes creation

        Implement all these operations in this function. Keep as attributes of
        the class the spectrogram, the range of frequencies, the anchors, the 
        list of hashes, etc.

        Each hash can conveniently be represented by a Python dictionary 
        containing the time associated to its anchor (key: "t") and a numpy 
        array with the difference in time between the anchor and the target, 
        the frequency of the anchor and the frequency of the target 
        (key: "hash")


        Parameters
        ----------

        fs: int
           sampling frequency [Hz]
        s: numpy array
           sampled signal
        """

        self.fs = fs
        self.s = s

        # Insert code here
      #   spectro = spectrogram(s, fs, window = self.window, noverlap=32)
        spectro = spectrogram(s, fs, noverlap=32)
        f, t, Sxx = spectro
      #   Sxx = Sxx[f<20000, :]
      #   f = f[f<20000]
        peaks_coords = peak_local_max(Sxx, min_distance= 50, exclude_border=False)
      #   print(t.shape)
      #   print(peak.shape)
      #   print(peak)
        self.anchors = peaks_coords

        peaks_mask=np.zeros(Sxx.shape, dtype=np.int8)
        for item in peaks_coords:
           peaks_mask[item[0],item[1]]=int(1)
           Sxx_with_peaks = np.ma.masked_where(peaks_mask==0, Sxx).data

        constellation = peaks_coords
         

        list_constellation = constellation.tolist() #listes de couples indices
        list_constellation.sort(key=itemgetter(1))

        spectro = f, t, Sxx

      #   hashes = []
      #   for k in range(len(list_constellation)):
      #      for i in range(len(list_constellation)):
      #         if 0 < abs(t[anchor[1]] - t[peak[1]]) < deltaT and abs(f[anchor[0]] - f[peak[0]]) < deltaF:
      #            hashes.append({"t" : t[anchor[1]], "hash" : (t[peak[1]] - t[anchor[1]], f[anchor[0]], f[peak[0]])})

      #   self.hashes = hashes 

        N_neighbors=10
        list_hashes=[]
        for i in range(len(list_constellation)):
           f1 = f[list_constellation[i][0]-1]
           T1 = t[list_constellation[i][1]-1]
           for j in range(N_neighbors):
              if (i + j + 1) < len(list_constellation): 
                 f2 = f[list_constellation[i + j + 1][0]-1]
                 T2 = t[list_constellation[i + j + 1][1]-1]
                 list_hashes.append({"t" : T1, "hash": np.array([T2-T1,f1,f2])})
        self.hashes=list_hashes



    def display_spectrogram(self,display_anchors=True):

        """
        Display the spectrogram of the audio signal
        """
        
        f, t, Sxx = self.spectro

        plt.figure()

        plt.pcolormesh(t, f, Sxx, norm = colors.LogNorm(), shading = 'gouraud')
        plt.colorbar()
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (sec)')

        plt.show()


        if display_anchors:
           plt.scatter(self.anchors[:,0], self.anchors[:,1], color = 'r')
        plt.show()





# ----------------------------------------------------------------------------
# Compares two set of hashes in order to determine if two audio files match
# ----------------------------------------------------------------------------

class Matching:

    """
    Compare the hashes from two audio files to determine if these
    files match

    Attributes
    ----------

    hashes1: list of dictionaries
       hashes extracted as fingerprints for the first audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    hashes2: list of dictionaries
       hashes extracted as fingerprint for the second audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    matching: numpy array
       absolute times of the hashes that match together

    offset: numpy array
       time offsets between the matches
    """

    def __init__(self, hashes1, hashes2):

        """
        Class constructor

        Compare the hashes from two audio files to determine if these
        files match

        To Do
        -----

        Implement a code establishing correspondences between the hashes of
        both files. Once the correspondences computed, construct the 
        histogram of the offsets between hashes. Finally, search for a criterion
        based on the histogram that allows to determine if both audio files 
        match

        Parameters
        ----------

        hashes1: list of dictionaries
           hashes extracted as fingerprint for the first audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target

        hashes2: list of dictionaries
           hashes extracted as fingerprint for the second audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target
        """


        self.hashes1 = hashes1
        self.hashes2 = hashes2

        matching = []
        offset = []
        for h in self.hashes1:
           for k in self.hashes2:
              if max(abs(list(h.values())[0] - list(k.values())[0])) < 0.00001:
                 matching.append([h["t"],k["t"]])
                 offset.append(np.abs(h["t"]-k["t"]))
         

        self.matching = np.array(matching)
        self.offset = np.array(offset)

             
    def display_scatterplot(self):

        """
        Display through a scatterplot the times associated to the hashes
        that match
        """
         
        x = self.matching[:,0]
        y = self.matching[:,1]

        plt.scatter(x,y)
        plt.xlabel("Extrait 1")
        plt.ylabel("Extrait 2")
        plt.title("Scatterplot")
        plt.show()


    def display_histogram(self):

        """
        Display the offset histogram
        """

        plt.hist(self.offset)
        plt.title("Offset histogram")
        plt.show()

# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    encoder = Encoding()
    fs, s = read('./samples/Late truth - Audio Hertz.wav')
    encoder.process(fs, s[:900000])
    encoder.display_spectrogram(display_anchors=True)





