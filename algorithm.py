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

        self.window = window
        self.window_size = window_size


    def process(self, fs, s, deltaT = 1000, deltaF = 50):

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

        spectro = spectrogram(s, fs, noverlap=32, nperseg=128)
        f, t, Sxx = spectro
        self.spectro = spectro
        peak = peak_local_max(Sxx, min_distance= 50, exclude_border=False)
        self.anchors = peak

        hashes = []
        for i, anchor in enumerate(self.anchors):
           for peak in self.anchors[i:]:
              if (abs(anchor[1] - peak[1]) < deltaT) and (abs(anchor[0] - peak[0]) < deltaF) and (anchor[1] - peak[1] > 0):
                 hashes.append({"t" : anchor[1], "hash" : (peak[1] - anchor[1], anchor[0], peak[0])})
        self.hashes = hashes 



    def display_spectrogram(self,display_anchors=True):

        """
        Display the spectrogram of the audio signal
        """
        
        f, t, Sxx = self.spectro
        plt.pcolormesh(t, f, Sxx, norm = colors.LogNorm(), shading = "auto")
        plt.colorbar()

        if display_anchors :
           plt.scatter(t[self.anchors[:, 1]], f[self.anchors[:, 0]], color = 'r')

        plt.colorbar()
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (sec)')
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
        i=0
        for h in self.hashes1:
           for k in self.hashes2:
              i+=1
              if h["hash"] == k["hash"] :
                  matching.append([h["t"],k["t"]])
         
        self.matching = np.array(matching)

             
    def display_scatterplot(self):

        """
        Display through a scatterplot the times associated to the hashes
        that match.
        """

        print(len(self.matching[:, 1]), len(self.matching[:, 0]))
        plt.scatter(self.matching[:, 0],self.matching[:, 1], s= 1)
        plt.xlabel("Extrait 1")
        plt.ylabel("Extrait 2")
        plt.title("Scatterplot")
        plt.axis('equal')
        plt.show()


    def display_histogram(self):

        """
        Display the offset histogram
        """

        H = []
        for k in range(len(self.matching)):
           H.append(self.matching[k,0] - self.matching[k,1])

        plt.hist(H, 100)
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





