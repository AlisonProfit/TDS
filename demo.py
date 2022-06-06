"""
Description
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from algorithm import *

# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    # 1: Load the database
    with open('songs.pickle', 'rb') as handle:
        database = pickle.load(handle)

    # 2: Create an instance of the class Encoder
    # Insert code here
    encoder = Encoding()

    # 3: Randomly get an extract from one of the songs of the database
    songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']
    song = random.choice(songs)
    print('Selected song: ' + song[:-4])
    filename = './samples/' + song

    fs, s = read(filename)
    tmin = int(50*fs) # We select an extract starting at 50s ...
    duration = int(10*fs) # ... which lasts 10s

    # 4: Use the encoder to extract a fingerprint of the sample
    encoder.process(fs, s[tmin:tmin + duration])
    hashes1 = encoder.hashes
    # plt.scatter(encoder.anchors[:, 0], encoder.anchors[:, 1])
    # 5: Using the class Matching, compare the fingerprint to all the 
    # fingerprints in the database

    # Insert code here
    
    # print(hashes1) 

    song = random.choice(songs)
    print('Selected song: ' + song[:-4])
    filename = './samples/' + song
    
    fs, s = read(filename)
    hashes2 = encoder.process(fs, s)
    plt.scatter(encoder.anchors[:, 1], encoder.anchors[:, 0])
    plt.show()
    # print(hashes2)

    match = Matching(hashes2, hashes1)
    # match.display_histogram()
    match.display_scatterplot()
    match.display_histogram()





