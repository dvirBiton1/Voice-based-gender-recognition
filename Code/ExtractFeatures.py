import librosa
import numpy as np

class ExtractFeatures:
    def __init__(self):
        pass


    def extract_features(self, audio_path):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using wav2vec2

        :param audio_path (str) : path to wave file without silent moments.
        :return: Extracted features matrix.
        """
        # initialize hop length
        hop_length = 512
        # load audio file
        y, sr = librosa.load(audio_path)
        # compute MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        # compute delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc)
        # initialize list to store features
        features = []
        # compute mean, std, min, and max for each MFCC and delta MFCC
        for mfcc_coeffs, delta_mfcc_coeffs in zip(mfcc, mfcc_delta):
            mfcc_stats = [np.mean(mfcc_coeffs), np.std(mfcc_coeffs),
                          np.amin(mfcc_coeffs), np.amax(mfcc_coeffs)]
            delta_mfcc_stats = [np.mean(delta_mfcc_coeffs), np.std(delta_mfcc_coeffs),
                                np.amin(delta_mfcc_coeffs), np.amax(delta_mfcc_coeffs)]

            # concatenate MFCC and delta MFCC stats into a single list
            features.extend(mfcc_stats + delta_mfcc_stats)
        # return feature vector as a numpy array
        return np.array(features)




