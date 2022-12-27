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
        # initialize features
        hop_length = 512
        # load file
        y, sr = librosa.load(audio_path)
        # extract mfcc coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        # extract mean, standard deviation, min, and max value in mfcc frame, do this across all mfcc's
        mfcc_features = np.array([np.mean(mfcc[0]), np.std(mfcc[0]), np.amin(mfcc[0]), np.amax(mfcc[0]),
                                  np.mean(mfcc[1]), np.std(mfcc[1]), np.amin(mfcc[1]), np.amax(mfcc[1]),
                                  np.mean(mfcc[2]), np.std(mfcc[2]), np.amin(mfcc[2]), np.amax(mfcc[2]),
                                  np.mean(mfcc[3]), np.std(mfcc[3]), np.amin(mfcc[3]), np.amax(mfcc[3]),
                                  np.mean(mfcc[4]), np.std(mfcc[4]), np.amin(mfcc[4]), np.amax(mfcc[4]),
                                  np.mean(mfcc[5]), np.std(mfcc[5]), np.amin(mfcc[5]), np.amax(mfcc[5]),
                                  np.mean(mfcc[6]), np.std(mfcc[6]), np.amin(mfcc[6]), np.amax(mfcc[6]),
                                  np.mean(mfcc[7]), np.std(mfcc[7]), np.amin(mfcc[7]), np.amax(mfcc[7]),
                                  np.mean(mfcc[8]), np.std(mfcc[8]), np.amin(mfcc[8]), np.amax(mfcc[8]),
                                  np.mean(mfcc[9]), np.std(mfcc[9]), np.amin(mfcc[9]), np.amax(mfcc[9]),
                                  np.mean(mfcc[10]), np.std(mfcc[10]), np.amin(mfcc[10]), np.amax(mfcc[10]),
                                  np.mean(mfcc[11]), np.std(mfcc[11]), np.amin(mfcc[11]), np.amax(mfcc[11]),
                                  np.mean(mfcc[12]), np.std(mfcc[12]), np.amin(mfcc[12]), np.amax(mfcc[12]),
                                  np.mean(mfcc_delta[0]), np.std(mfcc_delta[0]), np.amin(mfcc_delta[0]),
                                  np.amax(mfcc_delta[0]),
                                  np.mean(mfcc_delta[1]), np.std(mfcc_delta[1]), np.amin(mfcc_delta[1]),
                                  np.amax(mfcc_delta[1]),
                                  np.mean(mfcc_delta[2]), np.std(mfcc_delta[2]), np.amin(mfcc_delta[2]),
                                  np.amax(mfcc_delta[2]),
                                  np.mean(mfcc_delta[3]), np.std(mfcc_delta[3]), np.amin(mfcc_delta[3]),
                                  np.amax(mfcc_delta[3]),
                                  np.mean(mfcc_delta[4]), np.std(mfcc_delta[4]), np.amin(mfcc_delta[4]),
                                  np.amax(mfcc_delta[4]),
                                  np.mean(mfcc_delta[5]), np.std(mfcc_delta[5]), np.amin(mfcc_delta[5]),
                                  np.amax(mfcc_delta[5]),
                                  np.mean(mfcc_delta[6]), np.std(mfcc_delta[6]), np.amin(mfcc_delta[6]),
                                  np.amax(mfcc_delta[6]),
                                  np.mean(mfcc_delta[7]), np.std(mfcc_delta[7]), np.amin(mfcc_delta[7]),
                                  np.amax(mfcc_delta[7]),
                                  np.mean(mfcc_delta[8]), np.std(mfcc_delta[8]), np.amin(mfcc_delta[8]),
                                  np.amax(mfcc_delta[8]),
                                  np.mean(mfcc_delta[9]), np.std(mfcc_delta[9]), np.amin(mfcc_delta[9]),
                                  np.amax(mfcc_delta[9]),
                                  np.mean(mfcc_delta[10]), np.std(mfcc_delta[10]), np.amin(mfcc_delta[10]),
                                  np.amax(mfcc_delta[10]),
                                  np.mean(mfcc_delta[11]), np.std(mfcc_delta[11]), np.amin(mfcc_delta[11]),
                                  np.amax(mfcc_delta[11]),
                                  np.mean(mfcc_delta[12]), np.std(mfcc_delta[12]), np.amin(mfcc_delta[12]),
                                  np.amax(mfcc_delta[12])])

        return mfcc_features


