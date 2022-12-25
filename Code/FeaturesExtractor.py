import io
import time
import librosa
import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from pydub import AudioSegment
from sklearn import preprocessing
from scipy.io.wavfile import read
from python_speech_features import mfcc
from python_speech_features import delta
# import os
# import IPython
# import matplotlib
# import matplotlib.pyplot as plt
# import torch
# import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import wav2vec
from speechbrain.pretrained import EncoderClassifier

class FeaturesExtractor:
    def __init__(self):
        pass
       
    def extract_features(self, audio_path):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double
        deltas.
     
        Args: 	    
            audio_path (str) : path to wave file without silent moments. 
        Returns: 	    
            (array) : Extracted features matrix. 	
        """
        rate, audio  = read(audio_path)
        mfcc_feature = mfcc(# The audio signal from which to compute features.
                            audio,
                            # The samplerate of the signal we are working with.
                            rate,
                            # The length of the analysis window in seconds. 
                            # Default is 0.025s (25 milliseconds)
                            winlen       = 0.05,
                            # The step between successive windows in seconds. 
                            # Default is 0.01s (10 milliseconds)
                            winstep      = 0.01,
                            # The number of cepstrum to return. 
                            # Default 13.
                            numcep       = 5,
                            # The number of filters in the filterbank.
                            # Default is 26.
                            nfilt        = 30,
                            # The FFT size. Default is 512.
                            nfft         = 512,
                            # If true, the zeroth cepstral coefficient is replaced 
                            # with the log of the total frame energy.
                            appendEnergy = True)
    
        
        mfcc_feature  = preprocessing.scale(mfcc_feature)
        deltas        = delta(mfcc_feature, 2)
        double_deltas = delta(deltas, 2)
        combined      = np.hstack((mfcc_feature, deltas, double_deltas))
        return combined
    def extract_features_2(self, audio_path):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using wav2vec2

        Args:
            audio_path (str) : path to wave file without silent moments.
        Returns:
            (array) : Extracted features matrix.
        """
        torch.random.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(device)
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(device)

        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                    savedir="pretrained_models/spkrec-xvect-voxceleb")
        xvector = classifier.encode_batch(waveform)
        xvector = xvector.detach().cpu().numpy()
        xvector = xvector[0][0]

        with torch.inference_mode():
            features, _ = model.extract_features(waveform)
            features = np.concatenate((features,xvector)).reshape(1,-1)
        # with torch.inference_mode():
        #     emission, _ = model(waveform)
        #     if emission.shape[2] != 562:
        #         return None
        #     print("features len: ", len(features)," and ", len(features[0]))
        #     print("------------------------------------")
        #     print("emission len: ",len(emission), " and ", len(emission[0]))
        print(features.shape)
        return features

    def extract_features_3(self, audio_path):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using wav2vec2

        Args:
            audio_path (str) : path to wave file without silent moments.
        Returns:
            (array) : Extracted features matrix.
        """
        audio, _ = torchaudio.load(audio_path)
        model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        features = model(audio)
        return features


