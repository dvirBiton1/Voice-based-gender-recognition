import pickle
import numpy as np
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from Code.ExtractFeatures import ExtractFeatures
from pydub import AudioSegment


class VoiceGenderRecognizer():

    def __init__(self) -> None:
        self.features_extractor = ExtractFeatures()
        self.gmm_male = pickle.load(open("gmm_male.gmm", 'rb'))
        self.gmm_female = pickle.load(open("gmm_female.gmm", 'rb'))

    def wav_converter(self,path):
        audio = AudioSegment.from_file(path, format='ogg')
        your_test= "Your_test_file.wav"
        audio.export(your_test, format='wav')
        return your_test

    def extract_features(self,file):
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                    savedir="pretrained_models/spkrec-xvect-voxceleb")
        feat_list = []
        vector = self.features_extractor.extract_features(file)
        signal, fs = torchaudio.load(file)
        embeddings = classifier.encode_batch(signal)
        embeddings = embeddings.detach().cpu().numpy()
        embedding = embeddings[0][0]
        feat_list.append(list(vector) + embedding.tolist())
        return feat_list

    def pca_calculator(self, feat_list):
        f= open("pca_model.pkl", "rb")
        pca_model = pickle.load(f)
        transformed_data = pca_model.transform(feat_list)
        f.close()
        return transformed_data

    def identify(self,path):
        file = self.wav_converter(path)
        features= self.extract_features(file)
        vector= self.pca_calculator(features)
        numpy_vector = np.array(vector).reshape(1, -1)
        is_male_score = self.gmm_male.score(numpy_vector)
        is_female_score = self.gmm_female.score(numpy_vector)
        print(f"before normalize:\nfemale score:{is_female_score} and male score: {is_male_score}")
        is_male_score, is_female_score = -1 * (is_male_score / (is_male_score + is_female_score)), (
                    is_female_score / (is_male_score + is_female_score)) * -1
        print(f"female score:{is_female_score} and male score: {is_male_score}")
        if is_female_score>is_male_score:
            return "Female!"
        else:
            return "Male!"


if __name__ == "__main__":
    VGR=  VoiceGenderRecognizer()
    prediction= VGR.identify(r"C:\Users\shira\Desktop\voices3\ohad_m.ogg")
    print(prediction)



