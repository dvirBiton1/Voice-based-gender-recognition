import os
import pickle
import warnings
import numpy as np
import torch
import torchaudio
from sklearn import mixture
from speechbrain.pretrained import EncoderClassifier
from texttable import Texttable
from ExtractFeatures import ExtractFeatures
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class VoiceGenderRecognizer:
    def __init__(self, females_files_path_train, males_files_path_train,females_files_path_test, males_files_path_test):
        self.females_files_path_train = females_files_path_train
        self.males_files_path_train = males_files_path_train
        self.females_files_path_test= females_files_path_test
        self.males_files_path_test = males_files_path_test
        self.features_extractor = ExtractFeatures()
        self.males_train_len=0
        self.females_train_len = 0
        self.females_test_len = 0
        self.males_test_len = 0
        self.features_matrix_transformed= []


    def get_files_from_paths(self, females_training_path, males_training_path,females_test_path, males_test_path):
        """
        This method get 4 paths of file directory's (Division by gender and train/test) and return 4 lists of those files
        :param females_training_path:
        :param males_training_path:
        :param females_test_path:
        :param males_test_path:
        :return: 4 lists of files for each path
        """
        females_train = [os.path.join(females_training_path, f) for f in os.listdir(females_training_path)]
        males_train = [os.path.join(males_training_path, f) for f in os.listdir(males_training_path)]
        females_test = [os.path.join(females_test_path, f) for f in os.listdir(females_test_path)]
        males_test = [os.path.join(males_test_path, f) for f in os.listdir(males_test_path)]
        return females_train, males_train ,females_test, males_test

    def gender_train_test_adapter(self):
        """
        This method combined all the 4th lists together to one list of files
        :return: combined list
        """
        f_train, m_train , f_test, m_test = self.get_files_from_paths(self.females_files_path_train, self.males_files_path_train, self.females_files_path_test, self.males_files_path_test)
        combined_list= f_train+m_train+f_test+m_test
        self.males_train_len=len(m_train)
        self.females_train_len = len(f_train)
        self.females_test_len = len(f_test)
        self.males_test_len = len(m_test)
        return combined_list

    def collect_features_from_all_data(self,files):
        """
        This method get a list of files, extract features for each file (by the featureExtractor) then
        combined those features with wav2vec library extract_features
        :param files:
        :return: matrix (list of lists) of all features extracted from the files
        """
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                    savedir="pretrained_models/spkrec-xvect-voxceleb")
        feat_list = []
        for file in files:
            print("%5s %10s" % ("PROCESSNG ", file))
            # extract MFCC & delta MFCC features from audio
            vector = self.features_extractor.extract_features(file)
            signal, fs = torchaudio.load(file)
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.detach().cpu().numpy()
            embedding = embeddings[0][0]
            feat_list.append(list(vector) + embedding.tolist())
        return feat_list

    def pca_calculator(self, matrix):
        """
        This method get a matrix and by the PCA return "Compressed data"
        :param matrix:
        :return:Linear transformation of the matrix according to PCA
        """
        df = pd.DataFrame(matrix)
        pca= PCA()
        pca.fit(df)

        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title('pca_before')
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.savefig("pca_plots/before_pca_graph.png")

        n_comp=self.n_components_calculator_by_slope(pca.explained_variance_ratio_)
        pca2= PCA(n_components=n_comp)
        pca2.fit(df)

        plt.plot(np.cumsum(pca2.explained_variance_ratio_))
        plt.title('pca_after')
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.savefig("pca_plots/after_pca_graph.png")

        transformed_data = pca2.transform(df)

        return transformed_data


    def n_components_calculator_by_slope(self, variance_ratio):
        """
        This method calculate the recommended n_components for the PCA according
         to the variance ratio sent and by a 0.00015 threshold
        :param variance_ratio:
        :return: The recommended n_components for the PCA
        """
        epsilon = 0.0000000015  # threshold for the slope
        for i in range(1, len(variance_ratio)):
            # calculate the slope between the current and previous explained variance ratios
            slope = variance_ratio[i] - variance_ratio[i - 1]
            if slope < epsilon:
                pass
                # print(f"PCA return i was: {i}")
                # return i
        # if something got wrong return 95% of the features
        return np.argmax(np.cumsum(variance_ratio) >= 0.99) + 1

    def build_and_fit_model(self):
        """
        This method Build Gaussian mixture model and fit it
        on the train files
        """
        combined_list= self.gender_train_test_adapter()
        features_matrix= self.collect_features_from_all_data(combined_list)
        self.features_matrix_transformed= self.pca_calculator(features_matrix)
        print(self.features_matrix_transformed)
        print(f"len:{len(self.features_matrix_transformed)} and len[0]: {len(self.features_matrix_transformed[0])}")
        mid= self.females_train_len
        mid2= self.males_train_len
        train_only_features_female= self.features_matrix_transformed[0:mid]
        train_only_features_male = self.features_matrix_transformed[mid:mid+mid2]
        gmm_male = mixture.GaussianMixture(n_components=1, covariance_type='diag', n_init=3)
        gmm_female = mixture.GaussianMixture(n_components=1, covariance_type='diag', n_init=3)
        gmm_male.fit(train_only_features_male)
        gmm_female.fit(train_only_features_female)
        self.save_gmm(gmm_male, "gmm_male")
        self.save_gmm(gmm_female, "gmm_female")


    def save_gmm(self, gmm, name):
        """
        Save Gaussian mixture model using pickle.
        :param gmm: Gaussian mixture model
        :param name (str) : File name.
        """
        filename = name + ".gmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print("%5s %10s" % ("SAVING", filename,))

    # def codetosave(self):
    #     gmm = pickle.load(open("VoiceGenderRecognizer.gmm", 'rb'))
    #     gmm.classes_ = np.array(['female', 'male'])
    #     mid= self.males_train_len+self.females_train_len
    #     test_only_features= self.features_matrix_transformed[mid:]
    #     # predict the class probabilities for a the test samples
    #     predictions= gmm.predict(test_only_features)
    #     gender_mid= self.females_test_len
    #     actual="female"
    #     male_predicted=0
    #     female_predicted=0
    #     male_predicted_false = 0
    #     female_predicted_false = 0
    #     print("ml",self.males_test_len,"fl", self.females_test_len)
    #     for i , prediction in enumerate(predictions):
    #         if i<gender_mid: # female case
    #             print(f"predict is: {gmm.classes_[prediction]}")
    #             if gmm.classes_[prediction]=="female":
    #                 print(f"female predicted! actually: {actual}")
    #                 female_predicted+=1
    #             else:
    #                 print(f"male predicted! actually: {actual}")
    #                 female_predicted_false+=1
    #         else: # male case
    #             print(f"predict is: {gmm.classes_[prediction]}")
    #             actual="male"
    #             if gmm.classes_[prediction]=="female":
    #                 print(f"female predicted! actually: {actual}")
    #                 male_predicted_false+=1
    #             else:
    #                 print(f"male predicted! actually: {actual}")
    #                 male_predicted+=1
    #     self.confusion_matrix(male_predicted, male_predicted_false, female_predicted_false, female_predicted)
        # print("---------- from here -----")
        # predictions = gmm.predict_proba(test_only_features)
        # gender_mid= self.females_test_len
        # actual="female"
        # male_predicted=0
        # female_predicted=0
        # male_predicted_false = 0
        # female_predicted_false = 0
        # print("---------")
        # print("ml",self.males_test_len,"fl", self.females_test_len)
        # for i , probabilities in enumerate(predictions):
        #     if i<gender_mid: # female case
        #         if probabilities[0]>probabilities[1]:
        #             print(f"female predicted! actually: {actual}")
        #             female_predicted+=1
        #         else:
        #             print(f"male predicted! actually: {actual}")
        #             female_predicted_false+=1
        #     else: # male case
        #         actual="male"
        #         if probabilities[0]>probabilities[1]:
        #             print(f"female predicted! actually: {actual}")
        #             male_predicted_false+=1
        #         else:
        #             print(f"male predicted! actually: {actual}")
        #             male_predicted+=1
        # self.confusion_matrix(male_predicted,male_predicted_false,female_predicted_false,female_predicted)
        # print("---------- to here -----")



    def identify_gender(self):
        """
        This method going through all the test samples and identify_gender
        by the gmm model saved
        calculate and print the accuracy and the confusion matrix of the model
        """
        gmm_male = pickle.load(open("gmm_male.gmm", 'rb'))
        gmm_female = pickle.load(open("gmm_female.gmm", 'rb'))
        mid= self.males_train_len+self.females_train_len
        test_only_features= self.features_matrix_transformed[mid:]
        gender_mid = self.females_test_len
        actual="female"
        male_predicted=0
        female_predicted=0
        male_predicted_false = 0
        female_predicted_false = 0
        for i, vector in enumerate(test_only_features):
            numpy_vector = np.array(vector).reshape(1, -1)
            is_male_score = gmm_male.score(numpy_vector)
            is_female_score = gmm_female.score(numpy_vector)
            if i<gender_mid:
                if is_female_score>is_male_score:
                    print(f"female predicted! actually: {actual}")
                    female_predicted+=1
                else:
                    print(f"male predicted! actually: {actual}")
                    female_predicted_false+=1
            else:
                actual="male"
                if is_female_score>is_male_score:
                    print(f"female predicted! actually: {actual}")
                    male_predicted_false+=1
                else:
                    print(f"male predicted! actually: {actual}")
                    male_predicted+=1
        self.confusion_matrix(male_predicted, male_predicted_false, female_predicted_false, female_predicted)


    def confusion_matrix(self,male_p,male_n,female_n,female_p):
        """
        This method print the accuracy of the model and print a confusion matrix of a the model
        """
        total_sample=  male_p + male_n + female_p + female_n
        t = Texttable()
        t.add_rows([['', 'male_predicted','female predicted'], ['actual male', male_p, male_n], ['actual female', female_n, female_p]])
        print(t.draw())
        accuracy = ((male_p+female_p)/total_sample) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 4)) + "% ***"
        print(accuracy_msg)

    def main(self):
        """
        This method build and train model
        and then test it
        """
        self.build_and_fit_model()
        # print("------ waiting for enter to continue -------")
        print("finished to train model! \n starting to test it! \n")
        self.identify_gender()

if __name__ == "__main__":
    gender_recognizer = VoiceGenderRecognizer("Train_to_Save/TrainingData/females", "Train_to_Save/TrainingData/males","TestingData/females", "TestingData/males")
    # gender_recognizer = VoiceGenderRecognizer("TrainingData/females", "TrainingData/males","TestingData_2/females", "TestingData_2/males")

    gender_recognizer.main()



