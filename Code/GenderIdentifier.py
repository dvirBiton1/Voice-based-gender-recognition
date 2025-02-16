import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
from texttable import Texttable

warnings.filterwarnings("ignore")


class GenderIdentifier:

    def __init__(self, females_files_path, males_files_path, females_model_path, males_model_path):
        self.females_training_path = females_files_path
        self.males_training_path = males_files_path
        self.error = 0
        self.total_sample = 0
        self.features_extractor = FeaturesExtractor()
        # load models
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm = pickle.load(open(males_model_path, 'rb'))
        self.male_p=0
        self.female_p=0
        self.female_n=0
        self.male_n=0

    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        # read the test directory and get the list of test audio files
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features_2(file)
            winner = self.identify_gender(vector)
            expected_gender = file.split("/")[1][:-1]

            print("%10s %6s %1s" % ("+ EXPECTATION", ":", expected_gender))
            print("%10s %3s %1s" % ("+ IDENTIFICATION", ":", winner))

            my_expected=expected_gender[0]

            if my_expected[0]==winner[0]:
                if winner=="male":
                    self.male_p+=1
                else:
                    self.female_p+=1
            else:
                if winner=="male":
                    self.female_n+=1
                else:
                    self.male_n+=1

            if winner not in expected_gender:
                self.error += 1
            print("----------------------------------------------------")

        # accuracy = (float(self.total_sample - self.error) / float(self.total_sample)) * 100
        # accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        # print(accuracy_msg)
        self.confusion_matrix()

    def confusion_matrix(self):
        t = Texttable()
        t.add_rows([['', 'male_predicted','female predicted'], ['actual male', self.male_p, self.male_n], ['actual female', self.female_n, self.female_p]])
        print(t.draw())
        accuracy = ((self.male_p+self.female_p)/self.total_sample) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)

    def get_file_paths(self, females_training_path, males_training_path):
        # get file paths
        females = [os.path.join(females_training_path, f) for f in os.listdir(females_training_path)]
        males = [os.path.join(males_training_path, f) for f in os.listdir(males_training_path)]
        files = females + males
        return files

    def identify_gender(self, vector):
        # female hypothesis scoring
        is_female_scores = np.array(self.females_gmm.score(vector))
        is_female_log_likelihood = is_female_scores.sum()
        # male hypothesis scoring
        is_male_scores = np.array(self.males_gmm.score(vector))
        is_male_log_likelihood = is_male_scores.sum()

        print("%10s %5s %1s" % ("+ FEMALE SCORE", ":", str(round(is_female_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_log_likelihood, 3))))

        if is_male_log_likelihood > is_female_log_likelihood:
            winner = "male"
        else:
            winner = "female"
        return winner


if __name__ == "__main__":
    gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "females.gmm", "males.gmm")
    gender_identifier.process()
