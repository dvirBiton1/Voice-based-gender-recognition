from pydub import AudioSegment
import os

i = 501


def con(file1, file2):
    sound1 = AudioSegment.from_wav(file1 + ".wav")
    sound2 = AudioSegment.from_wav(file2 + ".wav")
    # mix sound2 with sound1, starting at 5000ms into sound1)
    output = sound1.overlay(sound2, position=2000)
    # save the result
    output.export(f"{file1}_1" + ".wav", format="wav")


if __name__ == '__main__':
    directory = '.'
    file_to_mix = "airplane-fly-over-02a.wav"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and "test" not in f:
            os.rename(f, "fm" + str(i) + ".wav".format("wav"))
            i += 1
            print(f)
