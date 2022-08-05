import os
from pydub import AudioSegment

dir_str = "./MP3_Sources"

directory = os.fsencode(dir_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename_wo_ext = filename[:-4]
    if filename.endswith(".mp3"):
        sound = AudioSegment.from_mp3("./MP3_Sources/" + filename)
        dst = filename_wo_ext + ".wav"
        sound.export('WAV_Converted/' + dst, format="wav")
        continue
    else:
        continue