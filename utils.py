import os
from pydub import AudioSegment

def list_files(folder):
    r = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if name[0] == '.': # don't include hidden folders
                continue
            else:
                r.append(os.path.join(root, name))
    return r
