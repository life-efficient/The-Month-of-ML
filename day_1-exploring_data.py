# IMG
from PIL import Image
import numpy as np
import torch

'''
im = Image.open('photo-1518098268026-4e89f1a2cd8e.jpeg')
im.show()
print(type(im))

tensor = np.array(im)
print(tensor)
print(tensor.shape)

torch_tensor = torch.tensor(tensor)
print(torch_tensor)
'''

# CSV
'''
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('FL_insurance_sample.csv')
print(data)

my_cols = data.iloc[:, 3:9]
print(my_cols)
my_array = np.array(my_cols)
print(my_array)

t_2011 = my_array[:, -2]
t_2012 = my_array[:, -1]

idx = np.array(range(len(t_2012)))
diff = t_2012 - t_2011

pos = diff > 0
neg = diff < 0
print(pos)
print(neg)
pos_idxs = idx[pos]
pos_vals = diff[pos]
neg_idxs = idx[neg]
neg_vals = diff[neg]

plt.scatter(pos_idxs, pos_vals)
plt.scatter(neg_idxs, neg_vals)
#plt.scatter(idx, neg, c='r')

plt.show()

'''


# AUDIO
import librosa
import matplotlib.pyplot as plt
import librosa.display

audio = librosa.core.load('SampleAudio_0.4mb.mp3')
print(audio)
plt.plot(audio[0])
#plt.show()
mel = librosa.feature.melspectrogram(audio[0])
print(type(mel))
librosa.display.specshow(mel)

plt.subplot(111)
librosa.display.specshow(mel, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()

k
# VIDEO
#vid =

import pims
import matplotlib.pyplot as plt

vid = pims.Video('small.mp4')
print(len(vid))

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.show()

for frame_idx in range(len(vid) - 1):
    print(frame_idx)
    frame = vid[frame_idx]
    #print(frame)
    ax.imshow(frame)
    fig.canvas.draw()
    plt.pause(0.003)


