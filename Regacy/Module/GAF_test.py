import numpy as np
import matplotlib.pyplot as plt
#from pyts.image import GramianAngularField
from gaf import GramianAngularField
import matplotlib.cm as cm

x_train = np.load(r"C:\Users\USER-PC\haneol\DATASET_GAF\x_train.npy", allow_pickle=True)
x_test = np.load(r"C:\Users\USER-PC\haneol\DATASET_GAF\x_test.npy", allow_pickle=True)
lbl = np.load(r"C:\Users\USER-PC\haneol\DATASET_GAF\lbl.npy", allow_pickle=True).tolist()
test_idx = np.load(r"C:\Users\USER-PC\haneol\DATASET_GAF\test_idx.npy", allow_pickle=True).tolist()
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
test_CH_CODEs = list(map(lambda x: lbl[x][0], test_idx))
CH_CODEs = ['0', '1', '2']

test_snrs = 10
test_chcodes = CH_CODEs[0]
sample_id = 0

test_X_i = x_test[np.where(np.array(test_SNRs) == test_snrs)]
# test_X_i = test_X_i[np.where(np.array(test_mods) == str(test_mod))]

# i_data = test_X_i[sample_id, 1, :].reshape(1, -1)
# q_data = test_X_i[sample_id, 0, :].reshape(1, -1)
data = test_X_i[sample_id, 0, :].reshape(1, -1)

def generate_gaf(sin_data, image_size=128):
    # sin_data = x_train[1, 1, :].reshape(1, -1)
    # sin_data = np.loadtxt('sinx.csv', delimiter=",", skiprows=0).reshape(1, -1)
    min_ = np.amin(sin_data)
    max_ = np.amax(sin_data)
    scaled_serie = (2 * sin_data - max_ - min_) / (max_ - min_)
    gasf = GramianAngularField(image_size=image_size, method='summation',) #sample_range=(min_, max_)
    sin_gasf = gasf.transform(scaled_serie)
    gadf = GramianAngularField(image_size=image_size, method='difference',)
    sin_gadf = gadf.transform(scaled_serie)
    images = [sin_gasf[0], sin_gadf[0]]
    return images


# i_images = generate_gaf(i_data)
# q_images = generate_gaf(q_data)
demod_images = generate_gaf(data)

images = demod_images

resultant = np.array(images)
min_val, max_val = np.amin(resultant), np.amax(resultant)

titles = ['GASF image']
plt.imshow(images[0], interpolation='bilinear', cmap=cm.RdBu_r, vmin=min_val, vmax=max_val)
plt.show()