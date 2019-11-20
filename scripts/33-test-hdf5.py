import h5py
import matplotlib.pyplot as plt

out = h5py.File("cube-meta2.hdf5", "r")

for i in range(3):
    print (out["cams"][i])
    print (out["lights"][i])
    # img = out["imgs"][i]
    # plt.imshow(img)
    # plt.show()
