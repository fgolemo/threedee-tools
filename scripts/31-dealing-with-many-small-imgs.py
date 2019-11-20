import glob
# i = 0
# for file in glob.iglob("/Users/florian/Downloads/vis_input/*.png"):
#     i+= 1
#
# print (i)

# no = 438957
# os.path.getsize('C:\\Python27\\Lib\\genericpath.py')

import tarfile
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

tar = tarfile.open("/Users/florian/Downloads/cube.tar")
files = tar.getmembers()

print(len(files))
for i in range(10):
    print(files[i])

i = 0
for member in files:
    print (dir(member))
    # if ".txt" in member.name:
    #     print(member.name)
    #     f = tar.extractfile(member)
    #     content = f.read()
    #     print(content)

    if ".npy" in member.name:
        print (member.name)
        f = tar.extractfile(member)
        array_file = BytesIO()
        array_file.write(f.read())
        array_file.seek(0)
        a = np.load(array_file)
        print (a)

    if ".png" in member.name:
        f = tar.extractfile(member)
        img = plt.imread(f)
        print (np.min(img), np.max(img))
        plt.imshow(img)
        plt.title(member.name)
        plt.show()

    i += 1
    if i >= 10:
        break

tar.close()
