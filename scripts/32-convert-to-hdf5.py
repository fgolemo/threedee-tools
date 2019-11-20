import tarfile
import os
from io import BytesIO

import h5py
import numpy as np
from tqdm import tqdm

def get_fileno(x):
    if len (x.name) < 5 or (x.name[-4:] != ".png" and x.name[-4:] != ".npy"):
        return x.name
    else:
        parts = x.name.split(".")
        numbers = parts[-2].split("input")
        out = numbers[0] + str(parts[-2][-5:]) + parts[-1]
        return out

tar = tarfile.open(os.path.expanduser("~/Downloads/cubes_sample.tar")) <-- CHANGE THIS
files = tar.getmembers()
files.sort(key=get_fileno)

print(len(files))

for i in range(20):
    print(files[i])

print ("=======")
no = len(files)

out = h5py.File("cube-meta3.hdf5", "w")
light = out.create_dataset("lights", (no,3), dtype='f')
cam = out.create_dataset("cams", (no,3), dtype='f')

i = 0
buffer = {}

for member in tqdm(files):
    print (member)
    if ".npy" in member.name:
        f = tar.extractfile(member)
        array_file = BytesIO()
        array_file.write(f.read())
        array_file.seek(0)
        a = np.load(array_file)
        if ("light" in member.name):
            handle = "lights"
        elif ("_depth" not in member.name):
            handle = "cams"
        else:
            continue
        buffer[handle] = a


    if ".png" in member.name:
        f = tar.extractfile(member)
        if "_depth" in member.name:
            ftype = "depth"
        else:
            ftype = "rgb"
        outfile = open("out/{:06d}-{}.png".format(i, ftype), "wb")
        outfile.write(f.read())
        outfile.close()

        if ftype == "rgb":

            out["lights"][i] = buffer["lights"]
            out["cams"][i] = buffer["cams"]

            buffer = {}
            i += 1

out.close()
tar.close()
