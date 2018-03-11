import scipy.io as spio
import os
import numpy as np
from random import randint
from untar import extract


def matlab_file_to_cad(matlab_file_path):
    return spio.loadmat(matlab_file_path, squeeze_me=True).get('instance')


def prepare_data_set(dataset_dir, batch_size, channels, limit=None, balanced=True, fuzzing_mode=False, num_of_voxels_to_augment=0):
    cad_vector =[]
    label_vector = []
    counter = 0
    labels = os.listdir(dataset_dir)
    data_set_sizes = [len(os.listdir(os.path.join(dataset_dir, label))) for label in labels]
    data_set_sizes_all_equal = len(set(data_set_sizes)) == 1
    if limit is None and balanced and not data_set_sizes_all_equal:
        limit = max(data_set_sizes)
    for i, label in enumerate(labels):
        print "creating data for lable:", label, "--", "ord:", i
        l = np.zeros(len(labels), dtype=int)
        l[i] = 1

        for raw_data_path in os.listdir(os.path.join(dataset_dir, label))[:limit]:
            if fuzzing_mode:
                # NEVER SET THIS FLAG TRUE unless you know what you're doing
                l[i] = randint(0, len(labels) - 1)
            cad = matlab_file_to_cad(os.path.join(dataset_dir, label, raw_data_path))
            fuzz_cad(cad, num_of_voxels_to_augment)
            cad = cad.reshape(cad.shape[0], cad.shape[1], cad.shape[2], channels)
            counter +=1
            cad_vector.append(cad)
            label_vector.append(l)
                #batch.append[cad,[l]]
            if counter % batch_size == 0:

                batch = np.array(cad_vector), np.array(label_vector)
                yield batch
                cad_vector = []
                label_vector = []


def prepare_data_set_smart_wrapper(dataset_dir, batch_size, channels, limit=None, balanced=True, fuzzing_mode=False):
    tar_gz_suffix = ".tar.gz"
    if dataset_dir.endswith(tar_gz_suffix):
        print "untaring dataset: {}, please wait".format(dataset_dir)
        extract(dataset_dir)
        return prepare_data_set(dataset_dir[:dataset_dir.index(tar_gz_suffix)],  batch_size, channels, limit=limit, balanced=balanced, fuzzing_mode=fuzzing_mode)
    return prepare_data_set(dataset_dir, batch_size, channels, limit=limit, balanced=balanced, fuzzing_mode=fuzzing_mode)


def flip_bit(bit):
    flipper = {1:0, 0:1}
    return flipper[bit]


def choose_coordinates(dims, vec_size):
    coords = []
    for _ in range(dims):
        coords.append(randint(0, vec_size - 1))
    return coords


def fuzz_cad(cad, num_of_flips):
    for _ in range(num_of_flips):
        x, y, z = choose_coordinates(3, len(cad))
        cad[x][y][z] = flip_bit(cad[x][y][z])


if __name__ == "__main__":
    data_gen = prepare_data_set("train_cad")
    for _ in data_gen:
        print _