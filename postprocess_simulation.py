import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import re

def rotate(x, y, angle):
    return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)

def main():
    y_pitch = 4.5625
    z_pitch = 4.5625

    rot_angles = (-1.) * np.pi * np.arange(14) / 7. + np.pi

    bins_edges = -np.pi + np.pi / 7. * np.arange(15) - np.pi / 14

    images = []
    coordinates_dict = {}
    coordinates = []

    for file_num in tqdm(range(1, 1001)):
        input_filename = f"/data/PETScanner/data_new/input_files/input_mult_{file_num}.det"
        with open(input_filename, 'r') as file:
            for line in file:
                if line.startswith('GEN,MAT1,'):
                    match = re.match(r"GEN,MAT1,(-*\d*.\d*)XS,(-*\d*.\d*)XL,(-*\d*.\d*)YS,(-*\d*.\d*)YL,(-*\d*.\d*)ZS,(-*\d*.\d*)ZL", line)
                    if match:
                        x, y, z = map(float, match.groups())
                        coordinates.append((x, y, z))
                        coordinates_dict.setdefault(file_num, {})[line.strip()] = (x, y, z)

    coordinates = np.asarray(coordinates, dtype=np.float32)

    plt.scatter(coordinates[:, 0], coordinates[:, 2], marker='^', s=1)
    plt.show()

    for file_num in tqdm(range(1, 1000)):
        for i in range(200):
            fates_filename = f"/data/PETScanner/data_new/{file_num}/FATES{''.join([str(i)] if i != 0 else [])}"
            with open(fates_filename, 'r') as file:
                data = []
                for line in file:
                    if line.strip() == '':
                        continue
                    results = line.split()
                    fate = int(results[0])
                    if fate == 1:
                        x_data, y_data, z_data = map(float, results[8:11])
                        data.append([x_data, y_data, z_data])

                data = np.array(data)

                phi = np.arctan2(data[:, 1], data[:, 0])
                phi[phi < bins_edges[0]] += 2 * np.pi

                r = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2 + data[:, 2] ** 2)

                ang_ind = np.digitize(phi, bins_edges) - 1
                h, _ = np.histogram(phi, bins_edges)
                max_ang = np.argmax(h)

                correct = max_ang == 7

                if not correct:
                    continue

                x_0, y_0 = rotate(data[ang_ind == max_ang, 0], data[ang_ind == max_ang, 1], rot_angles[max_ang])
                z_0 = data[ang_ind == max_ang, 2]

                one_down = (14 + max_ang - 1) % 14
                shift = 38.997 * math.tan(math.pi / 14) * 2

                x_down, y_down = rotate(data[ang_ind == one_down, 0], data[ang_ind == one_down, 1], rot_angles[one_down])
                z_down = data[ang_ind == one_down, 2]
                y_down -= shift

                one_up = (14 + max_ang + 1) % 14

                x_up, y_up = rotate(data[ang_ind == one_up, 0], data[ang_ind == one_up, 1], rot_angles[one_up])
                z_up = data[ang_ind == one_up, 2]
                y_up += shift

                y_fit = np.concatenate([y_down, y_0, y_up])
                z_fit = np.concatenate([z_down, z_0, z_up])

                if not correct:
                    plt.scatter(y_fit, z_fit, marker='^', s=1)
                    plt.show()

                img, _, _ = np.histogram2d(y_fit, z_fit, [12, 16], [[-27.375, 27.375], [-36.1, -36.1 + 16 * z_pitch]])
                images.append((img, coordinates_dict[file_num]["FATES{}".format('' if i == 0 else str(i))]))

    with open('/data/PETScanner/data_new/data.pkl', 'wb') as pkl:
        pickle.dump(images, pkl)

if __name__ == "__main__":
    main()
