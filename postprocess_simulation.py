import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


def rotate(x, y, a):
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)


def main():
    y_pitch = 4.5625
    z_pitch = 4.5625

    rot_angle = (-1.) * np.pi * np.array(range(14)) / 7. + np.pi

    bins_edges = -np.pi + np.pi / 7. * np.array(range(15)) - np.pi / 14

    images = []

    for file in tqdm(range(1000)):
        for i in range(200):
            fates_filename = "/data/stpidhorskyi/Detect2000/%d/FATES%s" % (file + 1, "" if i == 0 else str(i))
            lineList = [line.rstrip('\n') for line in open(fates_filename)]
            data = []
            for line in lineList:
                if line == '':
                    continue
                results = line.split()
                fate = int(results[0])
                if fate == 1:
                    x_data, y_data, z_data = map(float, [results[8], results[9], results[10]])
                    data.append([x_data, y_data, z_data])

            data = np.stack(data)

            x_data, y_data, z_data = data[:, 0], data[:, 1], data[:, 2]

            phi = np.arctan2(y_data, x_data)
            phi[phi < bins_edges[0]] += 2 * np.pi

            r = np.sqrt(x_data ** 2 + y_data ** 2 + z_data ** 2)

            ang_ind = np.digitize(phi, bins_edges) - 1
            h, _ = np.histogram(phi, bins_edges)
            max_ang = np.argmax(h)

            correct = max_ang == 7
            max_ang = 7

            if not correct:
                plt.hist(phi, bins_edges)
                plt.show()

                plt.scatter(x_data, y_data, marker='^', s=1)
                plt.scatter(40 * np.cos(bins_edges), 40 * np.sin(bins_edges), marker='o', s=3)
                plt.axis('equal')
                plt.show()

            x_0, y_0 = rotate(x_data[ang_ind == max_ang], y_data[ang_ind == max_ang], rot_angle[max_ang])
            z_0 = z_data[ang_ind == max_ang]

            one_down = (14 + max_ang - 1) % 14
            shift = 9.12335829*2.
            shift = 38.997 * math.tan(math.pi/14) * 2

            x_down, y_down = rotate(x_data[ang_ind == one_down], y_data[ang_ind == one_down], rot_angle[one_down])
            z_down = z_data[ang_ind == one_down]
            y_down -= shift

            one_up = (14 + max_ang + 1) % 14

            x_up, y_up = rotate(x_data[ang_ind == one_up], y_data[ang_ind == one_up], rot_angle[one_up])
            z_up = z_data[ang_ind == one_up]
            y_up += shift

            y_fit = np.concatenate([y_down, y_0, y_up])
            z_fit = np.concatenate([z_down, z_0, z_up])

            if not correct:
                plt.scatter(y_fit, z_fit, marker='^', s=1)
                plt.show()

            img, xedges, yedges = np.histogram2d(y_fit, z_fit, [12, 16], [[-27.375, 27.375], [-36.1, -36.1 + 16 * y_pitch]])

            images.append((img, file, i))

    with open('data.pkl', 'wb') as pkl:
        pickle.dump(images, pkl)


if __name__ == "__main__":
    main()
