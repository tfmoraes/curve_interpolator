from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.misc import imread

import ply_writer

def img2points(img):
    bw_img = (img > 127) * 1
    points = []
    for y in range(bw_img.shape[0]):
        for x in range(bw_img.shape[1]):
            if bw_img[y, x]:
                points.append((x, y))
                break
        if points:
            break

    xi, yi = points[-1]
    while True:
        xn, yn = points[-1]
        nexts = [(xn+i, yn+j) for i,j in ((+1, 0), (0, +1), (-1, 0), (0, -1),
                                          (+1, +1), (-1, +1), (-1, -1), (+1, -1)) \
                 if ((0 < xn + i < img.shape[1]) and (0 < yn + j < img.shape[0]))]
        appended = False
        for x, y in nexts:
            if bw_img[y, x] == 1:
                bw_img[y, x] = 2
                points.append((x, y))
                appended = True
                break

        if not appended:
            return points


def normalize_curve(points, npoints):
    px = points[:, 0]
    py = points[:, 1]

    t = np.linspace(0, 1, len(points))
    fx = interpolate.CubicSpline(t, px)
    fy = interpolate.CubicSpline(t, py)

    nt = np.linspace(0, 1, npoints)
    npx = fx(nt)
    npy = fy(nt)

    return npx, npy


def interpolate_curve(px1, py1, px2, py2, z):
    px3 = np.zeros(shape=(z.shape[0], px1.shape[0]))
    py3 = np.zeros(shape=(z.shape[0], py1.shape[0]))

    t = np.array((0.0, 1.0))

    for i in range(px1.shape[0]):
        fx = interpolate.CubicSpline(t, np.array((px1[i], px2[i])))
        fy = interpolate.CubicSpline(t, np.array((py1[i], py2[i])))

        px3[:, i] = fx(z)
        py3[:, i] = fy(z)

    return px3, py3


def to3dsurface(px, py, pz):
    points = np.zeros(shape=(px.size, 3))
    faces = []
    k = 0
    for i in range(px.shape[0]):
        for j in range(px.shape[1]):
            x = px[i, j]
            y = py[i, j]
            z = pz[i]
            points[k] = x, y, z
            k += 1

    for i in range(px.shape[0]-1):
        for j in range(px.shape[1] - 1):
            faces.append((i*px.shape[1] + j, i*px.shape[1] + j + 1, (i+1)*px.shape[1] + j))
            faces.append((i*px.shape[1] + j + 1, (i+1)*px.shape[1] + j + 1, (i+1)*px.shape[1] + j))

    return points, np.array(faces)

def main():
    img1 = imread(sys.argv[1])[:, :, 0]
    points1 = np.array(img2points(img1), dtype='float64')
    points1[:, 0] = points1[:, 0] / float(img1.shape[1])
    points1[:, 1] = points1[:, 1] / float(img1.shape[0])

    img2 = imread(sys.argv[2])[:, :, 0]
    points2 = np.array(img2points(img2), dtype='float64')
    points2[:, 0] = points2[:, 0] / float(img2.shape[1])
    points2[:, 1] = points2[:, 1] / float(img2.shape[0])

    npx1, npy1 = normalize_curve(points1, 50)
    npx2, npy2 = normalize_curve(points2, 50)

    npz3 = np.linspace(0, 1, 50)
    npx3, npy3 = interpolate_curve(npx1, npy1, npx2, npy2, npz3)

    points, faces = to3dsurface(npx3, npy3, npz3)

    writer = ply_writer.PlyWriter(sys.argv[3])
    writer.from_faces_vertices_list(faces, points)
    plt.plot(npx1, npy1, label='curve 1')
    plt.plot(npx2, npy2, label='curve 2')
    for i in range(10):
        plt.plot(npx3[i], npy3[i], label='curve %d' % i)
    plt.show()

if __name__ == "__main__":
    main()
