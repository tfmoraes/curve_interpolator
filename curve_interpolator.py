from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from optparse import OptionParser
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


def animate_and_savegif(px, py, nframes, filename):
    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(px[i], py[i])
        return (line,)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    line, = ax.plot([], [])
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=nframes, interval=0.001, blit=True)
    anim.save(filename, writer='imagemagick')


def parse_cmdline():
    parser = OptionParser()
    parser.add_option('-i', dest='init_curve', help='first curve image')
    parser.add_option('-e', dest='end_curve', help='last curve image')
    parser.add_option('-o', dest='output', help='name of the output file may be a ply (mesh) or gif (animation)')
    parser.add_option('-n', dest='interp_curves', default=50,
                      type = "int", help='number of interpolated curve')
    parser.add_option('-c', dest='curve_size', default=50,
                      type = "int", help='number of points inside each curve')
    options, args = parser.parse_args()
    return options


def main():
    options = parse_cmdline()

    img1 = imread(options.init_curve)[:, :, 0]
    points1 = np.array(img2points(img1), dtype='float64')
    points1[:, 0] = points1[:, 0] / float(img1.shape[1])
    points1[:, 1] = points1[:, 1] / float(img1.shape[0])

    img2 = imread(options.end_curve)[:, :, 0]
    points2 = np.array(img2points(img2), dtype='float64')
    points2[:, 0] = points2[:, 0] / float(img2.shape[1])
    points2[:, 1] = points2[:, 1] / float(img2.shape[0])

    curve_size = options.curve_size
    npx1, npy1 = normalize_curve(points1, curve_size)
    npx2, npy2 = normalize_curve(points2, curve_size)

    ncurves = options.interp_curves + 2
    npz3 = np.linspace(0, 1, ncurves)
    npx3, npy3 = interpolate_curve(npx1, npy1, npx2, npy2, npz3)

    output = options.output
    if output.endswith('.ply'):
        points, faces = to3dsurface(npx3, npy3, npz3)
        writer = ply_writer.PlyWriter(output)
        writer.from_faces_vertices_list(faces, points)
    elif output.endswith('.gif'):
        animate_and_savegif(npx3, npy3, ncurves, output)
    else:
        print('Please, pass a ply or gif name file')

if __name__ == "__main__":
    main()
