#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import math

class PlyWriter(object):
    def __init__(self, filename):
        self.filename = filename

    def _write_header(self, ply_file, n_vertices, n_faces, has_normal_vertex=False, has_colours=False):
        ply_file.write('ply\n')
        ply_file.write('format ascii 1.0\n')
        ply_file.write('element vertex %d\n' % n_vertices)
        ply_file.write('property float x\n')
        ply_file.write('property float y\n')
        ply_file.write('property float z\n')

        if has_normal_vertex:
            ply_file.write('property float nx\n')
            ply_file.write('property float ny\n')
            ply_file.write('property float nz\n')

        if has_colours:
            ply_file.write('property uchar red\n')
            ply_file.write('property uchar green\n')
            ply_file.write('property uchar blue\n')
            ply_file.write('property uchar alpha\n')

        ply_file.write('element face %d\n' % n_faces)
        ply_file.write('property list uchar int vertex_indices\n')
        ply_file.write('end_header\n')

    def from_faces_vertices_list(self, faces, vertices, vnormals=None, colours=None):
        with open(self.filename, 'w') as ply_file:
            self._write_header(ply_file, len(vertices), len(faces), vnormals is not None, colours is not None)
            for k, v in enumerate(vertices):
                if vnormals is not None:
                    ply_file.write((' '.join(['%f' % i for i in v[:3]])))
                    nz, ny, nx = vnormals[k]
                    s = math.sqrt(nz*nz + ny*ny + nx*nx)
                    ply_file.write(' %f %f %f\n' % (nx/s, ny/s, nz/s))
                elif colours is not None:
                    ply_file.write((' '.join(['%f' % i for i in v[:3]])))
                    try:
                        ply_file.write(' %d %d %d 255\n' % colours[k])
                    except KeyError:
                        ply_file.write(' %d %d %d 255\n' % (0, 0, 0))
                else:
                    ply_file.write((' '.join(['%f' % i for i in v[:3]]) + '\n'))

            for face in faces:
                ply_file.write('%d %s\n' % (len(face), ' '.join(['%d' % i for i in face])))
