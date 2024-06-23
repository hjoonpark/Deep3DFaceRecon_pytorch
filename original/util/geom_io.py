"""
geom_io.py

Reader / writer of geometry formats
"""
import numpy as np

# from __future__ import print_function
# from general import Timer

# def writeOBJ(filepath, vertices, connectivity, bufsize=1):
#     """
#     Super-limited OBJ writer.
#     Only supports minimal geometry information (vertex positions, connectivity).
#     Connectivity is expected to be in the sparse matrix format from getConnectivity() in apiWrappers.py.

#     Sadly, Maya OBJ exporter seems to be much, much faster for some reason :(
#     Already tried speeding this up by buffering to cStringIO, not much improvement.

#     :param filepath: File path to save OBJ file
#     :type filepath: string

#     :param vertices: Vertex positions stored in ndarray
#     :type vertices: double ndarray (N_vertices, 3)

#     :param connectivity: Sparse matrix w/ each row corresponding to a face, each column corresponding to a vertex
#     :type connectivity: int CSR sparse matrix (N_faces, N_verts)
    
#     :return: None
#     :rtype: None
#     """
#     with Timer() as t:
#         with open(filepath, 'w', bufsize) as f:
#             f.write("# Exported by simple OBJ writer mrig/python/geometry/geom_io.writeObj()\n")
#             for v in vertices:
#                 f.write("v %.4f %.4f %.4f\n" % tuple(v))
#             for face in connectivity:
#                 f.write("f" + ((" %d" * len(face.indices)) % tuple(face.indices[face.data.argsort()])) + "\n")
#     print("[mrig/python/geometry/geom_io.writeOBJ] %s: %.3f sec" % (filepath, t.interval_system))

def writeOBJ(filepath, v, vt, f, fvt):
    
    with open(filepath, 'w') as file:
        for i in range(len(v)):
            file.write('v %f %f %f\n' % (v[i][0], v[i][1], v[i][2]))
    
        for i in range(len(vt)):
            file.write('vt %f %f\n' % (vt[i][0], vt[i][1]))
        
        for i in range(len(f)):
            if len(f[i]) == 3:
                file.write('f %d/%d %d/%d %d/%d\n' % (f[i][0] + 1, fvt[i][0] + 1, f[i][1] + 1, fvt[i][1] + 1, f[i][2] + 1, fvt[i][2] + 1))
            elif len(f[i]) == 4:
                file.write('f %d/%d %d/%d %d/%d %d/%d\n' % (f[i][0] + 1, fvt[i][0] + 1, f[i][1] + 1, fvt[i][1] + 1, f[i][2] + 1, fvt[i][2] + 1, f[i][3] + 1, fvt[i][3] + 1))
            #     file.write('f %d/%d %d/%d %d/%d\n' % (f[0][0] + 1, f[0][1] + 1, f[2][0] + 1, f[2][1] + 1, f[3][0] + 1, f[3][1] + 1))
            else:
                raise ValueError("[ERROR] Invalid face in HeadMesh")


def writeOBJ_group(filepath, v_list, vt_list, f_list, fvt_list, geo_list, material_list):
    
    with open(filepath, 'w') as file:
        file.write('mtllib TG_eyesTeeth.mtl\n')
        vNumAccum = 0
        vtNumAccum = 0
        for li in range(len(v_list)):
            v = v_list[li]
            vt = vt_list[li]
            f = f_list[li]
            fvt = fvt_list[li]
            geoName = geo_list[li]
            material = material_list[li]
            file.write('g {}\n'.format(geoName))
            file.write('usemtl {}\n'.format(material))
            for i in range(len(v)):
                file.write('v %f %f %f\n' % (v[i][0], v[i][1], v[i][2]))
        
            for i in range(len(vt)):
                file.write('vt %f %f\n' % (vt[i][0], vt[i][1]))
            
            for i in range(len(f)):
                if len(f[i]) == 3:
                    file.write('f %d/%d %d/%d %d/%d\n' % (f[i][0]+1+vNumAccum, fvt[i][0]+1+vtNumAccum, f[i][1]+1+vNumAccum, fvt[i][1]+1+vtNumAccum, f[i][2]+1+vNumAccum, fvt[i][2]+1+vtNumAccum))
                elif len(f[i]) == 4:
                    file.write('f %d/%d %d/%d %d/%d %d/%d\n' % (f[i][0]+1+vNumAccum, fvt[i][0]+1+vtNumAccum, f[i][1]+1+vNumAccum, fvt[i][1]+1+vtNumAccum, f[i][2]+1+vNumAccum, fvt[i][2]+1+vtNumAccum, f[i][3]+1+vNumAccum, fvt[i][3]+1+vtNumAccum))
                #     file.write('f %d/%d %d/%d %d/%d\n' % (f[0][0] + 1, f[0][1] + 1, f[2][0] + 1, f[2][1] + 1, f[3][0] + 1, f[3][1] + 1))
                else:
                    raise ValueError("[ERROR] Invalid face in HeadMesh")
            
            vNumAccum += len(v)
            vtNumAccum += len(vt)

def readOBJ(filepath, force_triangle=True):
    """
    Super-limited OBJ reader.
    Only supports minimal geometry information (vertex positions, connectivity).
    
    :param filepath: File path to OBJ file
    :type filepath: string

    :return vertices: Vertex positions stored in ndarray
    :type vertices: double ndarray (N_vertices, 3)

    :return triangles: vertex index of faces in ndarray
    :type triangles: int ndarray (N_faces, 3)
    """

    vertices = []
    normals = []
    texcoords = []
    faces = []
    triIdxs = []
    triTexIdxs = []

    material = None
    for line in open(filepath, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertices.append(v)
        elif values[0] == 'vn':
            v = list(map(float, values[1:4]))
            normals.append(v)
        elif values[0] == 'vt':
            texcoords.append(list(map(float, values[1:3])))
        # elif values[0] in ('usemtl', 'usemat'):
        #     material = values[1]
        # elif values[0] == 'mtllib':
        #     mtl = MTL(values[1])
        elif values[0] == 'f':
            face = []
            ftexcoords = []
            norms = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0])-1)
                if len(w) >= 2 and len(w[1]) > 0:
                    ftexcoords.append(int(w[1])-1)
                else:
                    ftexcoords.append(-1)
                if len(w) >= 3 and len(w[2]) > 0:
                    norms.append(int(w[2])-1)
                else:
                    norms.append(-1)
            # faces.append((face, ftexcoords, norms, material))
            if len(face) == 4 and force_triangle:
                # split a quad to two triangles
                tri0 = [face[0],face[1],face[2]]
                tri1 = [face[0],face[2],face[3]]
                triUV0 = [ftexcoords[0],ftexcoords[1],ftexcoords[2]]
                triUV1 = [ftexcoords[0],ftexcoords[2],ftexcoords[3]]
                triIdxs.append(tri0)
                triIdxs.append(tri1)
                triTexIdxs.append(triUV0)
                triTexIdxs.append(triUV1)
            else:
                triIdxs.append(face)
                triTexIdxs.append(ftexcoords)

    return np.array(vertices).astype(np.float32), np.array(texcoords).astype(np.float32), np.array(triIdxs).astype(np.int64), np.array(triTexIdxs).astype(np.int64)
