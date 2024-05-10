import maya.cmds as mc

vs = [1654, 1163, 1435, 1645, 9144, 7896, 8627, 8207, 7612, 7609, 7628, 11273, 4824, 7878, 1581, 7590, 1160, 8421, 11443, 8428, 5002, 11548, 8433, 1979, 134, 11707, 11652, 114, 3194, 11413, 11636, 11501, 11790, 7884, 7828, 8177, 8555, 1616, 11311, 8030, 1141, 1429, 11441, 5188, 11637, 8905, 130, 1984, 2456, 2201, 5053, 1447, 4992, 5257, 1033, 98, 9150, 1179, 1758, 1379, 5070, 11469, 2106, 1972, 2244, 11288, 8650, 9642, 4839, 2826, 2695, 2701, 8411, 8453, 8459, 8377, 1928, 2010, 2004, 1962, 7093, 4994, 11435, 5340, 11544, 11666, 7481, 11523, 11501, 7159, 79, 644, 457, 644, 87, 7093, 6906, 7174, 5197, 4986, 2170, 9275, 5266, 91, 536, 710, 725, 5275, 7162, 6910, 5176, 5188, 5277, 461, 727, 203, 5203, 7176, 8693, 8094, 8065, 8103, 2150, 125, 1728, 138]
vss = ["mark_hi_conformal_uv_tri1:Mesh.vtx[%d]"%i for i in vs]
vss += ["base1:Mesh.vtx[%d]"%i for i in vs]
vss += ["mark_hi_conformal_face_tri_uv:Mesh.vtx[%d]"%i for i in vs]
mc.select(vss, r=1)


kps = [16644, 16888, 16467, 16264, 32244, 32939, 33375, 33654, 33838, 34022, 34312, 34766, 35472, 27816, 27608, 27208, 27440, 28111, 28787, 29177, 29382, 29549, 30288, 30454, 30662, 31056, 31716, 8161,  8177,  8187,  8192,  6515,  7243,  8204,  9163,  9883, 2215,  3886,  4920,  5828,  4801,  3640, 10455, 11353, 12383, 14066, 12653, 11492,  5522,  6025,  7495,  8215,  8935, 10395, 10795,  9555,  8836,  8236,  7636,  6915,  5909,  7384,  8223, 9064, 10537,  8829,  8229,  7629]
vss = ["wrapped_MS2:Mesh.vtx[%d]"%i for i in kps]

mc.select(vss, r=1)

import maya.api.OpenMaya as om2

def getFnMesh(meshName):
    sel = om2.MSelectionList()
    sel.add(meshName)
    return om2.MFnMesh(sel.getDependNode(0))
    
# landmark idxs
sel = om2.MSelectionList()
sel.add("wrapped_MS2:MeshShape")
sel.add("transferred1:MeshShape")
fnMesh0 = om2.MFnMesh(sel.getDependNode(0))
fnMesh1 = om2.MFnMesh(sel.getDependNode(1))
pnts0 = fnMesh0.getPoints()
pnts1 = fnMesh1.getPoints()

import numpy as np
pnts0 = np.array(pnts0)
pnts1 = np.array(pnts1)

def nn(value, array, nbr_neighbors=1):
    return np.argsort(np.array([np.linalg.norm(value-x) for x in array]))[:nbr_neighbors]

kps2 = []
for i,v in enumerate(vtxs):
    print(i)
    kps2.append(nn(pnts0[kp], pnts1)[0])

kps3 = kps2[:]
kps3[27:31] = [137, 133, 129, 124]
kps3[31:36] = [7494, 8429, 115, 1980, 1045]
kps3[36:42] = [11600, 11592, 11587, 11758, 11445, 11548]
kps3[42:48] = [5336, 5139, 5141, 5211, 5093, 4988]
kps3[48:60] = [6911, 6855, 6848, 75, 399, 406, 462, 587, 506, 83, 6955, 7036]
kps3[60:68] = [6918, 7005, 70, 556, 469, 591, 80, 7040]

vss = ["transferred1:Mesh.vtx[%d]"%i for i in kps2]
vss = ["transferred1:Mesh.vtx[%d]"%i for i in kps3]
mc.select(vss, r=1)

vss = ["wrapped_MS2:Mesh.vtx[%d]"%i for i in kps]
mc.select(vss[48:], r=1)

# boundary vertices
def selToIdxs():
    def toList(rangeStr):
        if ":" in rangeStr:
            i,j = rangeStr.split(":")
            return list(range(int(i),int(j)+1))
        return [int(rangeStr)]
    idxs = [x.split("[")[1][:-1] for x in mc.ls(sl=1)]
    idxs = [toList(x) for x in idxs]
    return [i for subIdxs in idxs for i in subIdxs]

# get mesh shape
import scipy.spatial
srcMeshPnts = getFnMesh("wrapped_MS2:MeshShape").getPoints()
tgtMeshKD = scipy.spatial.KDTree(getFnMesh("MeshShape").getPoints())
srcMeshBoundIdxs = [tgtMeshKD.query(srcMeshPnts[i])[1] for i in idxs]

# get facial region (1)
facial_verts = selToIdxs()
len(facial_verts)

# get facial region (2)
srcMeshPnts = getFnMesh("mark_hi_conformal_face_tri_uv:MeshShape").getPoints()
tgtMeshKD = scipy.spatial.KDTree(getFnMesh("mark_hi_conformal_uv_tri:MeshShape").getPoints())
srcFacialRegion2 = [tgtMeshKD.query(srcMeshPnts[i])[1] for i in range(len(srcMeshPnts))]
srcFacialRegion2 = [i for i in srcFacialRegion2 if i < 15375]

def selVerts(name, idxs):
    mc.select(["%s.vtx[%d]"%(name,i) for i in idxs])
selVerts("Mesh", srcFacialRegion2)
#mc.select(["Mesh.vtx[%d]"%i for i in srcMeshBoundIdxs])
#mc.select(["wrapped_MS2:MeshShape.vtx[%d]"%i for i in idxs])
#mc.select(["MeshShape.vtx[%d]"%i for i in srcMeshBoundIdxs], add=1)

tgtFacialRegion = selToIdxs()
for k in kps3:
    if k not in srcFacialRegion:
        print(k)

mc.select(["Mesh.vtx[%d]"%i for i in srcMeshBoundIdxs], add=1)

import json
# json.dump(tgtFacialRegion, open("c:/users/jseo/facialRegion.json","w"))

### get masks
masks = json.load(open("c:/users/jseo/Downloads/bfm_train_mask.json"))
selVerts("wrapped_MS2:Mesh",masks["front_mask"])

### transfer front mask
srcMeshPnts = getFnMesh("wrapped_MS2:MeshShape").getPoints()
tgtMeshKD = scipy.spatial.KDTree(getFnMesh("transferred1:MeshShape").getPoints())
tgtFrontMask = [tgtMeshKD.query(srcMeshPnts[i])[1] for i in masks["front_mask"]]
selVerts("transferred1:Mesh", tgtFrontMask)
mc.select(["Mesh.vtx[%d]"%i for i in srcMeshBoundIdxs], add=1)
# refine selection
tgtFrontMask = selToIdxs()
tgtFacialRegionSet = set(tgtFacialRegion)
tgtFrontMask = [i for i in tgtFrontMask if i in tgtFacialRegionSet]
selVerts("transferred1:Mesh", tgtFrontMask)
# json.dump(tgtFrontMask, open("c:/users/jseo/frontMask.json","w"))

### transfer skin mask
srcSkinVerts = [i for i in range(len(masks["skin_mask"])) if masks["skin_mask"][i] == 1.0]
selVerts("wrapped_MS2:Mesh", srcSkinVerts)
tgtSkinMask = [tgtMeshKD.query(srcMeshPnts[i])[1] for i in srcSkinVerts]
# refine selection
tgtSkinMask = selToIdxs()
tgtSkinMask = [int(i) for i in tgtSkinMask if i in tgtFacialRegionSet]
selVerts("transferred1:Mesh", tgtSkinMask)
# json.dump(tgtSkinMask, open("c:/users/jseo/skinMask.json","w"))

### transfer front face mask
srcFrontFaceMask = list(set(np.array(masks['front_face_buf']).flatten()))
selVerts("wrapped_MS2:Mesh", srcFrontFaceMask)
tgtFrontFaceMask = [tgtMeshKD.query(srcMeshPnts[i])[1] for i in srcFrontFaceMask]
selVerts("transferred1:Mesh", tgtFrontFaceMask)
# refine selection
tgtFrontFaceMask = selToIdxs()
tgtFrontFaceMask = [int(i) for i in tgtFrontFaceMask if i in tgtFacialRegionSet]
selVerts("transferred1:Mesh", tgtFrontFaceMask)
# json.dump(tgtSkinMask, open("c:/users/jseo/FrontFaceMask.json","w"))

# key items: tgtFacialRegion, tgtFrontMask, tgtSkinMask, tgtFrontFaceMask

### create cropped face
selVerts("transferred1:Mesh", tgtFacialRegion)
# convert the selection to faces, duplicate, and triangulate. rename to "facialMesh"
facialMeshPnts = getFnMesh("facialMeshShape").getPoints()
facialMeshVertIdxs = [int(tgtMeshKD.query(p)[1]) for p in facialMeshPnts]
facialMeshDict = {"fullMeshVertIdxs":facialMeshVertIdxs, \
                  "frontMask":tgtFrontMask,\
                  "skinMask":tgtSkinMask,\
                  "frontFaceMask":tgtFrontFaceMask}
json.dump(facialMeshDict, open("c:/users/jseo/Downloads/facialMeshMasks.json","w"))
