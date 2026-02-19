"""
Functions to create meshes for cell mechanotransduction on nanopillars
"""

from typing import Tuple
import pathlib
import numpy as np
import dolfin as d
from mpi4py import MPI
from smart.mesh_tools import implicit_curve, gmsh_to_dolfin, facet_topology, compute_curvature

def assemble_ne_pm(
        ne_mesh: d.Mesh,
        pm_mesh: d.Mesh,
        hEdge: float = 0.5,
        hInnerEdge: float = 0.5,
        hNP: float = 0.3,
        interface_marker: int = 12,
        outer_marker: int = 10,
        inner_vol_tag: int = 2,
        outer_vol_tag: int = 1,
        nanopillars: Tuple[float, float, float] = [0, 0, 0],
        return_curvature: bool = True,
        sym_fraction: float = 0.25,
        use_tmp: bool = False,
        verbose: bool = False,
):

    if isinstance(nanopillars, list):
        nanopillar_rad, nanopillar_height, nanopillar_spacing = nanopillars[:]
        nanopillar_logic = np.all(np.array(nanopillars) != 0)
    elif isinstance(nanopillars, dict):
        nanopillar_rad = nanopillars["rNP"]
        nanopillar_height = nanopillars["hNP"]
        nanopillar_spacing = nanopillars["pNP"]
        nanopillar_logic = np.all(np.array([nanopillar_rad, nanopillar_height, nanopillar_spacing]) != 0)
    else:
        raise ValueError("Format of nanopillar specs not recognized")
    # now load into gmsh and create full mesh
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("ne_and_pm")
    zHalf = np.max(pm_mesh.coordinates()[:,2])/2
    zHalfNE = (np.min(ne_mesh.coordinates()[:,2]) + np.max(pm_mesh.coordinates()[:,2]))/2

    for v in d.vertices(pm_mesh):
        xcur = [v.x(0), v.x(1), v.x(2)]
        gmsh.model.geo.add_point(xcur[0], xcur[1], xcur[2], tag=v.index())
        if xcur[0] < 1e-6 and xcur[1] < 1e-6:
            if xcur[2] > zHalf:
                top_point_PM = v.index()
            else:
                bottom_point_PM = v.index()
    
    vOffset = gmsh.model.geo.get_max_tag(0)+1
    for v in d.vertices(ne_mesh):
        xcur = [v.x(0), v.x(1), v.x(2)]
        gmsh.model.geo.add_point(xcur[0], xcur[1], xcur[2], tag=v.index()+vOffset)
        if xcur[0] < 1e-6 and xcur[1] < 1e-6:
            if xcur[2] > zHalfNE:
                top_point_NE = v.index()+vOffset
            else:
                bottom_point_NE = v.index()+vOffset

    front_edge_PM = []
    back_edge_PM = []
    for f in d.facets(pm_mesh):
        idx = f.entities(0) # vertices
        gmsh.model.geo.add_line(idx[0], idx[1], tag=f.index())
        if sym_fraction == 0.125:
            if np.isclose(f.midpoint().y(), 0.0):
                front_edge_PM.append(f.index())
            elif np.isclose(np.arctan2(f.midpoint().y(),f.midpoint().x()), 2*sym_fraction*np.pi):
                back_edge_PM.append(f.index())
        elif sym_fraction == 0.25:
            if np.isclose(f.midpoint().y(), 0.0):
                front_edge_PM.append(f.index())
            elif np.isclose(f.midpoint().x(), 0.0):
                back_edge_PM.append(f.index())
        else:
            raise ValueError("sym_fraction must be 1/4 or 1/8")
    
    front_edge_NE = []
    back_edge_NE = []
    fOffset = gmsh.model.geo.get_max_tag(1)+1
    for f in d.facets(ne_mesh):
        idx = f.entities(0) # vertices
        gmsh.model.geo.add_line(idx[0]+vOffset, idx[1]+vOffset, 
                                tag=f.index()+fOffset)
        if sym_fraction == 0.125:
            if np.isclose(f.midpoint().y(), 0.0):
                front_edge_NE.append(f.index()+fOffset)
            elif np.isclose(np.arctan2(f.midpoint().y(),f.midpoint().x()), 2*sym_fraction*np.pi):
                back_edge_NE.append(f.index()+fOffset)
        elif sym_fraction == 0.25:
            if np.isclose(f.midpoint().y(), 0.0):
                front_edge_NE.append(f.index()+fOffset)
            elif np.isclose(f.midpoint().x(), 0.0):
                back_edge_NE.append(f.index()+fOffset)
        else:
            raise ValueError("sym_fraction must be 1/4 or 1/8")
    
    symm_axis_top = gmsh.model.geo.add_line(top_point_PM, top_point_NE)
    symm_axis_middle = gmsh.model.geo.add_line(top_point_NE, bottom_point_NE)
    symm_axis_bottom = gmsh.model.geo.add_line(bottom_point_NE, bottom_point_PM)
    
    pm_surf = []
    for c in d.cells(pm_mesh):
        idx = c.entities(1) # edges
        cur_loop = gmsh.model.geo.add_curve_loop([idx[0],idx[1],idx[2]], reorient=True)
        pm_surf.append(gmsh.model.geo.add_plane_surface([cur_loop]))
    
    ne_surf = []
    for c in d.cells(ne_mesh):
        idx = c.entities(1) # edges
        cur_loop = gmsh.model.geo.add_curve_loop(
            [idx[0]+fOffset,idx[1]+fOffset,idx[2]+fOffset], reorient=True)
        ne_surf.append(gmsh.model.geo.add_plane_surface([cur_loop]))

    front_loop = gmsh.model.geo.add_curve_loop(
        [*front_edge_PM, symm_axis_top, *front_edge_NE, symm_axis_bottom], reorient=True)
    front_surf = gmsh.model.geo.add_plane_surface([front_loop])
    back_loop = gmsh.model.geo.add_curve_loop(
        [*back_edge_PM, symm_axis_top, *back_edge_NE, symm_axis_bottom], reorient=True)
    back_surf = gmsh.model.geo.add_plane_surface([back_loop])
    pm_shell = gmsh.model.geo.add_surface_loop([front_surf, back_surf, *pm_surf, *ne_surf])
    pm_vol = gmsh.model.geo.add_volume([pm_shell])

    front_loop_ne = gmsh.model.geo.add_curve_loop(
        [*front_edge_NE, symm_axis_middle], reorient=True)
    front_surf_ne = gmsh.model.geo.add_plane_surface([front_loop_ne])
    back_loop_ne = gmsh.model.geo.add_curve_loop(
        [*back_edge_NE, symm_axis_middle], reorient=True)
    back_surf_ne = gmsh.model.geo.add_plane_surface([back_loop_ne])
    ne_shell = gmsh.model.geo.add_surface_loop([front_surf_ne, back_surf_ne, *ne_surf])
    ne_vol = gmsh.model.geo.add_volume([ne_shell])

    gmsh.model.geo.synchronize()

    gmsh.model.add_physical_group(2, pm_surf, tag=outer_marker)
    gmsh.model.add_physical_group(2, ne_surf, tag=interface_marker)
    gmsh.model.add_physical_group(3, [pm_vol], tag=outer_vol_tag)
    gmsh.model.add_physical_group(3, [ne_vol], tag=inner_vol_tag)

    zMid = np.mean(ne_mesh.coordinates()[:,2])
    rValsOuter = np.sqrt(pm_mesh.coordinates()[:,0]**2 + pm_mesh.coordinates()[:,1]**2)
    zValsOuter = pm_mesh.coordinates()[:,2]
    rValsInner = np.sqrt(ne_mesh.coordinates()[:,0]**2 + ne_mesh.coordinates()[:,1]**2)
    zValsInner = ne_mesh.coordinates()[:,2]
    if nanopillar_logic:
        num_pillars = 2*np.ceil(np.max(rValsOuter)/nanopillar_spacing) + 1
        rMax = nanopillar_spacing * np.ceil(np.max(rValsOuter)/nanopillar_spacing)
        test_coords = np.linspace(-rMax, rMax, int(num_pillars))
        xTest, yTest = np.meshgrid(test_coords, test_coords)
        xTest = np.reshape(xTest, (len(xTest)**2))
        yTest = np.reshape(yTest, (len(yTest)**2))
    else:
        xTest = []
        yTest = []

    ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
    RInnerVec = np.sqrt(rValsInner**2 + (zValsInner - zMid) ** 2)
    maxOuterDim = max(ROuterVec)
    maxInnerDim = max(RInnerVec)
    
    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM and hInnerEdge at the inner membrane
        # between these, the value is interpolated based on the relative distance
        # between the two membranes.
        # Inside the inner shape, the value is interpolated between hInnerEdge
        # and lc3, where lc3 = max(hInnerEdge, 0.2*maxInnerDim)
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*maxOuterDim in the center
        # hNP is also considered here (mesh resolution at nanopillars),
        # weighted by how close the point is to the nanopillar surface
        rCur = np.sqrt(x**2 + y**2)
        RCur = np.sqrt(rCur**2 + (z - zMid) ** 2)
        dist_to_outer = np.min(np.sqrt((rCur - rValsOuter) ** 2 + (z - zValsOuter) ** 2))
        if nanopillar_logic:
            xy_NP_dist = min(np.sqrt((x-xTest)**2 + (y-yTest)**2))
            if xy_NP_dist < nanopillar_rad:
                xy_NP_dist = 0
            else:
                xy_NP_dist -= nanopillar_rad
            z_NP_dist = max([0, z - nanopillar_height])
            dist_to_NP = np.sqrt(xy_NP_dist**2 + z_NP_dist**2)
            hNPWeight = np.exp(-dist_to_NP / 2.5)
        else:
            hNPWeight = 0
        
        if len(rValsInner) == 0:
            lc3 = 0.2 * maxOuterDim
            dist_to_inner = RCur
            in_outer = True
        else:
            inner_dist = np.sqrt((rCur - rValsInner) ** 2 + (z - zValsInner) ** 2)
            dist_to_inner = min(inner_dist)
            inner_idx = np.argmin(inner_dist)
            inner_rad = RInnerVec[inner_idx]
            R_rel_inner = RCur / inner_rad
            lc3 = max(hInnerEdge, 0.2 * maxInnerDim)
            in_outer = R_rel_inner > 1
        lc1 = (1-hNPWeight)*hEdge + hNPWeight*hNP
        lc2 = (1-hNPWeight)*hInnerEdge + hNPWeight*hNP
        lc3 = (1-hNPWeight)*lc3 + hNPWeight*hNP
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (dist_to_outer) / (dist_to_inner + dist_to_outer)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        return lcTest

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(3)
    rank = MPI.COMM_WORLD.rank
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_3dcell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"tmp_3dcell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "3dcell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # assign substrate markers
    substrate_markers = d.MeshFunction("size_t", dmesh, 2)
    if isinstance(nanopillars, dict):
        contactRad = nanopillars["contactRad"]
        edgeThresh = hEdge/100
    else:
        contactRad = np.max(np.sqrt(dmesh.coordinates()[:,0]**2 + dmesh.coordinates()[:,1]**2))
        edgeThresh = hEdge/10
    xCurv = 0.2
    xSteric = 0.05
    zOffset = nanopillar_height
    for f in d.facets(dmesh):
        topology, cellIndices = facet_topology(f, mf3)
        if topology == "boundary":
            # test if it is on the outward surface (not substrate)
            rCur = np.sqrt(f.midpoint().x()**2 + f.midpoint().y()**2)
            zCur = f.midpoint().z()
            thetaCur = np.arctan2(f.midpoint().y(), f.midpoint().x())
            if zCur == 0:
                substrate_markers.set_value(f.index(), outer_marker)
            elif not(sym_fraction != 1 and (
                np.isclose(thetaCur, 0.0, atol=1e-4) or np.isclose(thetaCur, 2*np.pi*sym_fraction, atol=1e-4))):
                if nanopillar_logic:
                    if rCur < contactRad - edgeThresh and zCur < zOffset + hEdge/100:
                        substrate_markers.set_value(f.index(), outer_marker)

    curv_markers = compute_curvature(dmesh, mf2, mf3, [10,12], [1,2])
    if return_curvature:
        # manually set curvatures on the substrate (these are known)
        for f in d.facets(dmesh):
            if substrate_markers.array()[f.index()] == outer_marker:
                for v in d.vertices(f):
                    # explictly set curvature for membrane on nanopillars (or set to zero for no nanopillars)
                    zVal = v.midpoint().z()
                    if zVal >= nanopillar_height:
                        curv_markers.set_value(v.index(), 0.0)
                    elif zVal > nanopillar_height - xSteric: # curved from cyl to top
                        cosCur = (zVal - (nanopillar_height-xSteric))/xSteric
                        sinCur = np.sqrt(1 - cosCur**2)
                        cm = 1/xSteric
                        cp = sinCur / (nanopillar_rad + xSteric*sinCur)
                        rVal = nanopillar_rad + xSteric*sinCur
                        cpAlt = (rVal - nanopillar_rad) / (rVal*xSteric)
                        curv_markers.set_value(v.index(), -(cm+cp)/2) 
                    elif zVal > xCurv: # cylinder
                        curv_markers.set_value(v.index(), -0.5/nanopillar_rad)
                    elif zVal > 0: # curved to substrate
                        cosCur = (xCurv - zVal)/xCurv
                        sinCur = np.sqrt(1 - cosCur**2)
                        cm = 1/xCurv
                        cp = -sinCur / (nanopillar_rad + xSteric + xCurv*(1-sinCur))
                        rVal = nanopillar_rad + xSteric + xCurv - xCurv*sinCur
                        cpAlt = (rVal - (nanopillar_rad + xSteric + xCurv)) / (rVal*xCurv)
                        curv_markers.set_value(v.index(), (cm+cp)/2)
                    else: # substrate
                        curv_markers.set_value(v.index(), 0.0)
                    
        return (dmesh, mf2, mf3, substrate_markers, curv_markers)
    else:
        return (dmesh, mf2, mf3, substrate_markers)

def create_3dcell(
    contactRad: float = 10.0,
    hEdge: float = 0,
    hInnerEdge: float = 0,
    hNP: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_vol_tag: int = 2,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
    return_curvature: bool = False,
    nanopillars: Tuple[float, float, float] = [0, 0, 0],
    use_tmp: bool = False,
    sym_fraction: float = 1.0,
    nuc_compression: float = 0.0,
    no_nuc: bool = False,
    no_np: bool = False,
    shape_coords: list = [],
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction, d.MeshFunction, 
           d.MeshFunction, d.Function, d.Function]:
    """
    Creates a 3d cell mesh.
    The inner contour representing the nucleus is defined
    implicitly through innerExpr, which is rotated about the z axis
    to form an axisymmetric shape (e.g. unit sphere centered at (0, 2) 
    defined by innerExpr = "r**2 + (z-2)**2 - 1")
    The outer cell contour is also defined by an expression of r and z.
    It is assumed that substrate is present at z = 0, so if the curve extends
    below z = 0 , there is a sharp cutoff.

    Args:
        outerExpr: String implicitly defining an r-z curve for the outer surface
        innerExpr: String implicitly defining an r-z curve for the inner surface
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
            of the inner compartment
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on the outer ellipsoid with
        inner_vol_tag: The value to mark the inner ellipsoidal volume with
        outer_vol_tag: The value to mark the outer ellipsoidal volume with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
        return_curvature: If true, return curvatures as a vertex mesh function
        nanopillars: tuple storing [nanopillar_rad, nanopillar_height, nanopillar_spacing],
                     all in microns
        use_tmp: logical variable, indicating whether to use tmp directory for gmsh operations
        sym_fraction: 0.5 for half-cell geometry, 1/8 for one-eigth cell geometry, etc.
        nuc_compression: nuclear indentation in microns
                         if negative, this results in a vertical shift of the nucleus
    Returns:
        Tuple (dmesh, mf2, mf3, substrate_markers, u_nuc, a_nuc)
    Or, if return_curvature = True, Returns:
        Tuple (dmesh, mf2, mf3, substrate_markers, curv_markers, u_nuc, a_nuc)
    Where
        dmesh: dolfin mesh for whole geometry
        mf2: mesh facet markers (surface mesh function)
        mf3: mesh cell markers (volume mesh function)
        substrate_markers: marks location of substrate (incl nanopillars) (surface mesh function)
        curv_markers: 0D mesh function (points) defining curvature at each vertex
        u_nuc: dolfin function defining nuclear deformations over reference ellipsoid
        a_nuc: dolfin function defining nuclear envelope stretch over reference ellipsoid
    """
    import gmsh
    
    nanopillar_rad, nanopillar_height, nanopillar_spacing = nanopillars[:]
    zOffset = nanopillar_height
    xSteric = 0.05
    xCurv = 0.2
    if len(shape_coords)==6:
        rValsOuter, zValsOuter, rValsInner, zValsInner, innerParam, u_nuc = shape_coords
    else:
        rValsOuter, zValsOuter, rValsInner, zValsInner, innerParam, u_nuc, rScale, outParam = get_shape_coords(
                                            contactRad, nanopillars, nuc_compression, no_nuc=no_nuc)
    
    if return_curvature:
        rValsOuterClosed = np.concatenate((rValsOuter, -rValsOuter[::-1]))
        zValsOuterClosed = np.concatenate((zValsOuter,  zValsOuter[::-1]))
        curvFcnOuter = compute_curvature_1D(rValsOuterClosed, zValsOuterClosed, 
                                    curvRes=0.1, incl_parallel=True)
    if not len(rValsInner) == 0:
        aInner, bInner, r0Inner, z0Inner = innerParam[:]
        zMid = np.mean(zValsInner)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        RInnerVec = np.sqrt(rValsInner**2 + (zValsInner - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
        maxInnerDim = max(RInnerVec)
    else:
        zMid = np.mean(zValsOuter)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
    
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * maxOuterDim
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * maxOuterDim if len(rValsInner) == 0 else 0.2 * maxInnerDim
    if np.isclose(hNP, 0):
        hNP = 0.1 * maxOuterDim
    # Create the two axisymmetric body mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("3dcell")

    # first add outer body and revolve
    dtheta = hEdge/max(rValsOuter)
    cell_plane_tag = []
    edge_surf_list = []
    top_point = gmsh.model.occ.add_point(0.0, 0.0, zValsOuter[0])
    # rotate shape 2*pi in the case of no theta dependence
    outer_tag_list = []
    line_tag_list = []
    for i in range(len(rValsOuter)):
        if i == 0:
            outer_tag_list.append(top_point)
        else:
            cur_tag = gmsh.model.occ.add_point(rValsOuter[i], 0.0, zValsOuter[i])
            line_tag = gmsh.model.occ.add_line(cur_tag, outer_tag_list[-1])
            line_tag_list.append(line_tag)
            outer_tag_list.append(cur_tag)
    outer_spline = gmsh.model.occ.add_spline(outer_tag_list)
    origin_tag = gmsh.model.occ.add_point(0, 0, 0)
    symm_axis_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[0])
    bottom_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[-1])
    outer_loop_tag = gmsh.model.occ.add_curve_loop(
        [outer_spline, symm_axis_tag, bottom_tag]
    )
    cell_plane_tag = gmsh.model.occ.add_plane_surface([outer_loop_tag])
    outer_shape = gmsh.model.occ.revolve([(2, cell_plane_tag)], 0, 0, 0, 0, 0, 1, 2*np.pi*sym_fraction)
    
    if np.all(np.array(nanopillars) != 0) and not no_np:
        zero_idx = np.nonzero(zValsOuter <= 0.0)
        num_pillars = 2*np.ceil(rValsOuter[zero_idx]/nanopillar_spacing) + 1
        rMax = nanopillar_spacing * np.ceil(rValsOuter[zero_idx]/nanopillar_spacing)
        test_coords = np.linspace(-rMax[0], rMax[0], int(num_pillars[0]))
        xTest, yTest = np.meshgrid(test_coords, test_coords)
        rTest = np.sqrt(xTest**2 + yTest**2)
        thetaTest = np.arctan2(yTest, xTest)
        if sym_fraction==1:
            keep_logic = rTest <= rValsOuter[zero_idx]+nanopillar_rad+xCurv+xSteric
        else:
            keep_logic1 = rTest <= rValsOuter[zero_idx]+nanopillar_rad+xCurv+xSteric
            keep_logic2 = np.logical_and(thetaTest < (2*np.pi*sym_fraction + np.pi/100), thetaTest > (0.0-np.pi/100))
            keep_logic = np.logical_and(keep_logic1, keep_logic2)
        xTest, yTest = xTest[keep_logic], yTest[keep_logic]
        for i in range(len(xTest)):
            # add points and then rotate?
            np_tag_list = []
            np_line_list = []
            zCur = nanopillar_height + xSteric
            dtheta = np.pi/8
            thetaCurv1 = np.arange(np.pi/2, 0, -dtheta)
            xCurv1 = nanopillar_rad + xSteric*np.cos(thetaCurv1)
            zCurv1 = nanopillar_height + xSteric*(np.sin(thetaCurv1)-1)
            thetaCurv2 = np.arange(np.pi, 3*np.pi/2, dtheta)
            xCurv2 = nanopillar_rad + xSteric + xCurv*(1+np.cos(thetaCurv2))
            zCurv2 = xCurv*(1+np.sin(thetaCurv2))
            xCurVec = xTest[i] + np.concatenate((np.array([0]), xCurv1, 
                                                np.array([nanopillar_rad+xSteric]), xCurv2, 
                                                np.array([nanopillar_rad+xSteric+xCurv])))
            yCurVec = yTest[i] * np.ones([len(xCurVec),1])
            zCurVec = np.concatenate((np.array([nanopillar_height]), zCurv1, 
                                     np.array([nanopillar_height-xSteric]), zCurv2, 
                                     np.array([0])))
            for j in range(len(zCurVec)):
                cur_tag = gmsh.model.occ.add_point(xCurVec[j], yCurVec[j], zCurVec[j])
                if j > 0:
                    cur_line = gmsh.model.occ.add_line(cur_tag, np_tag_list[-1])
                    np_line_list.append(cur_line)
                np_tag_list.append(cur_tag)

            # np_spline_tag = gmsh.model.occ.add_spline(np_tag_list)
            origin_np_tag = gmsh.model.occ.add_point(xTest[i], yTest[i], 0)
            bottom_np_tag = gmsh.model.occ.add_line(origin_np_tag, np_tag_list[-1])
            symm_np_tag = gmsh.model.occ.add_line(np_tag_list[0], origin_np_tag)
            np_loop_tag = gmsh.model.occ.add_curve_loop([*np_line_list, bottom_np_tag, symm_np_tag])
            np_plane_tag = gmsh.model.occ.add_plane_surface([np_loop_tag])
            np_shape = gmsh.model.occ.revolve([(2, np_plane_tag)], xTest[i], yTest[i], 0,
                                              0, 0, 1, 2 * np.pi)
            np_shape_tags = []
            for j in range(len(np_shape)):
                if np_shape[j][0] == 3:  # pull out tags associated with 3d objects
                    np_shape_tags.append(np_shape[j][1])
            assert len(np_shape_tags) == 1  # should be just one 3D body from the full revolution
            (outer_shape, outer_shape_map) = gmsh.model.occ.cut(outer_shape, [(3, np_shape_tags[0])])
            outer_shape_list = []
            for j in range(len(outer_shape_map)):
                if outer_shape_map[j]!=[]:
                    outer_shape_list.append(outer_shape_map[j][0])
            outer_shape = outer_shape_list

    outer_shape_tags = []
    for i in range(len(outer_shape)):
        if outer_shape[i][0] == 3:  # pull out tags associated with 3d objects
            outer_shape_tags.append(outer_shape[i][1])
    assert len(outer_shape_tags) == 1  # should be just one 3D body from the full revolution

    if len(rValsInner) == 0:
        # No inner shape in this case
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(3, outer_shape_tags, tag=outer_vol_tag)
        facets = gmsh.model.getBoundary([(3, outer_shape_tags[0])])
        facet_tags = []
        for i in range(len(facets)):
            facet_tags.append(facets[i][1])
        gmsh.model.add_physical_group(2, facet_tags, tag=outer_marker)
    else:
        if np.all(np.array(nanopillars) != 0) and not no_np:
            xMax = np.ceil(aInner / nanopillar_spacing) * nanopillar_spacing
            xNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
            yNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
            xNP, yNP = np.meshgrid(xNP, yNP)
            xNP = xNP.flatten()
            yNP = yNP.flatten()
            inner_spline_list = []
            # line_list_list = []
            edge_surf_list = []
            # cut_vol_list = []
            # first_points_list = []
            if u_nuc == []:
                u_top, u_bottom = 0, 0 #[0.,0.,0.], [0.,0.,0.]
            else:
                u_top = 0.0 #u_nuc(0.0, 0.0, zValsInner[0]-z0Inner)
                u_bottom = calc_def_NP(0.0, 0.0, zValsInner[-1], aInner, bInner, z0Inner, xNP, yNP, nanopillars)
            top_point = gmsh.model.occ.add_point(0.0, 0.0, zValsInner[0]+u_top)
            bottom_point = gmsh.model.occ.add_point(0.0, 0.0, zValsInner[-1]+u_bottom)
            num_theta = np.ceil(321*sym_fraction)
            thetaVecInner = np.linspace(0, 2*np.pi*sym_fraction, int(num_theta))
            sValsInner = np.zeros_like(rValsInner)
            for i in range(1,len(rValsInner)): # define arc length
                sValsInner[i] = sValsInner[i-1] + np.sqrt((rValsInner[i]-rValsInner[i-1])**2 + 
                                                          (zValsInner[i]-zValsInner[i-1])**2)
            for j in range(len(thetaVecInner)):
                thetaCur = thetaVecInner[j]
                if j == (len(thetaVecInner)-1) and sym_fraction == 1:
                    inner_spline_list.append(inner_spline_list[0])
                    # first_points_list.append(first_points_list[0])
                    # line_list_list(line_list_list[0])
                else:
                    inner_tag_list = []
                    # Compute r,z profile for current value of z, see compute_profile function for rules governing spacing
                    rValsCur, zValsCur = compute_profile(thetaCur, rValsInner, zValsInner, sValsInner, 
                                                         nanopillars, nuc_compression, innerParam, xTest, yTest)
                    for i in range(len(rValsCur)):
                        if i == 0:
                            inner_tag_list.append(top_point)
                        elif i == len(rValsCur)-1:
                            inner_tag_list.append(bottom_point)
                        else:
                            xCur = rValsCur[i]*np.cos(thetaCur)
                            yCur = rValsCur[i]*np.sin(thetaCur)
                            zCur = zValsCur[i]
                            cur_tag = gmsh.model.occ.add_point(xCur, yCur, zCur)
                            inner_tag_list.append(cur_tag)
                    inner_spline_list.append(gmsh.model.occ.add_bspline(inner_tag_list, degree=1))
                if j > 0:
                    edge_loop_tag = gmsh.model.occ.add_curve_loop(
                        [inner_spline_list[j], inner_spline_list[j-1]])
                    cur_surf = gmsh.model.occ.add_bspline_filling(edge_loop_tag, type="Coons")
                    edge_surf_list.append(cur_surf)
            
            if sym_fraction < 1:
                symm_line = gmsh.model.occ.add_line(bottom_point, top_point)
                front_loop = gmsh.model.occ.add_curve_loop([symm_line, inner_spline_list[0]])
                # front_loop = gmsh.model.occ.add_curve_loop([symm_line, *line_list_list[0]])
                front_surf = gmsh.model.occ.add_plane_surface([front_loop])
                back_loop = gmsh.model.occ.add_curve_loop([symm_line, inner_spline_list[-1]])
                # back_loop = gmsh.model.occ.add_curve_loop([symm_line, *line_list_list[-1]])
                back_surf = gmsh.model.occ.add_plane_surface([back_loop])
                inner_surf_loop = gmsh.model.occ.add_surface_loop([front_surf, back_surf, *edge_surf_list])
                inner_shape = gmsh.model.occ.add_volume([inner_surf_loop])
                inner_shape = [(3, inner_shape)]
            else:
                # now define total inner shape from edge_segments and edge_surf_list    
                inner_surf_loop = gmsh.model.occ.add_surface_loop(edge_surf_list)
                inner_shape = gmsh.model.occ.add_volume([inner_surf_loop])
                inner_shape = [(3, inner_shape)]
        else:
            # Inner shape is just a spheroid
            inner_tag_list = []
            for i in range(len(rValsInner)):
                cur_tag = gmsh.model.occ.add_point(rValsInner[i], 0, zValsInner[i])
                inner_tag_list.append(cur_tag)
            inner_spline_tag = gmsh.model.occ.add_spline(inner_tag_list)
            symm_inner_tag = gmsh.model.occ.add_line(inner_tag_list[0], inner_tag_list[-1])
            inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, symm_inner_tag])
            inner_plane_tag = gmsh.model.occ.add_plane_surface([inner_loop_tag])
            inner_shape = gmsh.model.occ.revolve([(2, inner_plane_tag)], 
                                                0, 0, 0, 0, 0, 1, 2 * np.pi * sym_fraction)

        inner_shape_tags = []
        for i in range(len(inner_shape)):
            if inner_shape[i][0] == 3:  # pull out tags associated with 3d objects
                inner_shape_tags.append(inner_shape[i][1])
        assert len(inner_shape_tags) == 1  # should be just one 3D body from the full revolution

        # Create interface between 2 objects
        two_shapes, (outer_shape_map, inner_shape_map) = gmsh.model.occ.fragment(
            [(3, outer_shape_tags[0])], [(3, inner_shape_tags[0])]
        )
        gmsh.model.occ.synchronize()

        # Get the outer boundary
        outer_shell = gmsh.model.getBoundary(two_shapes, oriented=False)
        outer_shell_tags = []
        for i in range(len(outer_shell)):
            outer_shell_tags.append(outer_shell[i][1])
        # assert (
        #     len(outer_shell) == 2
        # )  # 2 boundaries because of bottom surface at z = 0, both belong to PM
        # Get the inner boundary
        inner_shell = gmsh.model.getBoundary(inner_shape_map, oriented=False)
        inner_shell_tags = []
        for i in range(len(inner_shell)):
            inner_shell_tags.append(inner_shell[i][1])
        # assert len(inner_shell) == 1
        # Add physical markers for facets
        gmsh.model.add_physical_group(
            outer_shell[0][0], outer_shell_tags, tag=outer_marker
        )
        gmsh.model.add_physical_group(inner_shell[0][0], inner_shell_tags, tag=interface_marker)
        # if nuc_compression > 0:
        #     for i in range(len(inner_shell_tags)):
        #         gmsh.model.mesh.setSmoothing(2, inner_shell_tags[i], 2)

        # Physical markers for
        all_volumes = [tag[1] for tag in outer_shape_map]
        inner_volume = [tag[1] for tag in inner_shape_map]
        outer_volume = []
        for vol in all_volumes:
            if vol not in inner_volume:
                outer_volume.append(vol)
        gmsh.model.add_physical_group(3, outer_volume, tag=outer_vol_tag)
        gmsh.model.add_physical_group(3, inner_volume, tag=inner_vol_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM and hInnerEdge at the inner membrane
        # between these, the value is interpolated based on the relative distance
        # between the two membranes.
        # Inside the inner shape, the value is interpolated between hInnerEdge
        # and lc3, where lc3 = max(hInnerEdge, 0.2*maxInnerDim)
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*maxOuterDim in the center
        # hNP is also considered here (mesh resolution at nanopillars),
        # weighted by how close the point is to the nanopillar surface
        rCur = np.sqrt(x**2 + y**2)
        RCur = np.sqrt(rCur**2 + (z - zMid) ** 2)
        outer_dist = np.sqrt((rCur - rValsOuter) ** 2 + (z - zValsOuter) ** 2)
        np.append(outer_dist, z)  # include the distance from the substrate
        if np.all(np.array(nanopillars) != 0) and not no_np:
            xy_NP_dist = min(np.sqrt((x-xTest)**2 + (y-yTest)**2))
            if xy_NP_dist < nanopillar_rad:
                xy_NP_dist = 0
            else:
                xy_NP_dist -= nanopillar_rad
            z_NP_dist = max([0, z - nanopillar_height])
            dist_to_NP = np.sqrt(xy_NP_dist**2 + z_NP_dist**2)
            hNPWeight = np.exp(-dist_to_NP / 2.5)
        else:
            hNPWeight = 0
        dist_to_outer = min(outer_dist)
        if len(rValsInner) == 0:
            lc3 = 0.2 * maxOuterDim
            dist_to_inner = RCur
            in_outer = True
        else:
            inner_dist = np.sqrt((rCur - rValsInner) ** 2 + (z - zValsInner) ** 2)
            dist_to_inner = min(inner_dist)
            inner_idx = np.argmin(inner_dist)
            inner_rad = RInnerVec[inner_idx]
            R_rel_inner = RCur / inner_rad
            lc3 = max(hInnerEdge, 0.2 * maxInnerDim)
            in_outer = R_rel_inner > 1
        lc1 = (1-hNPWeight)*hEdge + hNPWeight*hNP
        lc2 = (1-hNPWeight)*hInnerEdge + hNPWeight*hNP
        lc3 = (1-hNPWeight)*lc3 + hNPWeight*hNP
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (dist_to_outer) / (dist_to_inner + dist_to_outer)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        return lcTest

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(3)
    rank = MPI.COMM_WORLD.rank
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_3dcell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"tmp_3dcell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "3dcell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()
    
    if nuc_compression > 0:
        a_nuc = compute_stretch(u_nuc, aInner, bInner, nanopillars, z0Inner)
        d.ALE.move(u_nuc.function_space().mesh(), u_nuc)
    else:
        a_nuc = []

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags) and substrate_markers (2d)
    substrate_markers = d.MeshFunction("size_t", dmesh, 2)
    for f in d.facets(dmesh):
        topology, cellIndices = facet_topology(f, mf3)
        if topology == "boundary":
            # test if it is on the outward surface (not substrate)
            rCur = np.sqrt(f.midpoint().x()**2 + f.midpoint().y()**2)
            zCur = f.midpoint().z()
            thetaCur = np.arctan2(f.midpoint().y(), f.midpoint().x())
            if zCur == 0:
                mf2.set_value(f.index(), outer_marker)
                substrate_markers.set_value(f.index(), outer_marker)
            elif sym_fraction != 1 and (
                np.isclose(thetaCur, 0.0, atol=1e-4) or np.isclose(thetaCur, 2*np.pi*sym_fraction, atol=1e-4)):
                # then a no flux surface
                mf2.set_value(f.index(), 0)
            else: #then either on a nanopillar or outer surface
                mf2.set_value(f.index(), outer_marker)
                if np.all(np.array(nanopillars) != 0) and not no_np:
                    if rCur < contactRad - hEdge/100 and zCur < zOffset + hEdge/100:
                        substrate_markers.set_value(f.index(), outer_marker)

    if return_curvature:
        curv_markers = d.MeshFunction("double", dmesh, 0)
        # first set curvatures on substrate
        for f in d.facets(dmesh):
            if substrate_markers.array()[f.index()] == outer_marker:
                for v in d.vertices(f):
                    # explictly set curvature for membrane on nanopillars (or set to zero for no nanopillars)
                    zVal = v.midpoint().z()
                    if zVal >= nanopillar_height:
                        curv_markers.set_value(v.index(), 0.0)
                    elif zVal > nanopillar_height - xSteric: # curved from cyl to top
                        cosCur = (zVal - (nanopillar_height-xSteric))/xSteric
                        sinCur = np.sqrt(1 - cosCur**2)
                        cm = 1/xSteric
                        cp = sinCur / (nanopillar_rad + xSteric*sinCur)
                        rVal = nanopillar_rad + xSteric*sinCur
                        cpAlt = (rVal - nanopillar_rad) / (rVal*xSteric)
                        curv_markers.set_value(v.index(), -(cm+cp)/2) 
                    elif zVal > xCurv: # cylinder
                        curv_markers.set_value(v.index(), -0.5/nanopillar_rad)
                    elif zVal > 0: # curved to substrate
                        cosCur = (xCurv - zVal)/xCurv
                        sinCur = np.sqrt(1 - cosCur**2)
                        cm = 1/xCurv
                        cp = -sinCur / (nanopillar_rad + xSteric + xCurv*(1-sinCur))
                        rVal = nanopillar_rad + xSteric + xCurv - xCurv*sinCur
                        cpAlt = (rVal - (nanopillar_rad + xSteric + xCurv)) / (rVal*xCurv)
                        curv_markers.set_value(v.index(), (cm+cp)/2)
                    else: # substrate
                        curv_markers.set_value(v.index(), 0.0)
        
        # curvature on free PM (not on substrate) and NM
        for f in d.facets(dmesh):
            if (substrate_markers.array()[f.index()] == 0 and 
                mf2.array()[f.index()] == outer_marker):
                # set curvature at other boundaries (map from 2d mesh case)
                for v in d.vertices(f):
                    rVal = np.sqrt(v.midpoint().x()**2 + v.midpoint().y()**2)
                    zVal = v.midpoint().z()
                    curv_markers.set_value(v.index(), curvFcnOuter(rVal, 0, zVal))
            elif mf2.array()[f.index()] == interface_marker:
                for v in d.vertices(f):
                    xVal = v.midpoint().x()
                    yVal = v.midpoint().y()
                    rVal = np.sqrt(xVal**2 + yVal**2)
                    zVal = v.midpoint().z()
                    if nuc_compression > 0:
                        curv_val = calc_curv_NP(xVal,yVal,zVal,aInner,bInner,z0Inner,xNP,yNP,nanopillars)
                        curv_markers.set_value(v.index(), curv_val)
                    else:
                        curv_val = compute_curvature_ellipse_alt(np.array([rVal]), np.array([zVal]), 
                                                                aInner, bInner, 
                                                                r0Inner, z0Inner, incl_parallel=True)
                        curv_markers.set_value(v.index(), curv_val[0])
        return (dmesh, mf2, mf3, substrate_markers, curv_markers, u_nuc, a_nuc)
    else:
        return (dmesh, mf2, mf3, substrate_markers, u_nuc, a_nuc)
    
def create_substrate_new(
    LBox: float = 0.6,
    TBox: float = 0.6,
    hEdge: float = 0,
    x_gap: float = 0.4,
    y_gap: float = 0.4,
    z_gap: float = 0.4, 
    outer_marker: int = 10,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
    nanopillars: Tuple[float, float, float] = [0.05, 0.3, 0.1],
    use_tmp: bool = False,
    contact_rad: float = np.inf,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh of the substrate.
    """
    import gmsh
    nanopillar_rad = nanopillars[0]
    nanopillar_height = nanopillars[1]
    nanopillar_spacing = nanopillars[2]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("3dcell")

    # first add outer body and revolve
    outer_shape = gmsh.model.occ.add_box(x_gap, y_gap, z_gap, LBox, LBox, TBox)
    outer_shape = [(3, outer_shape)]
    if np.all(np.array(nanopillars) != 0):
        num_pillars = 2*np.floor(LBox/(2*nanopillar_spacing)) + 1
        rMax = nanopillar_spacing * np.floor(LBox/(2*nanopillar_spacing))
        test_coords = np.linspace(-rMax, rMax, int(num_pillars))
        xTest, yTest = np.meshgrid(test_coords, test_coords)
        xTest = np.reshape(xTest, (len(xTest)**2))
        yTest = np.reshape(yTest, (len(yTest)**2))
        for i in range(len(xTest)):
            rTest = np.sqrt(xTest[i]**2 + yTest[i]**2)
            xCurv = 0.2
            xSteric = 0.05
            keep_logic1 = rTest <= contact_rad-nanopillar_rad-xCurv-xSteric
            keep_logic2 = rTest > contact_rad+nanopillar_rad
            keep_logic = np.logical_or(keep_logic1, keep_logic2)
            if keep_logic:
                np_shape = gmsh.model.occ.add_cylinder(xTest[i]+x_gap, yTest[i]+y_gap, z_gap, -nanopillar_height, 0, 0, nanopillar_rad)
                (outer_shape, outer_shape_map) = gmsh.model.occ.fuse(outer_shape, [(3, np_shape)])
                outer_shape_list = []

    outer_shape_tags = []
    for i in range(len(outer_shape)):
        if outer_shape[i][0] == 3:  # pull out tags associated with 3d objects
            outer_shape_tags.append(outer_shape[i][1])
    # assert len(outer_shape_tags) == 1  # should be just one 3D body from the full revolution
    gmsh.model.occ.synchronize()
    gmsh.model.add_physical_group(3, outer_shape_tags, tag=outer_vol_tag)
    facets = gmsh.model.getBoundary([(3, outer_shape_tags[0])])
    facet_tags = []
    for i in range(len(facets)):
        facet_tags.append(facets[i][1])
    gmsh.model.add_physical_group(2, facet_tags, tag=outer_marker)

    gmsh.model.mesh.generate(3)
    rank = MPI.COMM_WORLD.rank
    
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_3dcell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"/repo/examples/example5")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "substrate.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    return dmesh, mf2, mf3
    
def NE_mesh(
    rRad: float = 6.625,
    zRad: float = 3.0,
    thickness: list = [0.05,0.05],
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_vol_tag: int = 2,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
    use_tmp: bool = False,
    sym_fraction: float = 1.0,
    NE_layers: int = 1,
    hTop: float = 0.0,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a 3d cell mesh.
    The inner contour representing the nucleus is defined
    implicitly through innerExpr, which is rotated about the z axis
    to form an axisymmetric shape (e.g. unit sphere centered at (0, 2) 
    defined by innerExpr = "r**2 + (z-2)**2 - 1")
    The outer cell contour is also defined by an expression of r and z.
    It is assumed that substrate is present at z = 0, so if the curve extends
    below z = 0 , there is a sharp cutoff.

    Args:
        outerExpr: String implicitly defining an r-z curve for the outer surface
        innerExpr: String implicitly defining an r-z curve for the inner surface
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
            of the inner compartment
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on the outer ellipsoid with
        inner_vol_tag: The value to mark the inner ellipsoidal volume with
        outer_vol_tag: The value to mark the outer ellipsoidal volume with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
        return_curvature: If true, return curvatures as a vertex mesh function
        nanopillars: tuple storing [nanopillar_rad, nanopillar_height, nanopillar_spacing],
                     all in microns
        use_tmp: logical variable, indicating whether to use tmp directory for gmsh operations
        sym_fraction: 0.5 for half-cell geometry, 1/8 for one-eigth cell geometry, etc.
        nuc_compression: nuclear indentation in microns
                         if negative, this results in a vertical shift of the nucleus
    Returns:
        Tuple (dmesh, mf2, mf3, substrate_markers, u_nuc, a_nuc)
    Or, if return_curvature = True, Returns:
        Tuple (dmesh, mf2, mf3, substrate_markers, curv_markers, u_nuc, a_nuc)
    Where
        dmesh: dolfin mesh for whole geometry
        mf2: mesh facet markers (surface mesh function)
        mf3: mesh cell markers (volume mesh function)
        substrate_markers: marks location of substrate (incl nanopillars) (surface mesh function)
        curv_markers: 0D mesh function (points) defining curvature at each vertex
        u_nuc: dolfin function defining nuclear deformations over reference ellipsoid
        a_nuc: dolfin function defining nuclear envelope stretch over reference ellipsoid
    """
    import gmsh
    maxOuterDim = max([rRad, zRad])
    maxInnerDim = max([rRad-thickness[0], zRad-thickness[1]])
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * maxOuterDim
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.1 * maxInnerDim
    if np.isclose(hTop, 0):
        hTop = 0.2 * maxInnerDim
    if rRad-thickness[0] <= 0 or zRad-thickness[1] <= 0:
        raise ValueError('Nuclear radius must be greater than thickness of NE')
    if np.isclose(thickness[0], 0) or np.isclose(thickness[1], 0):
        raise ValueError('NE thickness must be nonzero')
    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("3dcell")

    # first add outer body and revolve
    outer_top = gmsh.model.occ.add_point(0.0, 0.0, 2*zRad)
    outer_bottom = gmsh.model.occ.add_point(0.0, 0.0, 0.0)

    outer_edge = gmsh.model.occ.add_ellipse(0, 0, zRad, rRad, zRad, angle1=-np.pi/2, angle2=np.pi/2,
                                            zAxis=[0,1,0], xAxis=[1,0,0])
    symm_axis_tag = gmsh.model.occ.add_line(outer_top, outer_bottom)
    outer_loop_tag = gmsh.model.occ.add_curve_loop(
        [outer_edge, symm_axis_tag]
    )
    cell_plane_tag = gmsh.model.occ.add_plane_surface([outer_loop_tag])
    # gmsh.model.occ.synchronize()
    outer_shape = gmsh.model.occ.revolve([(2, cell_plane_tag)], 0, 0, 0, 0, 0, 1, 2*np.pi*sym_fraction)

    outer_shape_tags = []
    for i in range(len(outer_shape)):
        if outer_shape[i][0] == 3:  # pull out tags associated with 3d objects
            outer_shape_tags.append(outer_shape[i][1])
    assert len(outer_shape_tags) == 1  # should be just one 3D body from the full revolution

    # Create inner shape
    inner_top = gmsh.model.occ.add_point(0.0, 0.0, 2*zRad-thickness[1])
    inner_bottom = gmsh.model.occ.add_point(0.0, 0.0, thickness[1])
    
    inner_edge = gmsh.model.occ.add_ellipse(0, 0, zRad, rRad-thickness[0], zRad-thickness[1], 
                                            angle1=-np.pi/2, angle2=np.pi/2,
                                            zAxis=[0,1,0], xAxis=[1,0,0])
    symm_inner_tag = gmsh.model.occ.add_line(inner_top, inner_bottom)
    inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_edge, symm_inner_tag])
    inner_plane_tag = gmsh.model.occ.add_plane_surface([inner_loop_tag])
    inner_shape = gmsh.model.occ.revolve([(2, inner_plane_tag)], 
                                        0, 0, 0, 0, 0, 1, 2 * np.pi * sym_fraction)

    inner_shape_tags = []
    for i in range(len(inner_shape)):
        if inner_shape[i][0] == 3:  # pull out tags associated with 3d objects
            inner_shape_tags.append(inner_shape[i][1])
    assert len(inner_shape_tags) == 1  # should be just one 3D body from the full revolution

    # Create interface between 2 objects
    two_shapes, (outer_shape_map, inner_shape_map) = gmsh.model.occ.fragment(
        [(3, outer_shape_tags[0])], [(3, inner_shape_tags[0])]
    )
    # outer_shape_tags = inner_shape_tags

    gmsh.model.occ.synchronize()

    # Get the outer boundary
    outer_shell = gmsh.model.getBoundary(two_shapes, oriented=False)
    outer_shell_tags = []
    for i in range(len(outer_shell)):
        outer_shell_tags.append(outer_shell[i][1])
        # embed additional points on boundary for mesh refinement
    # gmsh.model.occ.synchronize()
    # outer_shell_init = gmsh.model.getBoundary([(3,outer_shape_tags[0])], oriented=False)
    # embed_points = []
    # center_pt = gmsh.model.occ.add_point(0.0, 0.0, zRad)
    for j in range(NE_layers-1):
        cur_rthickness = (j+1)*thickness[0]/NE_layers
        cur_zthickness = (j+1)*thickness[1]/NE_layers
        # rRadCur = rRad - cur_rthickness
        # zRadCur = zRad - cur_zthickness
        # xval = 0.05
        # yval = 0.005
        # xRight = rRadCur/np.sqrt(1+(yval/xval)**2)
        # yRight = rRadCur/np.sqrt(1+(xval/yval)**2)
        # zvaltop = zRad + zRadCur*np.sqrt(1 - (xval**2+yval**2)/rRadCur**2)
        # zvalbottom = zRad - zRadCur*np.sqrt(1 - (xval**2+yval**2)/rRadCur**2)
        # layer_top = gmsh.model.occ.add_point(xval, yval, zvaltop)
        # layer_middle = gmsh.model.occ.add_point(xRight, yRight, zRad)
        # layer_bottom = gmsh.model.occ.add_point(xval, yval, zvalbottom)
        # layer_edge = gmsh.model.occ.add_ellipse_arc(layer_top, center_pt, layer_middle, layer_bottom)
        d_angle = 0.02*np.pi/2
        layer_edge = gmsh.model.occ.add_ellipse(0, 0, zRad, rRad-cur_rthickness, zRad-cur_zthickness, 
                                            angle1=-np.pi/2 + d_angle, angle2=np.pi/2 - d_angle,
                                            zAxis=[0,1,0], xAxis=[1,0,0])
        gmsh.model.occ.rotate([(1, layer_edge)], 0, 0, 0, 0, 0, 1, d_angle)
        layer_shape = gmsh.model.occ.revolve([(1, layer_edge)], 
                                            0, 0, 0, 0, 0, 1, 2 * np.pi * sym_fraction - 2*d_angle)
        gmsh.model.occ.fragment(outer_shape_map, layer_shape)
        # embed_points.append((0, inner_top))
        # embed_points.append((0, inner_bottom))
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    # outer_shell_init = fragment_out[0][0][1]
    
    # Add physical markers for facets
    gmsh.model.add_physical_group(
        outer_shell[0][0], outer_shell_tags, tag=outer_marker
    )

    # Get the inner boundary
    inner_shell = gmsh.model.getBoundary(inner_shape_map, oriented=False)
    inner_shell_tags = []
    for i in range(len(inner_shell)):
        inner_shell_tags.append(inner_shell[i][1])
    gmsh.model.add_physical_group(inner_shell[0][0], inner_shell_tags)#, tag=interface_marker)

    # Physical markers for volumes
    all_volumes = []
    for i in range(len(outer_shape_map)):
        if outer_shape_map[i][0] == 3:
            all_volumes.append(outer_shape_map[i][1])

    inner_volume = [tag[1] for tag in inner_shape_map]
    outer_volume = []
    for vol in all_volumes:
        if vol not in inner_volume:
            outer_volume.append(vol)
    gmsh.model.add_physical_group(3, outer_volume, tag=outer_vol_tag)
    gmsh.model.add_physical_group(3, inner_volume, tag=inner_vol_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM and hInnerEdge at the inner membrane
        # between these, the value is interpolated based on the relative distance
        # between the two membranes.
        # Inside the inner shape, the value is interpolated between hInnerEdge
        # and lc3, where lc3 = max(hInnerEdge, 0.2*maxInnerDim)
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*maxOuterDim in the center
        # hNP is also considered here (mesh resolution at nanopillars),
        # weighted by how close the point is to the nanopillar surface
        R_rel_outer = np.sqrt(
            (x / rRad) ** 2 + (y / rRad) ** 2 + ((z-zRad) / zRad) ** 2
        )
        R_rel_inner = np.sqrt(
            (x / (rRad-thickness[0])) ** 2 + (y / (rRad-thickness[0])) ** 2 + ((z-zRad) / (zRad-thickness[1])) ** 2
        )
        lc3 = max(hInnerEdge, 0.2 * maxInnerDim)
        innerRad_scale = np.mean(
            [(rRad-thickness[0]) / rRad, (rRad-thickness[0]) / rRad, (zRad-thickness[1]) / zRad]
        )
        in_outer = R_rel_inner > 1
        lc1 = hEdge
        lc2 = hInnerEdge
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (1 - R_rel_outer) / (1 - innerRad_scale)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        # also scale by distance from z = 0 axis
        zWeight = (1 - np.tanh((z-zRad)/2))/2
        lcTest = (1-zWeight)*hTop + zWeight*lcTest
        return lcTest

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(3)
    rank = MPI.COMM_WORLD.rank
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_3dcell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"tmp_3dcell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "3dcell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # redetermine surface markers
    rad_eff1 = (rRad**2 * zRad)**(1/3)
    rad_eff2 = ((rRad-thickness[0])**2 * (zRad-thickness[1]))**(1/3)
    thickness_thresh = ((rad_eff1 - rad_eff2) / NE_layers)
    mf2 = d.MeshFunction("size_t", dmesh, 2, 0)
    # class OuterSurf(d.SubDomain):
    #     def inside(self, x, on_boundary):
    #         bound_val = pow(pow(x[0]/rRad,2) + pow(x[1]/rRad,2) + 
    #                     pow((x[2]-zRad)/zRad,2),0.5)
    #         cutoffFrac = 0.99#1-0.95*thickness_thresh/rad_eff1
    #         return bound_val > cutoffFrac and on_boundary
    # class OuterSurf(d.SubDomain):
    #     def inside(self, x, on_boundary):
    #         if sym_fraction == 1:
    #             return on_boundary 
    #         elif sym_fraction == 1/2:
    #             return on_boundary and not x[0]==0
    #         elif sym_fraction == 1/4:
    #             return on_boundary and not (x[0]==0 or x[1]==0)
    #         else:
    #             thetaCur = np.arctan2(x[1], x[0])
    #             return (on_boundary and not 
    #                     (x[0]==0 or np.isclose(thetaCur,2*np.pi*sym_fraction)))
    # class InnerSurf(d.SubDomain):
    #     def inside(self, x, on_boundary):

    #         bound_val = pow(pow(x[0]/(rRad-thickness[0]),2) + pow(x[1]/(rRad-thickness[0]),2) +
    #                     pow((x[2]-zRad)/(zRad-thickness[1]),2),0.5)
    #         cutoffFrac1 = 1.01#1+0.95*thickness_thresh/rad_eff2
    #         cutoffFrac2 = 0.99#1-0.95*thickness_thresh/rad_eff2
    #         return  bound_val < cutoffFrac1 and bound_val > cutoffFrac2
    # outerSurf = OuterSurf()
    # innerSurf = InnerSurf()
    # outerSurf.mark(mf2, outer_marker)
    # innerSurf.mark(mf2, interface_marker)
    for f in d.facets(dmesh):
        f_type = facet_topology(f, mf3)[0]
        if f_type == "interface":
            mf2[f] = interface_marker
        elif f_type == "boundary":
            if sym_fraction == 1:
                mf2[f] = outer_marker
            elif sym_fraction == 1/2:
                if f.midpoint().x() > 1e-6:
                    mf2[f] = outer_marker
            elif sym_fraction == 1/4:
                if f.midpoint().x() > 1e-6 and f.midpoint().y() > 1e-6:
                    mf2[f] = outer_marker
            else:
                thetaCur = np.arctan2(f.midpoint().y(), f.midpoint().x())
                if f.midpoint().x() > 1e-6 and thetaCur < 0.999*2*np.pi*sym_fraction:
                    mf2[f] = outer_marker

    return (dmesh, mf2, mf3)

def create_substrate(
    LBox: float = 20.0,
    TBox: float = 1.0,
    hEdge: float = 0,
    outer_marker: int = 10,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
    nanopillars: Tuple[float, float, float] = [0, 0, 0],
    use_tmp: bool = False,
    contact_rad: float = np.inf,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh of the substrate.
    """
    import gmsh
    nanopillar_rad = nanopillars[0]
    nanopillar_height = nanopillars[1]
    nanopillar_spacing = nanopillars[2]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("substrate")

    # first add outer body and revolve
    outer_shape = gmsh.model.occ.add_box(-LBox/2,-LBox/2,-TBox,LBox,LBox,TBox)
    outer_shape = [(3, outer_shape)]
    # outer_shape = []
    
    if np.all(np.array(nanopillars) != 0):
        num_pillars = 2*np.floor(LBox/(2*nanopillar_spacing)) + 1
        rMax = nanopillar_spacing * np.floor(LBox/(2*nanopillar_spacing))
        test_coords = np.linspace(-rMax, rMax, int(num_pillars))
        xTest, yTest = np.meshgrid(test_coords, test_coords)
        xTest = np.reshape(xTest, (len(xTest)**2))
        yTest = np.reshape(yTest, (len(yTest)**2))
        for i in range(len(xTest)):
            rTest = np.sqrt(xTest[i]**2 + yTest[i]**2)
            xCurv = 0.2
            xSteric = 0.05
            keep_logic1 = rTest <= contact_rad-nanopillar_rad-xCurv-xSteric
            keep_logic2 = rTest > contact_rad+nanopillar_rad
            keep_logic = np.logical_or(keep_logic1, keep_logic2)
            if keep_logic:
                np_shape = gmsh.model.occ.add_cylinder(xTest[i], yTest[i], 0.0, 0, 0, nanopillar_height, nanopillar_rad)
                # outer_shape.append((3, np_shape))
                (outer_shape, outer_shape_map) = gmsh.model.occ.fuse(outer_shape, [(3, np_shape)])
                outer_shape_list = []

    outer_shape_tags = []
    for i in range(len(outer_shape)):
        if outer_shape[i][0] == 3:  # pull out tags associated with 3d objects
            outer_shape_tags.append(outer_shape[i][1])
    # assert len(outer_shape_tags) == 1  # should be just one 3D body from the full revolution
    gmsh.model.occ.synchronize()
    gmsh.model.add_physical_group(3, outer_shape_tags, tag=outer_vol_tag)
    # facets = gmsh.model.getBoundary([(3, outer_shape_tags[0])])
    facets = gmsh.model.get_entities(dim=2)
    facet_tags = []
    for i in range(len(facets)):
        facet_tags.append(facets[i][1])
    gmsh.model.add_physical_group(2, facet_tags, tag=outer_marker)
    def meshSizeCallback(dim, tag, x, y, z, lc):
        return hEdge

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)

    gmsh.model.mesh.generate(3)
    rank = MPI.COMM_WORLD.rank
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_3dcell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"tmp_3dcell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "substrate.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    return dmesh, mf2, mf3

def get_shape_coords(contactRad, nanopillars, nuc_compression, no_nuc=False):
    aVecRef = np.array([35, 2000, 1000, 1200, 1500, 1000])
    bVecRef = np.array([0.94, 0.4, 0.4, 0.4, 0.4, 0.4])
    cVecRef = np.array([0.01, 9.72, 15, 24.5, 33.2, 42])
    dVecRef = np.array([30, 15, 10, 1.0, 0.12, 0.0322])
    RList = np.array([10, 13, 15, 20, 25, 30])
    scaleFactor = 0.82
    nucScaleFactor = 0.8
    aVecRef = aVecRef / scaleFactor**4
    cVecRef = cVecRef / scaleFactor
    dVecRef = dVecRef / scaleFactor**4
    RList = RList / scaleFactor
    findMatch = np.isclose(RList, contactRad)
    if np.any(findMatch):
        idx = np.nonzero(findMatch)[0][0]
        refParam = [aVecRef[idx], bVecRef[idx], cVecRef[idx], dVecRef[idx], RList[idx]]
    elif contactRad < max(RList) and contactRad > min(RList):
        aVal = np.interp(contactRad, np.array(RList), np.array(aVecRef))
        bVal = np.interp(contactRad, np.array(RList), np.array(bVecRef))
        cVal = np.interp(contactRad, np.array(RList), np.array(cVecRef))
        dVal = np.interp(contactRad, np.array(RList), np.array(dVecRef))
        refParam = [aVal, bVal, cVal, dVal, contactRad]
    else:
        raise ValueError("This shape is outside the specified range")

    nuc_vol = (4/3)*np.pi*5.3*5.3*2.4/nucScaleFactor**3
    nanopillar_rad = nanopillars[0]
    nanopillar_height = nanopillars[1]
    nanopillar_spacing = nanopillars[2]
    if np.all(np.array(nanopillars) != 0):
        xSteric = 0.05
        xCurv = 0.2
        num_pillars = 2*np.ceil(contactRad/nanopillar_spacing) + 1
        rMax = nanopillar_spacing * np.ceil(contactRad/nanopillar_spacing)
        test_coords = np.linspace(-rMax, rMax, int(num_pillars))
        xTest, yTest = np.meshgrid(test_coords, test_coords)
        rTest = np.sqrt(xTest**2 + yTest**2)
        dr = nanopillar_rad+xCurv+xSteric
        dr0 = nanopillar_rad + xSteric # neglect curved portion for volume approx
        keep_logic = rTest <= contactRad+dr
        xTest, yTest = xTest[keep_logic], yTest[keep_logic]
        frac_np = np.ones_like(xTest)
        for i in range(len(xTest)):
            rCur = np.sqrt(xTest[i]**2 + yTest[i]**2)
            if rCur >= contactRad-dr0 and rCur <= contactRad+dr0:
                # frac_np[i] = (contactRad-(rCur-dr))/(2*dr) # approximation!
                # better approximation...
                d = rCur
                R0 = contactRad
                sqrtArg = (-d+dr0+R0)*(d+dr0-R0)*(d-dr0+R0)*(d+dr0+R0)
                Acur = (dr0**2 * np.arccos((d**2 + dr0**2 - R0**2)/(2*d*dr0)) + 
                        R0**2*np.arccos((d**2+R0**2-dr0**2)/(2*d*R0)) - 0.5*np.sqrt(sqrtArg))
                if np.isnan(Acur):
                    print('what?')
                frac_np[i] = Acur / (np.pi*dr0**2)
        
        num_pillars_tot = sum(frac_np) #len(xTest)
        nanopillar_vol = num_pillars_tot * np.pi*(
            (nanopillar_rad+xSteric)**2 * (nanopillar_height-xSteric-xCurv) +
            xSteric*(nanopillar_rad**2 + nanopillar_rad*xSteric*np.pi/2 + 2*xSteric**2/3) +
            xCurv*((nanopillar_rad+xSteric+xCurv)**2 - (nanopillar_rad+xSteric+xCurv)*xCurv*np.pi/2 + 
                   2*xCurv**2/3))
        zOffset = nanopillar_height
    else:
        nanopillar_vol = 0
        zOffset = 0
    
    targetVol = 480/scaleFactor**3 *4 + nuc_vol + nanopillar_vol
    # rValsOuter, zValsOuter = implicit_curve(make_expr(*refParam))
    rValsOuter, zValsOuter, outParam = shape_adj_axisymm(refParam, zOffset, targetVol)
    rValsOuter, zValsOuter = dilate_axisymm(rValsOuter, zValsOuter, targetVol, 0.0)

    zMax = max(zValsOuter)
    if no_nuc:
        innerExpr = ""
    else:
        innerExpr = get_inner(zOffset, zMax, nucScaleFactor, nuc_compression)
        aInner, bInner, r0Inner, z0Inner = get_inner_param(zOffset, zMax, nucScaleFactor, nuc_compression)
        innerParam = [aInner, bInner, r0Inner, z0Inner]
    if not innerExpr == "" and np.all(np.array(nanopillars) != 0):
        u_nuc, aInner, bInner = get_u_nuc(zOffset, zMax, nucScaleFactor, nuc_compression, nanopillars)
        aMod = aInner - innerParam[0]
        bMod = bInner - innerParam[1]
        # update inner coordinates according to uEllipsoid
        innerParam[0] += aMod
        innerParam[1] += bMod
        assert bMod == 0
        innerExpr = get_inner(zOffset, zMax, nucScaleFactor, nuc_compression, aMod, bMod)
        if nuc_compression > 0:
            rValsInner, zValsInner = implicit_curve(innerExpr, num_points=501)
        else:
            rValsInner, zValsInner = implicit_curve(innerExpr, num_points=101)
        rScale = (aInner + aMod) / aInner
    elif not innerExpr == "":
        rValsInner, zValsInner = implicit_curve(innerExpr, num_points=101)
        u_nuc = []
        rScale = 1
    else:
        rValsInner = []
        zValsInner = []
        innerParam = []
        u_nuc = []
        rScale = 1

    return [rValsOuter, zValsOuter, rValsInner, zValsInner, innerParam, u_nuc, rScale, outParam]

def get_u_nuc(zOffset, zMax, nucScaleFactor, nuc_compression, nanopillars):
    nanopillar_rad, nanopillar_height, nanopillar_spacing = nanopillars[:]
    xSteric = 0.05
    nanopillar_rad += 2*xSteric # to avoid collision with PM
    aInner0, bInner, r0Inner, z0Inner = get_inner_param(zOffset, zMax, nucScaleFactor, nuc_compression)
    aInner = aInner0
    # define nanopillar locations
    xMax = np.ceil(aInner / nanopillar_spacing) * nanopillar_spacing
    xNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    yNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    xNP, yNP = np.meshgrid(xNP, yNP)
    xNP = xNP.flatten()
    yNP = yNP.flatten()

    indentation = nuc_compression
    zNP = -bInner + indentation

    def np_displ(coords, xNP, yNP, zNP, radNP):
        if len(zNP) == 1:
            zNP = zNP * np.ones_like(xNP)
        else:
            assert len(zNP) == len(xNP)
        dist_vals = np.sqrt(np.power(xNP-coords[0], 2) + np.power(yNP-coords[1], 2)) 
        uVal = 0
        for i in range(len(xNP)):
            uRef = zNP[i] - coords[2]
            if uRef > 0:
                if dist_vals[i] < radNP:
                    uVal = uRef
                    break
                else:
                    uVal += uRef*np.exp(-((dist_vals[i]-radNP)/0.2)**2)
        return np.array([0, 0, uVal])

    from smart import mesh_tools
    mesh_ellipsoid, mf2, mf3 = mesh_tools.create_ellipsoids(outerRad=[aInner,aInner,bInner], innerRad=[0,0,0], 
                                                hEdge=0.1, hInnerEdge=0.5)
    
    mesh_bound = d.create_meshview(mf2, 10)
    V_full = d.FunctionSpace(mesh_ellipsoid, d.VectorElement("P", mesh_ellipsoid.ufl_cell(), degree = 1, dim = 3))
    u_full = d.Function(V_full)
    coords_full = V_full.tabulate_dof_coordinates()
    uvec = u_full.vector()[:]
    vol_error = 100
    ref_vol = d.assemble(1.0 * d.Measure("dx", mesh_ellipsoid))
    vol_vec = []
    alpha_vec = [1]
    u_ellipsoid = d.Function(V_full)
    while vol_error > 0.01:
        if vol_error < 100: # not first iteration
            if len(alpha_vec) == 1:
                alpha_vec.append(np.sqrt(ref_vol/vol_vec[-1]))
            else:
                alpha_vec.append(alpha_vec[-2] + 
                                 (ref_vol-vol_vec[-2])*(alpha_vec[-1]-alpha_vec[-2])/(vol_vec[-1]-vol_vec[-2]))
            aInner = aInner0*alpha_vec[-1]
            u_ellipsoid = d.interpolate(d.Expression(("x[0]*(alpha-1)", "x[1]*(alpha-1)", "0.0"), 
                                            z0 = z0Inner, alpha=alpha_vec[-1], degree = 1), V_full)
            coords_full = V_full.tabulate_dof_coordinates()
            u_ellipsoid_vec = u_ellipsoid.vector()[:]
            for i in range(0, len(coords_full), 3):
                coords_full[i] += u_ellipsoid_vec[i:i+3]
                coords_full[i+1] += u_ellipsoid_vec[i:i+3]
                coords_full[i+2] += u_ellipsoid_vec[i:i+3]
        for i in range(0,len(coords_full),3):
            uvec[i:i+3] = np_displ(coords_full[i,:], xNP, yNP, [zNP], nanopillar_rad)
            dist_vals = np.sqrt(np.power(xNP-coords_full[i,0], 2) + np.power(yNP-coords_full[i,1], 2))
            if min(dist_vals) < nanopillar_rad and min(dist_vals) > nanopillar_rad-xSteric and coords_full[i,2] < zNP:
                assert uvec[i+2] > 0.99*(zNP-coords_full[i,2])
            
        uvec = uvec + u_ellipsoid.vector()[:]
        u_full.vector().set_local(uvec)
        u_full.vector().apply("insert")
        mesh_copy = d.Mesh(mesh_ellipsoid)
        d.ALE.move(mesh_copy, u_full)
        vol_vec.append(d.assemble(1.0 * d.Measure("dx", mesh_copy)))
        vol_error = 100 * np.abs(vol_vec[-1] - ref_vol) / ref_vol

    V_bound =  d.FunctionSpace(mesh_bound, d.VectorElement("P", mesh_bound.ufl_cell(), degree = 1, dim = 3))
    u_bound = d.Function(V_bound)
    uvec = u_bound.vector()[:]
    bound_coord = V_bound.tabulate_dof_coordinates()
    for i in range(0, len(uvec), 3):
        cur_coord = bound_coord[i]
        uvec[i:i+3] = u_full(cur_coord)
    u_bound.vector().set_local(uvec)
    u_bound.vector().apply("insert")
    u_bound.set_allow_extrapolation(True)
    # ellipsoid_deform = np.reshape(u_ellipsoid.vector()[:],[-1,3])
    aMod = aInner - aInner0 #max(ellipsoid_deform[:,0])
    bMod = 0.0 #max(ellipsoid_deform[:,2])
    # u_shift = d.project(d.Expression(("0.0", "0.0", "z0"), z0 = z0Inner, degree = 1), V_bound)
    # d.ALE.move(mesh_bound, u_shift)
    return (u_bound, aInner, bInner)

def compute_stretch(u: d.Function, aInner: float, bInner: float, nanopillars: list, z0Inner: float):
    nanopillar_spacing = nanopillars[2]
    xMax = np.ceil(aInner / nanopillar_spacing) * nanopillar_spacing
    xNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    yNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    xNP, yNP = np.meshgrid(xNP, yNP)
    xNP = xNP.flatten()
    yNP = yNP.flatten()
    
    I = d.Identity(3)
    F = I + d.grad(u)    # Deformation gradient
    C = F.T*F          # Right Cauchy-Green deformation tensor
    J = d.det(F)
    mesh_bound = u.function_space().mesh()
    V_vector = d.FunctionSpace(mesh_bound, d.VectorElement("P", mesh_bound.ufl_cell(), degree = 1, dim = 3))
    V_scalar = d.FunctionSpace(mesh_bound, "P", 1)
    normal_expr = d.Expression(("(x[0]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2],2)/pow(b,4))", 
                                "(x[1]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2],2)/pow(b,4))", 
                                "(x[2]/pow(b,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2],2)/pow(b,4))"),
                                a=aInner, b=bInner, degree=1)
    normals = d.project(normal_expr, V_vector)
    a_vector = J * d.dot(normals, d.inv(F))
    a_vector = d.project(a_vector, V_vector)
    a_scalar = d.Function(V_scalar)
    V_scalar_coords = V_scalar.tabulate_dof_coordinates()
    avec = a_scalar.vector()[:]
    for i in range(len(avec)):
        cur_coord = V_scalar_coords[i]
        avec[i] = calc_stretch_NP(cur_coord[0],cur_coord[1],cur_coord[2]+z0Inner,
                 aInner,bInner,z0Inner,xNP,yNP,nanopillars)
        # a_cur = a_vector(V_scalar_coords[i])
        # avec[i] = np.sqrt(a_cur[0]**2 + a_cur[1]**2 + a_cur[2]**2)
    a_scalar.vector().set_local(avec)
    a_scalar.vector().apply("insert")
    a_nuc = a_scalar
    a_nuc.set_allow_extrapolation(True)
    return a_nuc

def calc_vol_axisymm(rVec, zVec):
    vol = 0
    for i in range(1, len(rVec)):
        dr = rVec[i-1] - rVec[i]
        dz = zVec[i-1] - zVec[i]
        vol += np.pi*(rVec[i]**2 * dz + rVec[i]*dr*dz + dr**2 * dz / 3)
    return vol

def dilate_axisymm(rVec, zVec, volTarget, zThresh):
    volCur = calc_vol_axisymm(rVec, zVec)
    normals = [[0, 1]]
    sTot = 0
    for i in range(1, len(rVec)-1):
        dr1 = rVec[i] - rVec[i-1]
        dz1 = zVec[i] - zVec[i-1]
        ds1 = np.sqrt(dr1**2 + dz1**2)
        dr2 = rVec[i+1] - rVec[i]
        dz2 = zVec[i+1] - zVec[i]
        ds2 = np.sqrt(dr2**2 + dz2**2)
        curVec = [-(dz1/ds1+dz2/ds2)/2, (dr1/ds1+dr2/ds2)/2]
        curVec[0] = curVec[0] / np.sqrt(curVec[0]**2 + curVec[1]**2)
        curVec[1] = curVec[1] / np.sqrt(curVec[0]**2 + curVec[1]**2)
        normals.append(curVec)
        sTot += np.sqrt(ds1)
    sTot += np.sqrt(ds2)
    normals.append([1, 0])
    normalMove = 1.0*sTot/len(rVec)
    curSign = np.sign(volTarget-volCur)
    firstIt = True
    volPrev = volCur
    adjIdx = np.nonzero(zVec <= zThresh + 1e-12)
    if len(adjIdx[0]) == 0:
        adjIdx = len(rVec)
    else:
        adjIdx = adjIdx[0][0]
    while np.abs(volCur-volTarget) > 0.001*volTarget:
        for i in range(adjIdx-1):
            curMove = normalMove*np.sign(volTarget-volCur)*(len(rVec)-i)/len(rVec)
            rVec[i] += curMove*normals[i][0]
            zVec[i] += curMove*normals[i][1]
        volCur = calc_vol_axisymm(rVec, zVec)
        if firstIt:
            if curSign == np.sign(volTarget-volCur):
                # raise ValueError("Desired volume is too far off for adjustments")
                return (rVec, zVec)
            else:
                firstIt = False
        normalMove = normalMove * np.abs((volTarget-volCur)/(volCur-volPrev))
        volPrev = volCur
    return (rVec, zVec)

def make_expr(a, b, c, d, R):
        return (f"(1 - z**4/({a}+z**4)) * (r**2 + z**2) + "
                f"{b}*(r**2 + (z+{c})**2)*z**4 / ({d} + z**4) - {R**2}")

def shape_adj_axisymm(paramVec, zOffset, volTarget):
    rVec, zVec = implicit_curve(make_expr(*paramVec))
    zVec[0:-2] += zOffset
    volCur = calc_vol_axisymm(rVec, zVec)
    curSign = np.sign(volTarget-volCur)
    firstIt = True
    volPrev = volCur
    magChange = 50
    while np.abs(volCur-volTarget) > 0.01*volTarget:
        if volCur < volTarget:
            paramVec[0] = paramVec[0]/np.exp(np.log(magChange)/2)
            paramVec[1] = paramVec[1]/np.exp(np.log(magChange)/4)
            paramVec[2] = paramVec[2]/np.exp(np.log(magChange)/4)
            paramVec[3] = paramVec[3]*magChange
        else:
            paramVec[0] = paramVec[0]*np.exp(np.log(magChange)/2)
            paramVec[1] = paramVec[1]*np.exp(np.log(magChange)/4)
            paramVec[2] = paramVec[2]*np.exp(np.log(magChange)/4)
            paramVec[3] = paramVec[3]/magChange
        rVec, zVec = implicit_curve(make_expr(*paramVec))
        zVec[0:-2] += zOffset
        volCur = calc_vol_axisymm(rVec, zVec)
        if firstIt:
            if curSign == np.sign(volTarget-volCur):
                # raise ValueError("Desired volume is too far off for adjustments")
                return (rVec, zVec)
            else:
                firstIt = False
        magChange = 1 + (magChange-1)*np.abs((volTarget-volCur)/(volCur-volPrev))
        volPrev = volCur
        
    return (rVec, zVec, paramVec)

def get_inner(zOffset, zMax, scaleFactor, nuc_compression, aMod=0, bMod=0):
    # zMid = (zOffset + zMax) / 2
    zRad = 2.4/scaleFactor + bMod
    rRad = 5.3/scaleFactor + aMod
    if (zMax - zOffset - 1.0 + nuc_compression) <= zRad*2:
        zRad = (zMax-zOffset-1.0 + nuc_compression)/2 + bMod
        rRad = np.sqrt((5.3**2 * 2.4/scaleFactor**3) / zRad) + aMod
        if zRad < 0:
            raise ValueError("Nucleus does not fit")
    zMid = zOffset + zRad + 0.2 - nuc_compression
    return f"(r/{rRad})**2 + ((z-{zMid})/{zRad})**2 - 1"

def get_inner_param(zOffset, zMax, scaleFactor, nuc_compression, aMod=0, bMod=0):
    zRad = 2.4/scaleFactor + bMod
    rRad = 5.3/scaleFactor + aMod
    if (zMax - zOffset - 1.0 + nuc_compression) <= zRad*2:
        zRad = (zMax-zOffset-1.0 + nuc_compression)/2 + bMod
        rRad = np.sqrt((5.3**2 * 2.4/scaleFactor**3) / zRad) + aMod
        if zRad < 0:
            raise ValueError("Nucleus does not fit")
    zMid = zOffset + zRad + 0.2 - nuc_compression
    return (rRad, zRad, 0.0, zMid)

def compute_curvature_1D(
    rVals, zVals, 
    curvRes: float = 0.1,
    incl_parallel: bool = False, 
    use_tmp = False,
    comm: MPI.Comm = d.MPI.comm_world,
):
    """
    Use dolfin functions to estimate curvature on boundary mesh.
    Boundary meshes are created by extracting the Meshview associated
    with each facet marker value in facet_marker_vec.
    The length of cell_marker_vec must be the same length as facet_marker_vec,
    with each value in the list identifying the marker value for a domain that
    contains the boundary identified by the associated facet_marker.
    For instance, if facet_marker_vec=[10,12] and cell_marker_vec=[1,2], then
    the domain over which mf_cell=1 must contain the domain mf_facet=10 on its boundary
    and the domain mf_cell=2 must contain domain mf_facet=12 on its boundary.
    Mean curvature is approximated by first projecting facet normals (n) onto a boundary
    finite element space and then solving a variational form of kappa = -div(n).

    Args:
        ref_mesh: dolfin mesh describing the entire geometry (parent mesh)
        mf_facet: facet mesh function with boundary domain markers
        mf_cell: cell mesh function with cell domain markers
        facet_marker_vec: list with values of facet markers to iterate over
        cell_marker_vec: list with values of cell markers (see above)
        half_mesh_data: tuple with dolfin mesh for half domain and
                        facet mesh function over half domain. If not specified,
                        this is empty and the curvature values are not mapped
                        onto the half domain
    Returns:
        kappa_mf: Vertex mesh function containing mean curvature values
    """
    facet_marker = 10
    cell_marker = 1
    import gmsh
    gmsh.initialize()
    outer_tag_list = []
    line_list = []
    for i in range(len(rVals)):
        cur_tag = gmsh.model.occ.add_point(rVals[i], 0, zVals[i])
        outer_tag_list.append(cur_tag)
        if i > 0:
            cur_line = gmsh.model.occ.add_line(outer_tag_list[-2], cur_tag)
            line_list.append(cur_line)
    # outer_spline_tag = gmsh.model.occ.add_spline(outer_tag_list)
    outer_loop_tag = gmsh.model.occ.add_curve_loop(line_list)
    cell_plane_tag = gmsh.model.occ.add_plane_surface([outer_loop_tag])
    gmsh.model.occ.synchronize()
    facets = gmsh.model.getBoundary([(2, cell_plane_tag)])
    facet_tag_list = []
    for i in range(len(facets)):
        facet_tag_list.append(facets[i][1])
    gmsh.model.add_physical_group(1, facet_tag_list, tag=facet_marker)
    gmsh.model.add_physical_group(2, [cell_plane_tag], tag=cell_marker)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        return curvRes
    
    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(2)
    rank = MPI.COMM_WORLD.rank
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_2DCell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"tmp_2DCell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "2DCell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()
    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf_facet, mf_cell = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 2, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    
    mesh = d.MeshView.create(mf_cell, cell_marker)
    bmesh = d.MeshView.create(mf_facet, facet_marker)

    n = d.FacetNormal(mesh)
    # estimate facet normals in CG2 for better smoothness and accuracy
    V = d.VectorFunctionSpace(mesh, "CG", 2)
    u = d.TrialFunction(V)
    v = d.TestFunction(V)
    ds = d.Measure("ds", mesh)
    a = d.inner(u, v) * ds
    lform = d.inner(n, v) * ds
    A = d.assemble(a, keep_diagonal=True)
    L = d.assemble(lform)

    A.ident_zeros()
    nh = d.Function(V)
    d.solve(A, nh.vector(), L)  # project facet normals onto CG1

    Vb = d.FunctionSpace(bmesh, "CG", 1)
    Vb_vec = d.VectorFunctionSpace(bmesh, "CG", 1)
    nb = d.interpolate(nh, Vb_vec)
    p, q = d.TrialFunction(Vb), d.TestFunction(Vb)
    dx = d.Measure("dx", bmesh)
    a = d.inner(p, q) * dx
    lform = d.inner(d.div(nb), q) * dx
    A = d.assemble(a, keep_diagonal=True)
    L = d.assemble(lform)
    A.ident_zeros()
    kappab = d.Function(Vb)
    d.solve(A, kappab.vector(), L)

    if incl_parallel:  # then include out of plane (parallel) curvature as well
        kappab_vec = kappab.vector().get_local()
        nb_vec = nb.vector().get_local()
        nb_vec = nb_vec.reshape((int(len(nb_vec) / 3), 3))
        normal_angle = np.arctan2(nb_vec[:, 0], nb_vec[:, 2])
        x = Vb.tabulate_dof_coordinates()
        parallel_curv = kappab_vec
        logic = x[:,0] != 0
        parallel_curv[logic] = np.sin(normal_angle[logic]) / x[logic, 0]
        # parallel_curv = np.sin(normal_angle) / x[:, 0]
        # inf_logic = np.isinf(parallel_curv)
        # parallel_curv[inf_logic] = kappab_vec[inf_logic]
        kappab.vector().set_local((kappab_vec + parallel_curv) / 2.0)
        kappab.vector().apply("insert")

    # return kappab at r,z values
    kappab.set_allow_extrapolation(True)
    # curv_vals = np.zeros(len(rVals))
    # for i in range(len(rVals)):
    #     curv_vals[i] = kappab(rVals[i], 0, zVals[i])
    return kappab

def compute_curvature_ellipse(
    rVals, zVals,
    incl_parallel: bool = False, 
):
    """
    Give analytical value for curvature on ellipse surface.
    If incl_parallel, parallel curvature is also included for axisymmetric case,
    assuming that r = 0 is the axis of symmetry (3D shape is a spheroid in this case,
    i.e. an ellipsoid with 2 axes sharing the same length)
    """
    aVal = (max(rVals) - min(rVals))/2
    bVal = (max(zVals) - min(zVals))/2
    r0 = (max(rVals) + min(rVals))/2
    z0 = (max(zVals) + min(zVals))/2
    rVals = rVals - r0
    zVals = zVals - z0

    curv_vals = np.zeros(len(rVals))
    for i in range(len(rVals)):
        tCur = np.arctan2(zVals[i]/bVal, rVals[i]/aVal)
        curv_vals[i] = aVal*bVal/(bVal**2 * np.cos(tCur)**2 + aVal**2 * np.sin(tCur)**2)**(3/2)
        if incl_parallel:
            phiCur = np.arctan((aVal/bVal)*np.tan(tCur))
            if rVals[i]!=0: # if =0, then the same meridional and parallel curv
                cp = np.sin(np.pi/2 - phiCur) / rVals[i]
                curv_vals[i] = (curv_vals[i] + cp)/2
    return curv_vals

def compute_curvature_ellipse_alt(
    rVals, zVals, aVal, bVal, r0, z0,
    incl_parallel: bool = False, 
):
    """
    Give analytical value for curvature on ellipse surface.
    If incl_parallel, parallel curvature is also included for axisymmetric case,
    assuming that r = 0 is the axis of symmetry (3D shape is a spheroid in this case,
    i.e. an ellipsoid with 2 axes sharing the same length)
    """
    rVals = rVals - r0
    zVals = zVals - z0

    curv_vals = np.zeros(len(rVals))
    curv_alt = np.zeros(len(rVals))
    for i in range(len(rVals)):
        tCur = np.arctan2(zVals[i]/bVal, rVals[i]/aVal)
        curv_vals[i] = aVal*bVal/(bVal**2 * np.cos(tCur)**2 + aVal**2 * np.sin(tCur)**2)**(3/2)
        if incl_parallel:
            phiCur = np.arctan((aVal/bVal)*np.tan(tCur))
            if rVals[i]!=0: # if =0, then the same meridional and parallel curv
                cp = np.sin(np.pi/2 - phiCur) / rVals[i]
                curv_vals[i] = (curv_vals[i] + cp)/2
            # alt, use explicit formula
            num = bVal*(4*aVal**2 + 2*(aVal**2 - bVal**2)*(rVals[i]/aVal)**2)
            den = np.sqrt(2)*aVal*(2*aVal**2 + 2*(aVal**2 - bVal**2)*(rVals[i]/aVal)**2)**(3/2)
            curv_alt[i] = num / den
    # assert np.all(curv_alt == curv_vals)
    return curv_vals

def calc_curv_NP(x,y,z,
                 a,b,z0Inner,xNP,yNP,nanopillars):
    radNP, hNP, pNP = nanopillars[:]
    xSteric = 0.05
    radNP += 2*xSteric # to avoid collision with PM
    hNP += 0.2
    zNP = hNP * np.ones_like(xNP)
    dist_vals = np.sqrt(np.power(xNP-x, 2) + np.power(yNP-y, 2)) 
    alpha = np.sqrt(1 - (x**2+y**2)/a**2)
    zCur = z0Inner - b*alpha
    hex = b*x/(a**2 * alpha)
    hey = b*y/(a**2 * alpha)
    hexy = b*x*y / (a**4 * alpha**3)
    hexx = b*x**2 / (a**4 * alpha**3) + b / (a**2 * alpha)
    heyy = b*y**2 / (a**4 * alpha**3) + b / (a**2 * alpha)
    hx = hex
    hy = hey
    hxy = hexy
    hxx = hexx
    hyy = heyy
    for i in range(len(xNP)):
        uRef = zNP[i] - zCur
        if z < zNP[i]+1e-3:
            dxCur = x - xNP[i]
            dyCur = y - yNP[i]
            rlocal = np.sqrt(dxCur**2 + dyCur**2)
            drCur = rlocal - radNP
            sigma1 = np.exp(-(drCur/0.2)**2)
            if dist_vals[i] < radNP+1e-3:
                curv_val = 0
                return curv_val
            else:
                hx += -hex*sigma1 - 2*uRef*drCur*dxCur*sigma1/(0.2**2 * rlocal)
                hy += -hey*sigma1 - 2*uRef*drCur*dyCur*sigma1/(0.2**2 * rlocal)
                xyCur = (-hexy*sigma1 + 2*hex*drCur*dyCur*sigma1/(0.2**2 * rlocal) + 2*hey*drCur*dxCur*sigma1/(0.2**2 * rlocal) +
                        (2*sigma1*uRef*drCur*dxCur*dyCur/(0.2**2 * rlocal**2))*(2*drCur/0.2**2 + 1/rlocal) - 
                        2*uRef*dxCur*dyCur*sigma1/(0.2**2 * rlocal**2))
                xyCurAlt = (-hexy*sigma1 + 2*hey*drCur*dxCur*sigma1/(0.2**2 * rlocal) + 2*hex*drCur*dyCur*sigma1/(0.2**2 * rlocal) +
                        (2*sigma1*uRef*drCur*dxCur*dyCur/(0.2**2 * rlocal**2))*(2*drCur/0.2**2 + 1/rlocal) - 
                        2*uRef*dxCur*dyCur*sigma1/(0.2**2 * rlocal**2))
                assert np.isclose(xyCur, xyCurAlt), f"{xyCur} vs {xyCurAlt}"
                hxy += xyCurAlt
                hxx += (-hexx*sigma1 + 4*hex*drCur*dxCur*sigma1/(0.2**2 * rlocal) + 
                        (2*sigma1*uRef*drCur*dxCur**2/(0.2**2 * rlocal**2))*(2*drCur/0.2**2 + 1/rlocal) - 
                        2*uRef*dxCur**2*sigma1/(0.2**2 * rlocal**2) - 2*uRef*drCur*sigma1/(0.2**2 * rlocal))
                hyy += (-heyy*sigma1 + 4*hey*drCur*dyCur*sigma1/(0.2**2 * rlocal) + 
                        (2*sigma1*uRef*drCur*dyCur**2/(0.2**2 * rlocal**2))*(2*drCur/0.2**2 + 1/rlocal) - 
                        2*uRef*dyCur**2*sigma1/(0.2**2 * rlocal**2) - 2*uRef*drCur*sigma1/(0.2**2 * rlocal))
    num = (1+hy**2)*hxx - 2*hx*hy*hxy + (1+hx**2)*hyy
    den = 2*(1 + hx**2 + hy**2)**(3/2)
    curv_val = num/den   
    return curv_val

def calc_stretch_NP(x,y,zDef,
                 a,b,z0Inner,xNP,yNP,nanopillars):
    radNP, hNP, pNP = nanopillars[:]
    xSteric = 0.05
    radNP += 2*xSteric # to avoid collision with PM
    hNP += 0.2
    zNP = hNP * np.ones_like(xNP)
    dist_vals = np.sqrt(np.power(xNP-x, 2) + np.power(yNP-y, 2)) 
    alpha = np.sqrt(1 - (x**2+y**2)/a**2)
    zCur = z0Inner - b*alpha
    z = zCur - z0Inner
    N = np.sqrt((x**2+y**2)/a**4 + z**2/b**4)
    uzx = 0
    uzy = 0
    uzz = 0
    for i in range(len(xNP)):
        uRef = zNP[i] - zCur
        if zDef < zNP[i]+1e-6:
            dxCur = x - xNP[i]
            dyCur = y - yNP[i]
            rlocal = np.sqrt(dxCur**2 + dyCur**2)
            drCur = rlocal - radNP
            sigma1 = np.exp(-(drCur/0.2)**2)
            if dist_vals[i] < radNP:
                stretch_val = np.abs(z)/(N*b**2)
                return stretch_val
            else:
                uzx += -2*(uRef*dxCur*drCur/(rlocal*0.2**2)) * sigma1
                uzy += -2*(uRef*dyCur*drCur/(rlocal*0.2**2)) * sigma1
                uzz += -sigma1
    stretch_val = ((1+uzz)/N)*np.sqrt((x/(a**2) + z*uzx/(b**2 * (1+uzz)))**2 +
                                      (y/(a**2) + z*uzy/(b**2 * (1+uzz)))**2 +
                                      (z/(b**2 * (1+uzz)))**2)

    return stretch_val


def calc_def_NP(x,y,z,a,b,z0Inner,xNP,yNP,nanopillars):
    radNP, hNP, pNP = nanopillars[:]
    xSteric = 0.05
    radNP += 2*xSteric # to avoid collision with PM
    hNP += 0.2
    zNP = hNP * np.ones_like(xNP)
    dist_vals = np.sqrt(np.power(xNP-x, 2) + np.power(yNP-y, 2)) 
    alpha = np.sqrt(1 - (x**2+y**2)/a**2)
    zCur = z0Inner - b*alpha
    uz = 0
    for i in range(len(xNP)):
        uRef = zNP[i] - zCur
        if z < zNP[i]+1e-6:
            dxCur = x - xNP[i]
            dyCur = y - yNP[i]
            rlocal = np.sqrt(dxCur**2 + dyCur**2)
            drCur = rlocal - radNP
            sigma1 = np.exp(-(drCur/0.2)**2)
            if dist_vals[i] < radNP:
                uz = uRef
                return uz
            else:
                uz += uRef*sigma1

    return uz

def compute_profile(thetaCur, rValsInnerRef, zValsInnerRef, sValsInnerRef, 
                    nanopillars, nuc_compression, innerParam, xTest, yTest):
    rValsInnerCur = rValsInnerRef.copy()
    zValsInnerCur = zValsInnerRef.copy()
    sValsInnerCur = sValsInnerRef.copy()
    xSteric = 0.05
    aInner, bInner, r0Inner, z0Inner = innerParam[:]
    nanopillar_rad, nanopillar_height, nanopillar_spacing = nanopillars[:]
    xMax = np.ceil(aInner / nanopillar_spacing) * nanopillar_spacing
    xNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    yNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    xNP, yNP = np.meshgrid(xNP, yNP)
    xNP = xNP.flatten()
    yNP = yNP.flatten()
    # first, compute current deformations based on u_nuc
    for i in range(len(rValsInnerCur)):
        xRef = rValsInnerCur[i]*np.cos(thetaCur)
        yRef = rValsInnerCur[i]*np.sin(thetaCur)
        zRef = zValsInnerCur[i]
        if nuc_compression <= 0:
            uCur = 0
        else:
            uCur = calc_def_NP(xRef, yRef, zRef, aInner, bInner, z0Inner, xNP, yNP, nanopillars)
        zValsInnerCur[i] += uCur#[2]
        if i > 0:
            dsCur = np.sqrt((rValsInnerCur[i]-rValsInnerCur[i-1])**2 +
                            (zValsInnerCur[i]-zValsInnerCur[i-1])**2)
            sValsInnerCur[i] = sValsInnerCur[i-1] + dsCur
    half_idx = int(np.floor(len(rValsInnerCur)/2))
    sFirstHalf = np.linspace(0.0, sValsInnerCur[half_idx], 21)
    sValsCur = sFirstHalf
    dsBig = sValsInnerCur[-1] / 200
    dsSmall = sValsInnerCur[-1] / 200
    # determine where we are in theta direction - where do we potentially intersect nanopillars?
    sortIdx = np.argsort(xTest)
    xTest = xTest[sortIdx]
    yTest = yTest[sortIdx]
    rCross = []
    for n in range(len(xTest)):
        # find distance from the center of nanopillar n
        dCur = np.sqrt((xTest[n]*np.sin(thetaCur)**2 - yTest[n]*np.sin(thetaCur)*np.cos(thetaCur))**2 +
                        (yTest[n]*np.cos(thetaCur)**2 - xTest[n]*np.sin(thetaCur)*np.cos(thetaCur))**2)
        rCur = xTest[n]*np.cos(thetaCur) + yTest[n]*np.sin(thetaCur)
        assert np.all(np.diff(rValsInnerCur[-1:half_idx:-1]) > 0) # required for interpolation here
        zOrigCur = np.interp(rCur, rValsInnerRef[-1:half_idx:-1], zValsInnerRef[-1:half_idx:-1]) # can interpolate because second half has r strictly decreasing (flip order for interp)
        if dCur < nanopillar_rad + 2*xSteric and zOrigCur < nanopillar_height + 0.2:
            # solve for r values associated with crossing over the edge of a nanopillar
            quad_roots = np.roots([1, -2*(xTest[n]*np.cos(thetaCur)+yTest[n]*np.sin(thetaCur)), 
                                xTest[n]**2 + yTest[n]**2 - (nanopillar_rad+2*xSteric)**2])
            if not np.all(np.isreal(quad_roots)):
                raise ValueError("These roots must be real-valued!")
            if np.sum(quad_roots < 0)==1: # if r is negative, this should correspond to central nanopillar
                assert np.isclose(np.abs(quad_roots[0]), np.abs(quad_roots[1])) # satisfied for central nanopillar
            elif np.sum(quad_roots < 0)==2: # then on the other side
                continue #raise ValueError("These roots must be positive!")
            if np.all(quad_roots > rValsInnerCur[half_idx+1]):
                continue
            else:
                rCross.append(quad_roots[0])
                rCross.append(quad_roots[1])
    rCross = np.array(rCross)
    rCross = np.sort(rCross)[-1::-1]
    sCross = np.zeros_like(rCross)
    # if not np.any(rCross < 0):
    #     print("Center pillar should be included, what...?")
    if len(rCross) == 0:
        # then no intersections with nanopillars for this value of theta
        num_lower = np.ceil((sValsInnerCur[-1]-sValsInnerCur[half_idx])/dsBig)
        sLower = np.linspace(sValsInnerCur[half_idx] + dsBig, sValsInnerCur[-1], int(num_lower))
        sValsCur = np.concatenate((sValsCur, sLower))
    for i in range(len(rCross)):
        if rCross[i] > rValsInnerCur[half_idx+1]: # first intersection occurs in middle of NP
            continue
        elif rCross[i] < 0: # central nanopillar
            continue
        else:
            sCross[i] = np.interp(rCross[i], rValsInnerCur[-1:half_idx:-1], sValsInnerCur[-1:half_idx:-1])
    for i in range(0,len(rCross),2):
        # define ranges of arc length over each nanopillar crossing region
        if rCross[i] > rValsInnerCur[half_idx+1]: # first intersection occurs in middle of NP
            assert np.all(np.diff(zValsInnerCur[-2:0:-1]) > 0) # exlude first and last entry for interpolation
            sFirstCross = np.interp(nanopillar_height+0.2, zValsInnerCur[-2:0:-1], sValsInnerCur[-2:0:-1])
            rFirstCross = np.interp(nanopillar_height+0.2, zValsInnerCur[-2:0:-1], rValsInnerCur[-2:0:-1])
            numBefore = np.floor(2*(sFirstCross - sValsCur[-1])/(dsSmall+dsBig))
            dsSumCur = 2*(sFirstCross - sValsCur[-1])/numBefore
            dsBigCur = dsSumCur - dsSmall
            delta_s = (dsBig-dsSmall)/(numBefore-1)
        else:
            numBefore = np.floor(2*(sCross[i] - sValsCur[-1])/(dsSmall+dsBig))
            dsSumCur = 2*(sCross[i] - sValsCur[-1])/numBefore
            dsBigCur = dsSumCur - dsSmall
            delta_s = (dsBigCur-dsSmall)/(numBefore-1)
            rFirstCross = rCross[i]
            sFirstCross = sCross[i]
        if numBefore > 1:
            sBefore = np.zeros([int(numBefore),])
            sBefore[0] = sValsCur[-1] + dsBigCur
            for n in range(1,int(numBefore)):
                sBefore[n] = sBefore[n-1] + (dsBigCur-n*delta_s)
        else:
            sBefore = np.array([sFirstCross])
        
        if rCross[i+1] < 0:
            numIn = np.ceil(rFirstCross/dsSmall)
            sIn = np.linspace(sFirstCross+dsSmall, sValsInnerCur[-1], int(numIn))
            sAfter = np.array([])
        else:
            numIn = np.ceil((rFirstCross-rCross[i+1])/dsSmall)
            sIn = np.linspace(sFirstCross+dsSmall, sCross[i+1], int(numIn))
            if len(rCross) < i+3: #then no central nanopillar!
                sMidAfter = sValsInnerCur[-1]
            else:
                sMidAfter = (sCross[i+1] + sCross[i+2])/2
            numAfter = np.floor(2*(sMidAfter - sCross[i+1])/(dsSmall+dsBig))
            dsSumCur = 2*(sMidAfter - sCross[i+1])/numAfter
            dsBigCur = dsSumCur - dsSmall
            delta_s = (dsBig-dsSmall)/(numAfter-1)
            if numAfter > 1:
                sAfter = np.zeros([int(numAfter),])
                sAfter[0] = sCross[i+1] + (dsSmall)
                for n in range(1,int(numAfter)):
                    sAfter[n] = sAfter[n-1] + (dsSmall+n*delta_s)
            else:
                sAfter = np.array([sMidAfter])
        sValsCur = np.concatenate((sValsCur, sBefore, sIn, sAfter))

    # interpolate r and z values at specified arc length values
    rValsCur = np.interp(sValsCur, sValsInnerCur, rValsInnerCur)
    zValsCur = np.interp(sValsCur, sValsInnerCur, zValsInnerCur)
    return (rValsCur, zValsCur)

def create_multicell(
    cubeSize: float = 100.0,
    locVec1: list = [[0, 0, 0]],
    locVec2: list = [],
    cellRad1: float = 10.0,
    cellRad2: float = 10.0,
    hCube: float = 0,
    hCell: float = 0,
    interface_marker1: int = 11,
    interface_marker2: int = 12,
    outer_marker: int = 10,
    extracell_tag: int = 1,
    cell_vol_tag1: int = 2,
    cell_vol_tag2: int = 3,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh with an outer cube containing embedded cells at specified locations.
    Cells can be of two types, type 1 and type 2 throughout.
    Args:
        cubeSize: Length of cube sides
        locVec1: list of cell 1 locations (each list element is a list [x,y,z])
        locVec2: list of cell 2 locations (each list element is a list [x,y,z])
        cellRad1: either a float, a list of floats, or a list of lists, each
                  containing a list of three floats for each axis [rx,ry,rz]
        cellRad2: either a float, a list of floats, or a list of lists, each
                  containing a list of three floats for each axis [rx,ry,rz]
        hCube: maximum mesh size for cube
        hCell: maximum mesh size for cell surface
        interface_marker1: The value to mark facets on cell 1 surface
        outer_marker: The value to mark facets on the surface of the cube
        extracell_tag: Tag for extracellular volume
        cell_vol_tag1: Tag for the volume of type 1 cells
        outer_vol_tag2: Tag for the volume of type 2 cells
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
    Returns:
        A triplet (mesh, facet_marker, cell_marker)
    """
    import gmsh

    if np.isclose(cubeSize, 0):
        ValueError("Outer cube size is equal to zero")
    if np.isclose(hCube, 0):
        hCube = 0.1 * max(cubeSize)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))

    gmsh.model.add("multicell")
    # first add outer cube
    cube = gmsh.model.occ.addBox(
        -cubeSize / 2, -cubeSize / 2, -cubeSize / 2, cubeSize, cubeSize, cubeSize
    )
    meanRad1 = 0
    meanRad2 = 0
    if (len(locVec1) == 0) and (len(locVec2) == 0):
        # Just a cube!
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(3, [cube], tag=extracell_tag)
        facets = gmsh.model.getBoundary([(3, cube)])
        gmsh.model.add_physical_group(2, [facets[0][1]], tag=outer_marker)
    else:
        # Add cells
        cell_list = []
        cellRad1Vec = []
        cellRad2Vec = []
        # first add source cell(s)
        for i in range(len(locVec1)):
            if isinstance(cellRad1, float) or isinstance(cellRad1, int):
                cellRadCur = float(cellRad1)
            elif len(locVec1) == 1 and len(cellRad1) == 3:
                cellRadCur = cellRad1
            elif len(cellRad1) == len(locVec1):
                cellRadCur = cellRad1[i]
            else:
                raise ValueError("Radii must be floats or lists of 3 floats")
            if isinstance(cellRadCur, float):
                cellRads = [cellRadCur, cellRadCur, cellRadCur]
            elif len(cellRadCur) == 3:
                cellRads = cellRadCur
            else:
                raise ValueError("Radii must be floats or lists of 3 floats")

            cur_tag = gmsh.model.occ.addSphere(locVec1[i][0], locVec1[i][1], locVec1[i][2], 1.0)
            gmsh.model.occ.dilate(
                [(3, cur_tag)],
                locVec1[i][0],
                locVec1[i][1],
                locVec1[i][2],
                cellRads[0],
                cellRads[1],
                cellRads[2],
            )
            cell_list.append((3, cur_tag))
            meanRad1 += np.mean(cellRads)
            cellRad1Vec.append(cellRads)
        if meanRad1 > 0:
            meanRad1 /= len(locVec1)
        # now add additional cells
        for i in range(len(locVec2)):
            if isinstance(cellRad2, float) or isinstance(cellRad2, int):
                cellRadCur = float(cellRad2)
            elif len(locVec2) == 1 and len(cellRad2) == 3:
                cellRadCur = cellRad2
            elif len(cellRad2) == len(locVec2):
                cellRadCur = cellRad2[i]
            else:
                raise ValueError("Radii must be floats or lists of 3 floats")
            if isinstance(cellRadCur, float):
                cellRads = [cellRadCur, cellRadCur, cellRadCur]
            elif len(cellRadCur) == 3:
                cellRads = cellRadCur
            else:
                raise ValueError("Radii must be floats or lists of 3 floats")

            cur_tag = gmsh.model.occ.addSphere(locVec2[i][0], locVec2[i][1], locVec2[i][2], 1.0)
            gmsh.model.occ.dilate(
                [(3, cur_tag)],
                locVec2[i][0],
                locVec2[i][1],
                locVec2[i][2],
                cellRads[0],
                cellRads[1],
                cellRads[2],
            )
            cell_list.append((3, cur_tag))
            meanRad2 = np.mean(cellRads)
            cellRad2Vec.append(cellRads)
        if meanRad2 > 0:
            meanRad2 /= len(locVec2)
        # define hCell if it was previously set to zero
        if np.isclose(hCell, 0):
            hCell = (
                0.2 * cubeSize
                if (meanRad1 == 0 and meanRad2 == 0)
                else 0.2 * max([meanRad1, meanRad2])
            )
        # Create interface between cells and extracell
        full_geo, maps = gmsh.model.occ.fragment([(3, cube)], cell_list)
        cube_map = maps[0]
        cell_maps = maps[1:]
        gmsh.model.occ.synchronize()

        # Get the outer boundary
        outer_shells = gmsh.model.getBoundary(full_geo, oriented=False)
        # Get the inner boundary
        inner_shells1 = []
        inner_shells2 = []
        for i in range(0, len(locVec1)):
            inner_shells1.append(gmsh.model.getBoundary(cell_maps[i], oriented=False))
        for i in range(len(locVec1), len(cell_maps)):
            inner_shells2.append(gmsh.model.getBoundary(cell_maps[i], oriented=False))
        # Add physical markers for facets
        gmsh.model.add_physical_group(2, [faces[1] for faces in outer_shells], tag=outer_marker)
        gmsh.model.add_physical_group(
            2, [faces[0][1] for faces in inner_shells1], tag=interface_marker1
        )
        gmsh.model.add_physical_group(
            2, [faces[0][1] for faces in inner_shells2], tag=interface_marker2
        )

        # Physical markers for all volumes
        all_volumes = [tag[1] for tag in cube_map]
        inner_volumes1 = [tag[0][1] for tag in cell_maps[0 : len(locVec1)]]
        inner_volumes2 = [tag[0][1] for tag in cell_maps[len(locVec1) :]]
        outer_volume = []
        for vol in all_volumes:
            if (vol not in inner_volumes1) and (vol not in inner_volumes2):
                outer_volume.append(vol)
        gmsh.model.add_physical_group(3, outer_volume, tag=extracell_tag)
        gmsh.model.add_physical_group(3, inner_volumes1, tag=cell_vol_tag1)
        gmsh.model.add_physical_group(3, inner_volumes2, tag=cell_vol_tag2)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM (defaults to 0.1*outerRad,
        # or set when calling function) and hInnerEdge at the ERM
        # (defaults to 0.2*innerRad, or set when calling function)
        # between these, the value is interpolated based on r (polar coord),
        # and inside the value is interpolated between hInnerEdge and 0.2*innerRad
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*outerRad in the center

        if len(locVec1) == 0 and len(locVec2) == 0:
            return hCube
        elif len(locVec1) == 0:
            cell_locs1 = [np.inf]
            cell_locs2 = np.sqrt(
                ((x - np.array(locVec2)[:, 0]) / np.array(cellRad2Vec)[:, 0]) ** 2
                + ((y - np.array(locVec2)[:, 1]) / np.array(cellRad2Vec)[:, 1]) ** 2
                + ((z - np.array(locVec2)[:, 2]) / np.array(cellRad2Vec)[:, 2]) ** 2
            )
        elif len(locVec2) == 0:
            cell_locs1 = np.sqrt(
                ((x - np.array(locVec1)[:, 0]) / np.array(cellRad1Vec)[:, 0]) ** 2
                + ((y - np.array(locVec1)[:, 1]) / np.array(cellRad1Vec)[:, 1]) ** 2
                + ((z - np.array(locVec1)[:, 2]) / np.array(cellRad1Vec)[:, 2]) ** 2
            )
            cell_locs2 = [np.inf]
        else:
            cell_locs1 = np.sqrt(
                ((x - np.array(locVec1)[:, 0]) / np.array(cellRad1Vec)[:, 0]) ** 2
                + ((y - np.array(locVec1)[:, 1]) / np.array(cellRad1Vec)[:, 1]) ** 2
                + ((z - np.array(locVec1)[:, 2]) / np.array(cellRad1Vec)[:, 2]) ** 2
            )
            cell_locs2 = np.sqrt(
                ((x - np.array(locVec2)[:, 0]) / np.array(cellRad2Vec)[:, 0]) ** 2
                + ((y - np.array(locVec2)[:, 1]) / np.array(cellRad2Vec)[:, 1]) ** 2
                + ((z - np.array(locVec2)[:, 2]) / np.array(cellRad2Vec)[:, 2]) ** 2
            )
        closest_cell1 = min(cell_locs1)
        closest_cell2 = min(cell_locs2)
        if (closest_cell1 < 1.0) or (closest_cell2 < 1.0):
            return hCell
        else:
            if closest_cell1 < closest_cell2:
                cellWeight = np.exp(-(closest_cell1 - 1.0) / 0.1)
            else:
                cellWeight = np.exp(-(closest_cell2 - 1.0) / 0.1)
            return hCell * cellWeight + hCube * (1 - cellWeight)

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(3)
    rank = MPI.COMM_WORLD.rank
    tmp_folder = pathlib.Path(f"tmp_extracell_{cubeSize}_{cellRad1}_{cellRad2}_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "extracell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    return (dmesh, mf2, mf3)
