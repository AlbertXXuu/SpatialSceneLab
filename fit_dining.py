"""Fit rectangular dining furniture to even-frame observations and box/floor priors.

Ownership candidates recover visible geometry at competing prior boxes. The input
owner split is always retained in full for validation. No reference asset is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from fit_stool import cylinder_sdf, cylinder_mesh, least_squares, sha256, stats, validate_output, write_ply

TARGETS = ("same-table", "same-chair-west", "same-chair-east")
ASSUMPTIONS = [
    "Operator-selected rectangular table and rectangular seat/back chair families with four straight round supports each.",
    "Y is up; units are metres. A measured Floor upper plane anchors support feet.",
    "Even frames alone determine geometry, candidate selection and color. All originally owner-assigned odd-frame observations remain in validation.",
    "The supplied candidate prior boxes overlap. Robust proximity to fitted components selects candidate training evidence, reported separately from original ownership.",
    "Closed hidden faces and continuation to intended joints are structural completion. Point agreement does not identify unseen geometry uniquely.",
    "The final cushioned model keeps rectangular seat/back bases and fits separately closed convex half-ellipsoid caps to observed curved surfaces. Rectangular and domed slab controls are retained as failed method attempts.",
    "Seat cap fitting uses localized original-owner even points to avoid competing table geometry; candidate even points recover supports hidden by ownership competition. This is a bounded synthetic development case.",
]


def yaw_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def box_sdf(points, center, dimensions, rotation=None):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    q = np.abs((np.asarray(points) - center) @ rotation) - np.asarray(dimensions) / 2
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)


def domed_sdf(points, center, dimensions, cap_axis, cap_sign, cap_height, power=.5, rotation=None):
    """First-order signed cap residual plus exact rectangular side/bottom residual.

    The exported triangles, evaluated independently, define acceptance distances.
    """
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    local = (np.asarray(points) - center) @ rotation
    tangents = [i for i in range(3) if i != cap_axis]
    a, b = tangents
    ra, rb = dimensions[a] / 2, dimensions[b] / 2
    ua = np.clip(local[:, a] / ra, -.999999, .999999)
    ub = np.clip(local[:, b] / rb, -.999999, .999999)
    fa, fb = (1 - ua ** 2) ** power, (1 - ub ** 2) ** power
    crown = cap_height * fa * fb
    ga = -cap_height * 2 * power * ua / ra * (1 - ua ** 2) ** (power - 1) * fb
    gb = -cap_height * 2 * power * ub / rb * (1 - ub ** 2) ** (power - 1) * fa
    top = (cap_sign * local[:, cap_axis] - dimensions[cap_axis] / 2 - crown) / np.sqrt(1 + ga ** 2 + gb ** 2)
    q = np.column_stack((np.abs(local[:, a]) - ra, np.abs(local[:, b]) - rb,
                         -cap_sign * local[:, cap_axis] - dimensions[cap_axis] / 2, top))
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)


def part_distance(points, part):
    p = part["parameters"]
    if part["primitive_type"] == "capped_cylinder":
        return cylinder_sdf(points, p["start_m"], p["end_m"], p["radius_m"])
    if part["primitive_type"] == "box":
        return box_sdf(points, p["center_m"], p["dimensions_m"], p["rotation_world_3x3"])
    if part["primitive_type"] == "half_ellipsoid":
        return half_ellipsoid_residual(points, p["center_m"], p["radii_m"], p["rotation_world_3x3"], p["cap_axis"], p["cap_sign"])
    return domed_sdf(points, p["center_m"], p["dimensions_m"], p["cap_axis"], p["cap_sign"],
                     p["cap_height_m"], p["profile_power"], p["rotation_world_3x3"])


def component(name, role, primitive, **parameters):
    return {"name": name, "type": primitive, "primitive_type": primitive, "role": role,
            "parameters": parameters, "provenance_role": "fitted/structural", "editable": True}


def load_inputs(scene_path, observations_path):
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    if scene.get("units") != "m" or scene.get("up_axis") != "Y":
        raise ValueError("Requires Y-up metre scene inputs")
    records = scene.get("objects")
    if not isinstance(records, list) or not records:
        raise ValueError("Scene requires object boxes")
    for record in records + scene.get("structures", []):
        dims = np.asarray(record.get("dimensions_m"), float)
        transform = np.asarray(record.get("transform_world"), float)
        if dims.shape != (3,) or np.any(dims <= 0) or not np.isfinite(dims).all():
            raise ValueError("All prior dimensions must be finite positive triples")
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("All transforms must be finite 4x4")
        r = transform[:3, :3]
        if not (np.allclose(r.T @ r, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(r), 1, atol=1e-6)
                and np.allclose(r[:, 1], [0, 1, 0], atol=1e-6) and np.allclose(transform[3], [0, 0, 0, 1])):
            raise ValueError("Prior boxes must be rigid upright transforms")
    floors = [o for o in scene.get("structures", []) if o.get("category") == "Floor"]
    if not floors:
        raise ValueError("A measured Floor prior is required")
    floor = max(o["transform_world"][1][3] + o["dimensions_m"][1] / 2 for o in floors)
    frame_records = scene.get("frames", [])
    frame_ids = [f.get("frame_index") for f in frame_records]
    if not frame_ids or any(type(v) is not int or v < 0 for v in frame_ids) or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Scene frames need unique nonnegative integer indices")
    with np.load(observations_path, allow_pickle=False) as raw:
        required = ("points", "colors", "frame_index", "owner", "candidates")
        if any(k not in raw.files for k in required):
            raise ValueError("Observations require points/colors/frame_index/owner/candidates")
        data = {k: raw[k].copy() for k in required}
    n = len(data["points"])
    for key, shape in (("points", (n, 3)), ("colors", (n, 3)), ("frame_index", (n,)), ("owner", (n,)), ("candidates", (n, len(records)))):
        a = data[key]
        if a.shape != shape or np.iscomplexobj(a) or not np.isfinite(a).all():
            raise ValueError(f"Invalid {key} array shape/type/values")
    for key in ("frame_index", "owner"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f"{key} must contain integers")
    if not np.isin(data["frame_index"], frame_ids).all() or np.any(data["owner"] < -2) or np.any(data["owner"] >= len(records)):
        raise ValueError("Invalid observation frame/owner IDs")
    if np.any(data["colors"] < 0) or np.any(data["colors"] > 255) or np.any(data["colors"] != np.floor(data["colors"])):
        raise ValueError("Colors must be integer RGB in [0,255]")
    if not np.isin(data["candidates"], [0, 1]).all():
        raise ValueError("Candidates must be binary")
    data["candidates"] = data["candidates"].astype(bool)
    targets = []
    for uuid in TARGETS:
        matches = [(i, o) for i, o in enumerate(records) if o.get("uuid") == uuid]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {uuid} prior")
        index, obj = matches[0]
        if obj.get("category") != ("Table" if uuid == TARGETS[0] else "Chair"):
            raise ValueError("Target category disagrees with selected family")
        train = (data["frame_index"] % 2 == 0) & data["candidates"][:, index]
        valid = (data["frame_index"] % 2 == 1) & (data["owner"] == index)
        if np.sum(train) < 100 or np.sum(valid) < 20:
            raise ValueError("Insufficient even candidate/odd owner observations")
        targets.append((index, obj))
    return scene, data, targets, float(floor)


def sample(points, cap=4000):
    if len(points) <= cap:
        return points
    return points[np.random.default_rng(271828).choice(len(points), cap, replace=False)]


def fit_slab(points, initial, cap_axis=1, cap_sign=1, dome=False):
    """Fit one slab from a localized even-frame point pool with robust loss."""
    initial = np.asarray(initial, float)  # center xyz, dimensions xyz, yaw, crown
    lower = initial.copy(); upper = initial.copy()
    lower[:3] -= [.06, .06, .06]; upper[:3] += [.06, .06, .06]
    lower[3:6] *= [.7, .55, .7]; upper[3:6] *= [1.2, 1.9, 1.2]
    lower[6] -= .14; upper[6] += .14
    lower[7] = .003 if dome else 0.; upper[7] = .12 if dome else 1e-8
    initial[7] = np.clip(initial[7], lower[7] + 1e-10, upper[7] - 1e-10)
    selected = sample(points)
    def residual(x):
        r = yaw_matrix(x[6])
        if dome:
            return domed_sdf(selected, x[:3], x[3:6], cap_axis, cap_sign, x[7], rotation=r)
        return box_sdf(selected, x[:3], x[3:6], r)
    result = least_squares(residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=.003,
                           x_scale="jac", max_nfev=160)
    return result.x, {"success": bool(result.success), "cost": float(result.cost), "nfev": int(result.nfev),
                      "point_count": len(points), "sample_count": len(selected)}


def fit_support(points, center, floor, top, radius_guess):
    near = points[np.linalg.norm(points[:, [0, 2]] - center, axis=1) < radius_guess * 2.8]
    near = near[(near[:, 1] > floor + .018) & (near[:, 1] < top - .035)]
    if len(near) < 10:
        raise ValueError("Support has fewer than ten localized candidate observations")
    x0 = np.array([*center, *center, radius_guess])
    lo = x0 - [.06, .06, .09, .09, radius_guess * .65]
    hi = x0 + [.06, .06, .09, .09, radius_guess]
    def primitive(x):
        bx, bz, tx, tz, radius = x
        foot = floor
        horizontal = np.hypot(tx - bx, tz - bz)
        for _ in range(8):
            foot = floor + radius * horizontal / np.hypot(horizontal, top - foot)
        return np.array([bx, foot, bz]), np.array([tx, top, tz]), radius
    def residual(x):
        a, b, r = primitive(x)
        return cylinder_sdf(near, a, b, r)
    result = least_squares(residual, x0, bounds=(lo, hi), loss="soft_l1", f_scale=.0025,
                           x_scale="jac", max_nfev=180)
    a, b, r = primitive(result.x)
    return a, b, r, {"success": bool(result.success), "candidate_pool_count": len(near), "cost": float(result.cost)}


def fit_rail(points, start, end, radius=.02):
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = (end - start) / np.linalg.norm(end - start)
    near = points[(np.linalg.norm(np.cross(points - start, axis), axis=1) < .045) &
                  ((points - start) @ axis > -.02) & ((points - end) @ axis < .02)]
    if len(near) < 8:
        return None
    # Retain structural endpoints on the adjoining support axes; fit perpendicular translation/radius.
    seed = np.array([0., 1., 0.]) if abs(axis[1]) < .8 else np.array([1., 0., 0.])
    u = np.cross(axis, seed); u /= np.linalg.norm(u); v = np.cross(axis, u)
    def residual(x):
        offset = x[0] * u + x[1] * v
        return cylinder_sdf(near, start + offset, end + offset, x[2])
    result = least_squares(residual, [0., 0., radius], bounds=([-.025, -.025, .007], [.025, .025, .035]),
                           loss="soft_l1", f_scale=.002, max_nfev=100)
    offset = result.x[0] * u + result.x[1] * v
    return start + offset, end + offset, float(result.x[2])


def half_ellipsoid_residual(points, center, radii, rotation, axis, sign):
    local=(np.asarray(points)-center)@rotation
    radii=np.asarray(radii)
    implicit=np.sum((local/radii)**2,axis=1)-1
    gradient=2*np.linalg.norm(local/radii**2,axis=1)
    surface=implicit/np.maximum(gradient,1e-8)
    # Near the visible cap this is a first-order Euclidean surface residual.
    # Inside overlap the nearest component surface is independently evaluated.
    return np.maximum(surface,-sign*local[:,axis])


def fit_cushion(points, center, radii, rotation, axis, sign):
    center=np.asarray(center,float);radii=np.asarray(radii,float)
    local=(points-center)@rotation
    # Separate the curved cap from the exposed planar base before fitting.
    near=points[(sign*local[:,axis]>.012)&(np.sum((local/radii)**2,axis=1)<1.65)]
    initial=np.concatenate((center,radii))
    lo=initial.copy();hi=initial.copy();lo[:3]-=.025;hi[:3]+=.025
    lo[axis]=center[axis]-.004;hi[axis]=center[axis]+.004
    lo[3:]*=.7;hi[3:]*=1.35
    def residual(x):
        q=(near-x[:3])@rotation
        implicit=np.sum((q/x[3:])**2,axis=1)-1
        return implicit/np.maximum(2*np.linalg.norm(q/x[3:]**2,axis=1),1e-8)
    result=least_squares(residual,initial,bounds=(lo,hi),loss='soft_l1',f_scale=.0015,x_scale='jac',max_nfev=180)
    return result.x[:3],result.x[3:],{"success":bool(result.success),"candidate_pool_count":len(near),"cost":float(result.cost)}


def fit_object(record, training, floor, surface_model, owner_training=None):
    t = np.asarray(record["transform_world"]); rotation = t[:3, :3]
    origin = t[:3, 3].copy(); origin[1] = 0
    local = (training - origin) @ rotation
    owned_local = local if owner_training is None else (owner_training - origin) @ rotation
    dims = np.asarray(record["dimensions_m"])
    table = record["category"] == "Table"
    height = dims[1]
    optimizers, parts, joints = {}, [], []
    seat_cushion = None
    if table:
        high = local[local[:, 1] > floor + height * .80]
        top = float(np.median(high[high[:, 1] > np.quantile(high[:, 1], .55), 1]))
        thick = top - float(np.quantile(high[:, 1], .03))
        xspan = np.quantile(high[:, 0], [.015, .985]); zspan = np.quantile(high[:, 2], [.015, .985])
        slab = [(xspan[0]+xspan[1])/2, top-thick/2, (zspan[0]+zspan[1])/2,
                xspan[1]-xspan[0], thick, zspan[1]-zspan[0], 0., 0.]
        x, optimizers["top"] = fit_slab(high, slab)
        parts.append(component("top", "top", "box", center_m=x[:3], dimensions_m=x[3:6], rotation_world_3x3=yaw_matrix(x[6])))
        low = local[(local[:, 1] < top-thick-.045) & (np.abs(local[:, 0]) > x[3]*.32) & (np.abs(local[:, 2]) > x[5]*.25)]
        support_top = top-thick/2
        guesses = [(np.median(low[(low[:, 0]*sx>0)&(low[:, 2]*sz>0), 0]),
                    np.median(low[(low[:, 0]*sx>0)&(low[:, 2]*sz>0), 2])) for sx,sz in ((-1,-1),(-1,1),(1,-1),(1,1))]
        radius = min(x[3],x[5])*.032
    else:
        seat_y = floor + height * .49
        seat_pool = local[(local[:, 1] > floor+height*.38)&(local[:, 1] < floor+height*.59)]
        slab = [np.median(seat_pool[:,0]), seat_y-.025, 0., dims[0]*.88, height*.055, dims[2]*.84, 0., height*.055]
        # Keep the seat center tied to perimeter evidence rather than uneven visible density.
        slab[0] = (np.quantile(seat_pool[:,0],.01)+np.quantile(seat_pool[:,0],.99))/2
        slab_pool=seat_pool
        if surface_model=="cushioned":
            counts,edges=np.histogram(seat_pool[:,1],bins=100)
            rim=float((edges[np.argmax(counts)]+edges[np.argmax(counts)+1])/2)
            bottom=float(np.quantile(seat_pool[:,1],.015))
            slab[1]=(rim+bottom)/2;slab[4]=rim-bottom
            slab_pool=seat_pool[seat_pool[:,1]<rim+.003]
        x, optimizers["seat"] = fit_slab(slab_pool, slab, dome=surface_model=="domed")
        kind = "domed_box" if surface_model=="domed" else "box"
        params = dict(center_m=x[:3],dimensions_m=x[3:6],rotation_world_3x3=yaw_matrix(x[6]))
        if kind == "domed_box": params.update(cap_axis=1,cap_sign=1,cap_height_m=x[7],profile_power=.5)
        parts.append(component("seat","seat",kind,**params))
        if surface_model=="cushioned":
            center=np.array([x[0],x[1]+x[4]/2-.002,x[2]])
            cap_pool=owned_local[(owned_local[:,1]>center[1])&(owned_local[:,1]<center[1]+height*.12)]
            radii=np.array([x[3]*.45,max(.015,np.quantile(cap_pool[:,1],.98)-center[1]),x[5]*.45])
            c,r,optimizers["seat_cushion"]=fit_cushion(cap_pool,center,radii,yaw_matrix(x[6]),1,1)
            seat_cushion=component("seat_cushion","cushion","half_ellipsoid",center_m=c,radii_m=r,
                                   rotation_world_3x3=yaw_matrix(x[6]),cap_axis=1,cap_sign=1)
        support_top = x[1]
        low = local[(local[:,1]>floor+.018)&(local[:,1]<support_top-.06)&(abs(local[:,0])<x[3]*.48)&(abs(local[:,2])<x[5]*.55)]
        guesses = [(np.median(low[(low[:,0]*sx>0)&(low[:,2]*sz>0),0]),
                    np.median(low[(low[:,0]*sx>0)&(low[:,2]*sz>0),2])) for sx,sz in ((-1,-1),(-1,1),(1,-1),(1,1))]
        radius = min(x[3],x[5])*.047
    for i,guess in enumerate(guesses):
        a,b,r,optimizers[f"leg_{i+1}"] = fit_support(local,guess,floor,support_top,radius)
        parts.append(component(f"leg_{i+1}","floor_support","capped_cylinder",start_m=a,end_m=b,radius_m=r))
        joints.append({"parts":[parts[0]["name"],f"leg_{i+1}"]})
    if table:
        # Observation-derived horizontal height peak, excluding vertical supports.
        mid = local[(local[:,1]>floor+height*.12)&(local[:,1]<floor+height*.6)&(abs(local[:,0])<x[3]*.33)&(abs(local[:,2])<x[5]*.12)]
        counts, edges = np.histogram(mid[:,1],bins=60)
        rail_y = float((edges[np.argmax(counts)]+edges[np.argmax(counts)+1])/2)
        centers=[]
        for side,pair in enumerate(((1,2),(3,4))):
            ends=[]
            for pi in pair:
                p=parts[pi]["parameters"]; a,b=np.asarray(p["start_m"]),np.asarray(p["end_m"])
                ends.append(a+(b-a)*(rail_y-a[1])/(b[1]-a[1]))
            fitted=fit_rail(local,*ends)
            if fitted:
                a,b,r=fitted; name=f"cross_rail_{side+1}"
                parts.append(component(name,"rail","capped_cylinder",start_m=a,end_m=b,radius_m=r))
                centers.append((a+b)/2)
                joints.extend({"parts":[name,parts[pi]["name"]]} for pi in pair)
        if len(centers)==2:
            fitted=fit_rail(local,*centers)
            if fitted:
                a,b,r=fitted; parts.append(component("center_rail","rail","capped_cylinder",start_m=a,end_m=b,radius_m=r))
                joints.extend({"parts":["center_rail",f"cross_rail_{i}"]} for i in (1,2))
    else:
        if seat_cushion is not None:
            parts.append(seat_cushion);joints.append({"parts":["seat_cushion","seat"]})
        back_pool=local[(local[:,1]>floor+height*.61)&(local[:,2]<-dims[2]*.22)&(abs(local[:,0])<dims[0]*.41)]
        back_top=float(np.quantile(back_pool[:,1],.98)); back_bottom=float(np.quantile(back_pool[:,1],.03))
        rear=float(np.quantile(back_pool[:,2],.10)); front=float(np.quantile(back_pool[:,2],.88))
        base_depth=max(.015,(front-rear)*.48)
        slab=[x[0],(back_top+back_bottom)/2,front-base_depth/2,x[3]*.82,back_top-back_bottom,base_depth,0.,max(.012,(front-rear)*.52)]
        back_slab_pool=back_pool
        if surface_model=="cushioned":
            # A rear bowed cap and a dense planar front are distinct surfaces.
            counts,edges=np.histogram(back_pool[:,2],bins=100)
            midpoint=np.median(back_pool[:,2]);eligible=(edges[:-1]>midpoint)
            front_mode=float((edges[np.argmax(np.where(eligible,counts,-1))]+edges[np.argmax(np.where(eligible,counts,-1))+1])/2)
            slab[2]=front_mode-base_depth/2
            back_slab_pool=back_pool[back_pool[:,2]>front_mode-.007]
        back,optimizers["back"]=fit_slab(back_slab_pool,slab,cap_axis=2,cap_sign=-1,dome=surface_model=="domed")
        params=dict(center_m=back[:3],dimensions_m=back[3:6],rotation_world_3x3=yaw_matrix(back[6]))
        if kind=="domed_box":params.update(cap_axis=2,cap_sign=-1,cap_height_m=back[7],profile_power=.5)
        parts.append(component("back","back",kind,**params))
        if surface_model=="cushioned":
            center=back[:3].copy();center[2]-=back[5]/2-.002
            radii=np.array([back[3]/2,back[4]/2,max(.015,center[2]-np.quantile(back_pool[:,2],.02))])
            c,r,optimizers["back_cushion"]=fit_cushion(back_pool,center,radii,yaw_matrix(back[6]),2,-1)
            parts.append(component("back_cushion","cushion","half_ellipsoid",center_m=c,radii_m=r,
                                   rotation_world_3x3=yaw_matrix(back[6]),cap_axis=2,cap_sign=-1))
            joints.append({"parts":["back_cushion","back"]})
        back_posts=[]
        for i in (1,3):
            leg=parts[i]["parameters"]; start=np.asarray(leg["end_m"]).copy(); start[1]-=.012
            high=local[(local[:,1]>support_top+.06)&(abs(local[:,0]-start[0])<.045)&(local[:,2]<-dims[2]*.22)]
            post_top=float(np.quantile(high[:,1],.995))
            a,b,r,optimizers[f"back_post_{i}"]=fit_support(high,start[[0,2]],start[1],post_top,leg["radius_m"])
            a[1]=start[1]
            name=f"back_post_{len(back_posts)+1}";back_posts.append(name)
            parts.append(component(name,"rail","capped_cylinder",start_m=a,end_m=b,radius_m=r))
            joints.extend(({"parts":[name,"seat"]},{"parts":[name,parts[i]["name"]]},{"parts":[name,"back"]}))
        # A visible bottom back member is separately editable.
        rail_y=back[1]-back[4]/2+.006
        endpoints=[]
        for name in back_posts:
            p=next(p for p in parts if p["name"]==name)["parameters"];a,b=np.asarray(p["start_m"]),np.asarray(p["end_m"])
            endpoints.append(a+(b-a)*(rail_y-a[1])/(b[1]-a[1]))
        fitted=fit_rail(local,*endpoints,radius=.018)
        if fitted:
            a,b,r=fitted;parts.append(component("back_rail","rail","capped_cylinder",start_m=a,end_m=b,radius_m=r))
            joints.extend({"parts":["back_rail",name]} for name in back_posts)
            joints.append({"parts":["back_rail","back"]})
    # Export everything in a single explicit world Y-up coordinate convention.
    for p in parts:
        params=p["parameters"]
        if p["primitive_type"]=="capped_cylinder":
            for key in ("start_m","end_m"):params[key]=(np.asarray(params[key])@rotation.T+origin).tolist()
            params["radius_m"]=float(params["radius_m"])
        else:
            params["center_m"]=(np.asarray(params["center_m"])@rotation.T+origin).tolist()
            if "dimensions_m" in params:params["dimensions_m"]=np.asarray(params["dimensions_m"]).tolist()
            if "radii_m" in params:params["radii_m"]=np.asarray(params["radii_m"]).tolist()
            params["rotation_world_3x3"]=(rotation@params["rotation_world_3x3"]).tolist()
            if p["primitive_type"]=="domed_box":params["cap_height_m"]=float(params["cap_height_m"])
    return parts,joints,optimizers


def slab_mesh(parameters, grid=48):
    center=np.asarray(parameters["center_m"]); dimensions=np.asarray(parameters["dimensions_m"])
    rotation=np.asarray(parameters["rotation_world_3x3"])
    axis=parameters.get("cap_axis",1); sign=parameters.get("cap_sign",1)
    height=parameters.get("cap_height_m",0.); power=parameters.get("profile_power",.5)
    tangent=[i for i in range(3) if i!=axis];a,b=tangent
    vertices=[]
    for i in range(grid+1):
        u=-1+2*i/grid
        for j in range(grid+1):
            v=-1+2*j/grid;p=np.zeros(3);p[a]=u*dimensions[a]/2;p[b]=v*dimensions[b]/2
            p[axis]=sign*(dimensions[axis]/2+height*((1-u*u)*(1-v*v))**power)
            vertices.append(p)
    faces=[]
    for i in range(grid):
        for j in range(grid):
            k=i*(grid+1)+j;l=k+grid+1;faces.extend(((k,l,k+1),(k+1,l,l+1)))
    ring=[j for j in range(grid+1)]+[i*(grid+1)+grid for i in range(1,grid+1)]+[grid*(grid+1)+j for j in range(grid-1,-1,-1)]+[i*(grid+1) for i in range(grid-1,0,-1)]
    offset=len(vertices)
    for k in ring:
        p=np.asarray(vertices[k]).copy();p[axis]=-sign*dimensions[axis]/2;vertices.append(p)
    mid=len(vertices);p=np.zeros(3);p[axis]=-sign*dimensions[axis]/2;vertices.append(p)
    for i,k in enumerate(ring):
        j=(i+1)%len(ring);faces.extend(((k,offset+i,ring[j]),(ring[j],offset+i,offset+j),(mid,offset+j,offset+i)))
    vertices=np.asarray(vertices);faces=np.asarray(faces)
    faces=outward_faces(vertices,faces,np.zeros(3))
    return vertices@rotation.T+center,faces


def outward_faces(vertices,faces,interior):
    faces=np.asarray(faces).copy()
    normal=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    inward=np.einsum('ij,ij->i',normal,np.mean(vertices[faces],axis=1)-interior)<0
    faces[inward]=faces[inward][:,[0,2,1]]
    return faces


def box_mesh(parameters):
    dimensions=np.asarray(parameters['dimensions_m']);center=np.asarray(parameters['center_m'])
    rotation=np.asarray(parameters['rotation_world_3x3'])
    v=np.array([[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)],float)*dimensions/2
    f=np.array([(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
                (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)])
    return v@rotation.T+center,outward_faces(v,f,np.zeros(3))


def half_ellipsoid_mesh(parameters,rings=32,slices=96):
    r=np.asarray(parameters['radii_m']);axis=parameters['cap_axis'];sign=parameters['cap_sign']
    tangents=[i for i in range(3) if i!=axis];a,b=tangents
    vertices=[]
    for j in range(rings):
        phi=(np.pi/2)*j/rings
        for i in range(slices):
            theta=2*np.pi*i/slices;p=np.zeros(3)
            p[a]=r[a]*np.cos(phi)*np.cos(theta);p[b]=r[b]*np.cos(phi)*np.sin(theta)
            p[axis]=sign*r[axis]*np.sin(phi);vertices.append(p)
    pole=len(vertices);p=np.zeros(3);p[axis]=sign*r[axis];vertices.append(p)
    bottom=len(vertices);vertices.append(np.zeros(3));faces=[]
    for j in range(rings-1):
        for i in range(slices):
            n=(i+1)%slices;k=j*slices+i;l=j*slices+n
            faces.extend(((k,l,k+slices),(l,l+slices,k+slices)))
    for i in range(slices):
        n=(i+1)%slices;faces.extend(((pole,(rings-1)*slices+i,(rings-1)*slices+n),(bottom,n,i)))
    v=np.asarray(vertices);f=outward_faces(v,np.asarray(faces),np.eye(3)[axis]*sign*r[axis]*.2)
    return v@np.asarray(parameters['rotation_world_3x3']).T+parameters['center_m'],f


def measurements(points,frames,parts):
    distance=np.column_stack([np.abs(part_distance(points,p)) for p in parts])
    assigned=np.argmin(distance,axis=1); error=np.min(distance,axis=1)
    return {"point_count_weighted":stats(error),"per_closest_part":{
        p["name"]:{**stats(error[assigned==i]),"within_10mm_support":int(np.sum((assigned==i)&(error<=.010)))}
        for i,p in enumerate(parts)},"per_frame":{str(int(f)):stats(error[frames==f]) for f in np.unique(frames)}},assigned,error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",type=Path,required=True);parser.add_argument("--observations",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--surface-model",choices=("rectangular","domed","cushioned"),default="cushioned")
    args=parser.parse_args();started=time.time()
    try:
        validate_output(args.output,[args.scene,args.observations])
        input_hashes={p:sha256(p) for p in (args.scene,args.observations)}
        scene,data,targets,floor=load_inputs(args.scene,args.observations)
        if any(sha256(p)!=digest for p,digest in input_hashes.items()):
            raise ValueError("Input changed while reading; use unchanged capture files")
        objects=[]
        for index,record in targets:
            train=(data["frame_index"]%2==0)&data["candidates"][:,index]
            owned_train=(data["frame_index"]%2==0)&(data["owner"]==index)
            parts,joints,optimization=fit_object(record,data["points"][train],floor,args.surface_model,data["points"][owned_train])
            valid=(data["frame_index"]%2==1)&(data["owner"]==index)
            candidate_valid=(data["frame_index"]%2==1)&data["candidates"][:,index]
            candidate_metrics,assigned,error=measurements(data["points"][train],data["frame_index"][train],parts)
            selected=error<=.015
            training,_,_=measurements(data["points"][train][selected],data["frame_index"][train][selected],parts)
            heldout,_,_=measurements(data["points"][valid],data["frame_index"][valid],parts)
            owner_training,_,_=measurements(data["points"][owned_train],data["frame_index"][owned_train],parts)
            candidate_heldout,_,_=measurements(data["points"][candidate_valid],data["frame_index"][candidate_valid],parts)
            objects.append({"uuid":record["uuid"],"category":record["category"],"source_owner_index":index,
                            "transform_world":record["transform_world"],"components":parts,"optimizer":optimization,
                            "mesh":record["uuid"]+".ply","metrics":{"train":training,"validation":heldout,
                            "owner_train_all":owner_training,"candidate_train_all":candidate_metrics,"candidate_validation_all":candidate_heldout},
                            "observation_selection":{"candidate_train_count":int(np.sum(train)),"selected_train_count":int(np.sum(selected)),
                            "owned_train_count":int(np.sum(owned_train)),"owner_validation_count":int(np.sum(valid)),"owner_validation_excluded_count":0,
                            "selected_extra_owner_count":int(np.sum(data["owner"][train][selected]!=index)),
                            "selected_global_even_index_sha256":hashlib.sha256(np.flatnonzero(train)[selected].astype('<i8').tobytes()).hexdigest(),
                            "selection_distance_m":.015,"criterion":"Even candidate points within 15mm of fitted component surface; full candidate/owner metrics also reported"},
                            "contact_floor":{"floor_y_m":floor,"joints":joints,"feet":[{"name":p["name"]} for p in parts if p["role"]=="floor_support"]}})
    except (ValueError,OSError,RuntimeError) as error:
        parser.error(str(error))
    validate_output(args.output,[args.scene,args.observations]);args.output.mkdir(parents=True,exist_ok=True)
    passed=True
    for obj in objects:
        train=(data["frame_index"]%2==0)&data["candidates"][:,obj["source_owner_index"]]
        _,assignment,_=measurements(data["points"][train],data["frame_index"][train],obj["components"])
        colors=data["colors"][train];fallback=np.median(colors,axis=0).astype(int)
        palette=[np.median(colors[assignment==i],axis=0).astype(int) if np.any(assignment==i) else fallback for i in range(len(obj["components"]))]
        vertices=[];faces=[];vp=[];fp=[];offset=0
        for i,part in enumerate(obj["components"]):
            if part["primitive_type"]=="capped_cylinder":
                p=part["parameters"];v,f=cylinder_mesh({"start":np.asarray(p["start_m"]),"end":np.asarray(p["end_m"]),"radius":p["radius_m"]})
            elif part["primitive_type"]=="half_ellipsoid":v,f=half_ellipsoid_mesh(part["parameters"])
            elif part["primitive_type"]=="box":v,f=box_mesh(part["parameters"])
            else:v,f=slab_mesh(part["parameters"])
            part["mesh"]=obj["uuid"]+"-"+part["name"]+".ply";part["color_rgb"]=palette[i].tolist()
            part["training_closest_point_support"]=obj["metrics"]["train"]["per_closest_part"][part["name"]]["within_10mm_support"]
            part["vertex_count"]=len(v);part["face_count"]=len(f)
            write_ply(args.output/part["mesh"],v,f,np.full(len(v),i),np.full(len(f),i),palette)
            vertices.extend(v);faces.extend(f+offset);vp.extend([i]*len(v));fp.extend([i]*len(f));offset+=len(v)
            if part["role"]=="floor_support":
                foot=next(x for x in obj["contact_floor"]["feet"] if x["name"]==part["name"])
                foot.update(lowest_vertex_y_m=float(np.min(v[:,1])),gap_m=float(np.min(v[:,1])-floor))
        write_ply(args.output/obj["mesh"],np.asarray(vertices),np.asarray(faces),vp,fp,palette)
        metric=obj["metrics"]["validation"]["point_count_weighted"]
        obj["quality_gate"]={"passed":bool(metric["p95_m"]<=.010 and metric["within_10mm_fraction"]>=.95 and all(p["training_closest_point_support"]>=5 for p in obj["components"])),
            "validation_p95_limit_m":.010,"validation_within_10mm_minimum_fraction":.95,"minimum_training_support_per_component":5,
            "distance_role":"analytic primitive/cap approximation; independent exported triangle acceptance required"}
        passed=passed and obj["quality_gate"]["passed"]
    if any(sha256(p)!=digest for p,digest in input_hashes.items()):
        parser.error("Input changed during fitting; result is not accepted")
    result={"schema":"alvenx.spatial.dining-fit.v1","units":"m","up_axis":"Y","objects":objects,"assumptions":ASSUMPTIONS,
            "surface_model":args.surface_model,"input_files":{"scene":{"path":str(args.scene.resolve()),"sha256":input_hashes[args.scene]},
            "observations":{"path":str(args.observations.resolve()),"sha256":input_hashes[args.observations]},"fitter":{"path":str(Path(__file__).resolve()),"sha256":sha256(__file__)},
            "primitive_helpers":{"path":str(Path(__file__).with_name('fit_stool.py').resolve()),"sha256":sha256(Path(__file__).with_name('fit_stool.py'))}},
            "read_scope":"Only specified scene/observations files and fitter/helper source imports. No fixture construction or evaluation inputs.",
            "training_frames":sorted(set(data["frame_index"][data["frame_index"]%2==0].tolist())),"validation_frames":sorted(set(data["frame_index"][data["frame_index"]%2==1].tolist())),
            "validation_exclusion_count":0,"quality_gate":{"passed":bool(passed),"independent_triangle_evaluation_required":True},
            "elapsed_seconds":time.time()-started}
    with (args.output/'fit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({"output":str(args.output.resolve()),"surface_model":args.surface_model,"quality_gate":result["quality_gate"],"objects":[{"uuid":o["uuid"],"validation":o["metrics"]["validation"]["point_count_weighted"],"parts":{p["name"]:p["training_closest_point_support"] for p in o["components"]}} for o in objects]},indent=2))
    return 0 if passed else 2


if __name__=="__main__":sys.exit(main())
