import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.arcline_path_utils import discretize_lane
import numpy as np
from collections import deque
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
from pyquaternion import Quaternion
from scipy.interpolate import interp1d
from copy import deepcopy
from time import time
from itertools import product
import yaml
import pickle

from pdb import set_trace as st

def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def remove_duplicates(xy, eps):
    """Ensures that returned array always have at least 2
    rows.
    """
    N,_ = xy.shape
    assert(N >= 2), N

    diff = xy[1:] - xy[:-1]
    dist = np.linalg.norm(diff, axis=1)
    kept = np.ones(len(xy), dtype=bool)
    kept[:-1] = dist > eps
    assert(kept[0]), kept

    return xy[kept]


def check_duplicates(xy, eps):
    assert(xy.shape[0] >= 2), xy.shape
    diff = xy[1:] - xy[:-1]
    dist = np.linalg.norm(diff, axis=1)
    assert(np.all(dist > eps)), dist


def process_lanegraph(nmap, res_meters, eps):
    """lanes are
    {
        xys: n x 2 array (xy locations)
        in_edges: n x X list of lists
        out_edges: n x X list of lists
        edges: m x 5 (x,y,hcos,hsin,l)
        edgeixes: m x 2 (v0, v1)
        ee2ix: dict (v0, v1) -> ei
    }
    """
    lane_graph = {}

    # discretize the lanes (each lane has at least 2 points)
    for lane in nmap.lane + nmap.lane_connector:
        my_lane = nmap.arcline_path_3.get(lane['token'], [])
        discretized = np.array(discretize_lane(my_lane, res_meters))[:, :2]
        discretized = remove_duplicates(discretized, eps)
        check_duplicates(discretized, eps) # make sure each point is at least eps away from neighboring points
        lane_graph[lane['token']] = discretized

    # make sure connections don't have duplicates (now each lane has at least 1 point)
    for intok,conn in nmap.connectivity.items():
        for outtok in conn['outgoing']:
            if outtok in lane_graph:
                dist = np.linalg.norm(lane_graph[outtok][0] - lane_graph[intok][-1])
                if dist <= eps:
                    lane_graph[intok] = lane_graph[intok][:-1]
                    assert(lane_graph[intok].shape[0] >= 1), lane_graph[intok]

    xys = []
    laneid2start = {}
    for lid,lane in lane_graph.items():
        laneid2start[lid] = len(xys)
        xys.extend(lane.tolist())

    in_edges = [[] for _ in range(len(xys))]
    out_edges = [[] for _ in range(len(xys))]
    for lid,lane in lane_graph.items():
        for ix in range(len(lane)-1):
            out_edges[laneid2start[lid]+ix].append(laneid2start[lid]+ix+1)
        for ix in range(1, len(lane)):
            in_edges[laneid2start[lid]+ix].append(laneid2start[lid]+ix-1)
        for outtok in nmap.connectivity[lid]['outgoing']:
            if outtok in lane_graph:
                out_edges[laneid2start[lid]+len(lane)-1].append(laneid2start[outtok])
        for intok in nmap.connectivity[lid]['incoming']:
            if intok in lane_graph:
                in_edges[laneid2start[lid]].append(laneid2start[intok]+len(lane_graph[intok])-1)

    # includes a check that the edges are all more than length eps
    edges, edgeixes, ee2ix = process_edges(xys, out_edges, eps)

    # scclabels,sccs = find_sccs(out_edges)
    # scclen = np.array([len(sccs[i]) for i in scclabels])

    return {'xy': np.array(xys), 'in_edges': in_edges, 'out_edges': out_edges,
            'edges': edges, 'edgeixes': edgeixes, 'ee2ix': ee2ix}


def process_edges(xys, out_edges, eps):
    edges = []
    edgeixes = []
    ee2ix = {}
    for i in range(len(out_edges)):
        x0,y0 = xys[i]
        for e in out_edges[i]:
            x1,y1 = xys[e]
            diff = np.array([x1-x0, y1-y0])
            dist = np.linalg.norm(diff)
            assert(dist>eps), dist
            diff /= dist
            assert((i,e) not in ee2ix)
            ee2ix[i,e] = len(edges)
            edges.append([x0, y0, diff[0], diff[1], dist])
            edgeixes.append([i, e])
    return np.array(edges), np.array(edgeixes), ee2ix


def find_sccs(out_edges):
    n = len(out_edges)

    row = []
    col = []
    data = []
    for u in range(n):
        for v in out_edges[u]:
            row.append(u)
            col.append(v)
            data.append(1)
    mat = csr_matrix((data, (row, col)), shape=(n, n))
    ncc,labels = connected_components(mat, directed=True, connection='strong', return_labels=True)
    
    sccs = []
    vs = np.arange(n)
    for ci in range(ncc):
        sccs.append(vs[labels == ci])
    assert(sum((len(scc) for scc in sccs)) == n)
    sccs = sorted(sccs, key=lambda x: -len(x))
    
    return labels, sccs


def get_scenes(version, is_train):
    split = {
        'v1.0-trainval': {True: 'train', False: 'val'},
        'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
    }[version][is_train]
    scenes = create_splits_scenes()[split]
    return scenes


def get_sorted_samples(nusc, scenes):
    recs = [rec for rec in nusc.sample]
    for rec in recs:
        rec['scene_name'] = nusc.get('scene', rec['scene_token'])['name']
    recs = [rec for rec in recs if rec['scene_name'] in scenes]
    recs = sorted(recs, key=lambda x: (x['scene_name'], x['timestamp']))
    return recs


def interpret_instance(instance, t0):
    rot = Quaternion(instance['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    return {
        'x': instance['translation'][0],
        'y': instance['translation'][1],
        'hcos': np.cos(rot),
        'hsin': np.sin(rot),
        't': t0*1e-6,
    }


def get_scene2data(recs, nusc, egoonly=True):
    """All timestamps are converted to seconds (by multiplying by 1e-6)
    Every scene has an object with id "ego".
    Only objects that exist for >1 timestamps have "interp".
    {
        scene: {
            objs: {id: {traj:
                               [{x: y: hcos: hsin: t: }, ]
                        w: l: k: interp: },...}
            map_name:
            map_token:
        }
    }
    """
    scene2data = {}
    for rec in tqdm(recs):
        scene = rec['scene_name']
        log = nusc.get('log', nusc.get('scene', rec['scene_token'])['log_token'])
        if scene not in scene2data:
            scene2data[scene] = {
                'objs': {
                    'ego': {
                        'traj': [], 'w': 1.73, 'l': 4.084, 'k': 'vehicle.car',
                    },
                },
                'map_name': log['location'],
                'map_token': log['map_token'],
            }
        # add ego location
        egopose = nusc.get('ego_pose', nusc.get('sample_data',
                           rec['data']['LIDAR_TOP'])['ego_pose_token'])
        scene2data[scene]['objs']['ego']['traj'].append(interpret_instance(egopose, rec['timestamp']))
        # add objects
        if not egoonly:
            for ann in rec['anns']:
                instance = nusc.get('sample_annotation', ann)
                instance_name = instance['instance_token']
                assert(instance_name != 'ego'), instance_name
                if instance_name not in scene2data[scene]['objs']:
                    scene2data[scene]['objs'][instance_name] =\
                        {'traj': [], 'w': instance['size'][0],
                            'l': instance['size'][1], 'k': instance['category_name']}
                scene2data[scene]['objs'][instance_name]['traj'].append(interpret_instance(instance, rec['timestamp']))

    # make interpolators for objects around for more than 2 timestamps
    for _,scenedata in tqdm(scene2data.items()):
        for _,obj in scenedata['objs'].items():
            if len(obj['traj']) > 1:
                times = [row['t'] for row in obj['traj']]
                poses = [[row['x'], row['y'], row['hcos'], row['hsin']] for row in obj['traj']]
                obj['interp'] = interp1d(times, poses, kind='linear', axis=0, copy=False, bounds_error=True, assume_sorted=True)

    return scene2data


def trajectory_parse(nusc, is_train, egoonly=True):
    scenes = get_scenes(nusc.version, is_train)
    recs = get_sorted_samples(nusc, scenes)
    scene2data = get_scene2data(recs, nusc, egoonly=egoonly)

    return scene2data


def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def get_corners(box, lw):
    l, w = lw
    simple_box = np.array([
        [-l/2., -w/2.],
        [l/2., -w/2.],
        [l/2., w/2.],
        [-l/2., w/2.],
    ])
    h = np.arctan2(box[3], box[2])
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2]
    return simple_box


def plot_box(box, lw, color='g', alpha=0.7, no_heading=False):
    l, w = lw
    h = np.arctan2(box[3], box[2])
    simple_box = get_corners(box, lw)

    arrow = np.array([
        box[:2],
        box[:2] + l/2.*np.array([np.cos(h), np.sin(h)]),
    ])

    plt.fill(simple_box[:, 0], simple_box[:, 1], color=color, edgecolor='k',
             alpha=alpha, zorder=3, linewidth=1.0)
    if not no_heading:
        plt.plot(arrow[:, 0], arrow[:, 1], 'b', alpha=0.5)


def plot_car(x, y, h, l, w, color='b', alpha=0.5, no_heading=False):
    plot_box(np.array([x, y, np.cos(h), np.sin(h)]), [l, w],
             color=color, alpha=alpha, no_heading=no_heading)


def get_lane_matches(x, y, h, lane_graph, cdistmax, xydistmax):
    # check heading
    cdist = 1.0 - lane_graph['edges'][:,2]*np.cos(h) - lane_graph['edges'][:,3]*np.sin(h)
    kept = cdist < cdistmax

    if kept.sum() > 0:
        la_xy = lane_graph['edges'][kept,0:2]
        la_h = lane_graph['edges'][kept,2:4]
        la_l = lane_graph['edges'][kept,4]

        closest, dist = edge_closest_point(la_xy, la_h, la_l, np.array([x, y]))

        options = dist < xydistmax

        all_matches = {'closest': closest[options],
                       'ixes': lane_graph['edgeixes'][kept][options],
                      }
    else:
        all_matches = {
            'closest': np.empty((0, 2)),
            'ixes': np.empty((0, 2), dtype=np.int64),
        }

    return all_matches


def cluster_matches_combine(x, y, matches, lane_graph):
    """Clusters the matches and chooses the closest match for each cluster.
    """
    if len(matches['closest']) == 0:
        return matches

    ixes = []
    closest = []
    seen = {(v0,v1): False for v0,v1 in matches['ixes']}
    ordering = np.argsort(np.linalg.norm(np.array([[x, y]]) - matches['closest'], axis=1))
    for (v0,v1),close in zip(matches['ixes'][ordering], matches['closest'][ordering]):
        if seen[v0,v1]:
            continue
        ixes.append([v0,v1])
        closest.append(close)

        # make everything that is connected to this edge seen
        seen = cluster_bfs(v0, v1, seen, lane_graph, go_forward=True)
        seen = cluster_bfs(v0, v1, seen, lane_graph, go_forward=False)

    return {
        'ixes': np.array(ixes),
        'closest': np.array(closest),
    }


def edge_closest_point(la_xy, la_h, la_l, query):
    diff = query.reshape((1, 2)) - la_xy
    lmag = diff[:,0]*la_h[:,0] + diff[:,1]*la_h[:,1]
    lmag[lmag < 0] = 0.0
    lmagkept = lmag > la_l
    lmag[lmagkept] = la_l[lmagkept]
    closest = la_xy + lmag[:, np.newaxis] * la_h
    dist = query.reshape((1, 2)) - closest
    dist = np.linalg.norm(dist, axis=1)
    return closest, dist


def cluster_bfs(v0, v1, seen, lane_graph, go_forward):
    qu = deque()
    qu.append((v0,v1))
    while len(qu) > 0:
        c0,c1 = qu.popleft()
        seen[c0,c1] = True
        if go_forward:
            for newix in lane_graph['out_edges'][c1]:
                if (c1,newix) in seen and not seen[c1,newix]:
                    qu.append((c1,newix))
        else:
            for newix in lane_graph['in_edges'][c0]:
                if (newix,c0) in seen and not seen[newix,c0]:
                    qu.append((newix,c0))
    return seen


def expand_verts(v0, xys, conns, mindist):
    """Does BFS from e0.
    lanes are represented by {'v': [v0, v1,....],
                              'l': 0.0,
                              }

    Note that this function can return lanes of length less than mindist meters
    if a lane reaches a terminal node in the graph.
    """
    assert(mindist >= 0), f'only non-negative distances allowed {mindist}'

    qu = deque()
    qu.append({'v': [v0],
               'l': 0.0})
    all_lanes = []
    while len(qu) > 0:
        lane = qu.popleft()
        while lane['l'] <= mindist:
            v = lane['v'][-1]
            if len(conns[v]) == 0:
                break

            # put branches in the queue if there are any
            for outv in conns[v][1:]:
                newlane = deepcopy(lane)
                newlane['l'] += np.linalg.norm(xys[outv] - xys[v])
                newlane['v'].append(outv)
                qu.append(newlane)
            outv = conns[v][0]

            lane['l'] += np.linalg.norm(xys[outv] - xys[v])
            lane['v'].append(outv)

        all_lanes.append(lane)

    return all_lanes


def extend_forward(xys, le):
    diff = xys[-1] - xys[-2]
    diff /= np.linalg.norm(diff)
    newxy = xys[-1] + diff * le
    xys = np.concatenate((xys, newxy[np.newaxis]), axis=0)
    return xys


def extend_backward(xys, le):
    diff = xys[0] - xys[1]
    diff /= np.linalg.norm(diff)
    newxy = xys[0] + diff * le
    xys = np.concatenate((newxy[np.newaxis], xys), axis=0)
    return xys


def local_lane_closest(xys, ix0, egoxy):
    assert(xys.shape[0] >= 2), xys.shape

    diff = xys[1:] - xys[:-1]
    dist = np.linalg.norm(diff, axis=1)
    ec,ed = edge_closest_point(xys[:-1], diff / dist[:, np.newaxis], dist, egoxy)

    cix = ix0
    # find the closest point local to egoxy (this is a little complicated but justified)
    while 0<=cix-1 and ed[cix-1]<ed[cix]:
        cix -= 1
    while cix+1<len(ed) and ed[cix+1]<ed[cix]:
        cix += 1

    if cix+1<len(ed):
        assert(ed[cix+1]>=ed[cix]), f'{ed[cix]} {ed[cix+1]}'
    if 0<=cix-1:
        assert(ed[cix-1]>=ed[cix]), f'{ed[cix-1]} {ed[cix]}'

    return cix, ec[cix]


def xy2spline(xy, ix0, blim, flim, egoh):
    diff = xy[1:] - xy[:-1]
    dist = np.linalg.norm(diff, axis=1)
    head = diff / dist[:, np.newaxis]
    head = np.concatenate((
        head, head[[-1]],
    ), 0)
    xyhh = np.concatenate((xy, head), 1)

    # force spline to pass through current heading
    xyhh[ix0,2] = np.cos(egoh)
    xyhh[ix0,3] = np.sin(egoh)

    t = np.zeros(len(xy))
    t[1:] = np.cumsum(dist)
    t -= t[ix0]
    assert(t[0] < blim), f'{t[0]} {blim}'
    assert(t[-1] > flim), f'{t[-1]} {flim}'
    return interp1d(t, xyhh, kind='linear', axis=0, copy=False,
                    bounds_error=True, assume_sorted=True)

def xy2xyhs(xy, ix0, egoh):
    diff = xy[1:] - xy[:-1]
    dist = np.linalg.norm(diff, axis=1)
    head = diff / dist[:, np.newaxis]
    head = np.concatenate((
        head, head[[-1]],
    ), 0)
    xyhh = np.concatenate((xy, head), 1)

    # force spline to pass through current heading
    xyhh[ix0,2] = np.cos(egoh)
    xyhh[ix0,3] = np.sin(egoh)

    t = np.zeros(len(xy))
    t[1:] = np.cumsum(dist)
    t -= t[ix0]
    spline = interp1d(t, xyhh, kind='linear', axis=0, copy=False,
                    bounds_error=True, assume_sorted=True)
    return spline(np.linspace(t[0], t[-1],13))


def constant_heading_spline(egoxy, egoh, backdist, fordist):
    t = np.array([-backdist, fordist])
    x = np.array([
        [egoxy[0]-backdist*np.cos(egoh), egoxy[1]-backdist*np.sin(egoh), np.cos(egoh), np.sin(egoh)],
        [egoxy[0]+fordist*np.cos(egoh), egoxy[1]+fordist*np.sin(egoh), np.cos(egoh), np.sin(egoh)],
    ])
    return interp1d(t, x, kind='linear', axis=0, copy=False,
                    bounds_error=True, assume_sorted=True)


def get_prediction_splines(final_matches, lane_graph, backdist, fordist, xydistmax,
                           egoxy, lane_ds, lane_sig, sbuffer, egoh):
    """backdist: return splines that extend backwards at least this many meters
       fordist: return splines that extend forwards at least this many meters
       xydistmax: bound on how far the egoxy is from a lane
       egoxy: current (x,y) location of the car
       lane_ds: (meters) resolution when we warp the lane to pass through the ego
       lane_sig: (meters) how smoothly do splines return back to the lane graph
                 larger -> smoother
       sbuffer: (meters) needs to be large enough that when the splines are warped,
                 the spline is still "long enough".
       egoh: (radians) heading of the ego agent. Splines are guaranteed to pass exactly
               through the ego (x,y,h).

        Note that due to the buffers on the minimum distance for the splines, there is a 
        possibility that there are duplicates in the splines that are ultimately returned.

        Returns constant-heading spline if there are no lane matches.
    """

    if final_matches['ixes'].shape[0] == 0:
        return [constant_heading_spline(egoxy, egoh, backdist, fordist)]

    all_interps = []
    for (v0,v1),close in zip(final_matches['ixes'], final_matches['closest']):
        forward_lanes = expand_verts(v1, lane_graph['xy'], lane_graph['out_edges'],
                                     mindist=fordist+sbuffer+xydistmax)
        backward_lanes = expand_verts(v0, lane_graph['xy'], lane_graph['in_edges'],
                                      mindist=backdist+sbuffer+xydistmax)
        for flane in forward_lanes:
            for blane in backward_lanes:
                xys = np.concatenate((
                    lane_graph['xy'][blane['v'][::-1]],
                    lane_graph['xy'][flane['v']],
                ), axis=0)
                assert(xys.shape[0] >= 2), xys
                ix0 = len(blane['v']) - 1

                if flane['l'] <= fordist+sbuffer+xydistmax:
                    xys = extend_forward(xys, 1.0 + fordist + sbuffer + xydistmax - flane['l'])
                if blane['l'] <= backdist+sbuffer+xydistmax:
                    xys = extend_backward(xys, 1.0 + backdist + sbuffer + xydistmax - blane['l'])
                    ix0 += 1

                cix, cclose = local_lane_closest(xys, ix0, egoxy)

                tdist = np.zeros(len(xys))
                tdist[1:] = np.cumsum(np.linalg.norm(xys[1:] - xys[:-1], axis=1))
                tdist = tdist - tdist[cix] - np.linalg.norm(cclose - xys[cix])
                assert(tdist[0] < -backdist-sbuffer), f'{tdist[0]} {-backdist-sbuffer}'
                assert(tdist[-1] > fordist+sbuffer), f'{tdist[-1]} {fordist+sbuffer}'
                interp = interp1d(tdist, xys, kind='linear', axis=0, copy=False,
                                  bounds_error=True, assume_sorted=True)
                numback = int((backdist+sbuffer)/lane_ds)+1
                numfor = int((fordist+sbuffer)/lane_ds)+1
                teval = np.concatenate((
                    np.linspace(-backdist-sbuffer, 0.0, numback+1)[:-1],
                    np.linspace(0.0, fordist+sbuffer, numfor),
                ), 0)
                xys = interp(teval)
                xys = xys + (egoxy - cclose)[np.newaxis, :] * np.exp(-np.square(teval) / lane_sig**2)[:, np.newaxis]

                spline = xy2spline(xys, numback, blim=-backdist, flim=fordist, egoh=egoh)

                all_interps.append(spline)
    return all_interps


def xyh2speed(x0, y0, x1, y1, h1, dt):
    sabs = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) / dt
    ssign = 1 if (x1-x0)*np.cos(h1) + (y1-y0)*np.sin(h1) >= 0 else -1
    return ssign * sabs


def get_init_wstate(t0, t1, v, keptclasses):
    wstate = {'t': t1, 'objs': {}}
    for objid,obj in v['objs'].items():
        if 'interp' in obj and obj['k'] in keptclasses and obj['traj'][0]['t'] <= t0 and t1 <= obj['traj'][-1]['t']:
            x0,y0,hcos0,hsin0 = obj['interp'](t0)
            x1,y1,hcos1,hsin1 = obj['interp'](t1)
            h1 = np.arctan2(hsin1, hcos1)
            speed = xyh2speed(x0, y0, x1, y1, h1, t1 - t0)
            wstate['objs'][objid] = {
                'x': x1,
                'y': y1,
                'h': h1,
                's': speed,
                'l': obj['l'],
                'w': obj['w'],
            }
    return wstate


def viz_wstate(wstate, lane_graph, window, imname, scene_name):
    fig = plt.figure(figsize=(12, 12))
    gs = mpl.gridspec.GridSpec(1, 1, left=0.04, right=0.96, top=0.96, bottom=0.04)
    ax = plt.subplot(gs[0, 0])

    # plot lane graph
    plt.plot(lane_graph['edges'][:,0],lane_graph['edges'][:,1], '.', markersize=2)
    mag = 0.3
    plt.plot(lane_graph['edges'][:,0]+mag*lane_graph['edges'][:,2],
            lane_graph['edges'][:,1]+mag*lane_graph['edges'][:,3], '.', markersize=1)

    # plot objects
    for objid,obj in wstate['objs'].items():
        carcolor = 'g' if 'control' in obj else 'b'
        # if 'control' in obj:
        #     carcolor = 'b'
        #     if objid == 'ego' and debug_viz:
        #         vizdict = {'acc': obj['control']['acc'], 'stgt': obj['control']['stgt']}
        #         if 'warning' in obj['control']:
        #             vizdict['warning'] = obj['control']['warning']
        #         plt.title(f"s0: {obj['s']:.2f} Control: {vizdict}")
        plot_car(obj['x'], obj['y'], obj['h'], obj['l'], obj['w'], color=carcolor)

    # # plot lane assignments
    for objid,obj in wstate['objs'].items():
        plt.plot(obj['final_matches']['closest'][:,0], obj['final_matches']['closest'][:,1],
                'b.', markersize=8, alpha=0.3)

    # # plot lane predictions
    # for objid,obj in wstate['objs'].items():
    #     for interp in obj['splines']:
    #         lane = interp(np.arange(-1, 4))
    #         for lx,ly,lhcos,lhsin in lane:
    #             plot_car(lx, ly, np.arctan2(lhsin, lhcos), obj['l']*0.3, obj['w']*0.3,
    #                     color='yellow', alpha=0.5)

    if 'ego' in wstate['objs']:
        centerx,centery = wstate['objs']['ego']['x'], wstate['objs']['ego']['y']
    else:
        centerx,centery = 200.0,200.0
    plt.xlim((centerx - window, centerx + window))
    plt.ylim((centery - window, centery + window))
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title(scene_name)
    print('saving', imname)
    plt.savefig(imname)
    plt.close(fig)


def update_wstate(wstate, v, dt):
    t0 = wstate['t']
    t1 = t0 + dt
    newwstate = {'t': t1, 'objs': {}}
    for objid,obj in wstate['objs'].items():
        if 'control' in obj:
            speed = xyh2speed(obj['x'], obj['y'], obj['control']['x'],
                              obj['control']['y'], obj['control']['h'], dt)
            newwstate['objs'][objid] = {
                'x': obj['control']['x'],
                'y': obj['control']['y'],
                'h': obj['control']['h'],
                's': speed, 'l': obj['l'], 'w': obj['w'],
            }
        elif v['objs'][objid]['traj'][0]['t'] <= t1 <= v['objs'][objid]['traj'][-1]['t']:
            x,y,hcos,hsin = v['objs'][objid]['interp'](t1)
            h = np.arctan2(hsin, hcos)
            speed = xyh2speed(obj['x'], obj['y'], x, y, h, dt)
            newwstate['objs'][objid] = {'x': x, 'y': y, 'h': h,
                                        's': speed, 'l': obj['l'], 'w': obj['w']}
    return newwstate


def update_wstate_pred(wstate, v, dt, keptclasses):
    t0 = wstate['t']
    t1 = t0 + dt
    newwstate = {'t': t1, 'objs': {}}

    # update control
    obj = wstate['objs']['ego']
    speed = xyh2speed(obj['x'], obj['y'], obj['control']['x'],
                              obj['control']['y'], obj['control']['h'], dt)
    newwstate['objs']['ego'] = {
        'x': obj['control']['x'],
        'y': obj['control']['y'],
        'h': obj['control']['h'],
        's': speed, 'l': obj['l'], 'w': obj['w'],
    }

    for objid,obj in v['objs'].items():
        if objid != 'ego' and 'interp' in obj and obj['k'] in keptclasses and obj['traj'][0]['t'] <= t0 and t1 <= obj['traj'][-1]['t']:
            x0,y0,hcos0,hsin0 = obj['interp'](t0)
            x1,y1,hcos1,hsin1 = obj['interp'](t1)
            h1 = np.arctan2(hsin1, hcos1)
            speed = xyh2speed(x0, y0, x1, y1, h1, t1 - t0)
            newwstate['objs'][objid] = {
                'x': x1,
                'y': y1,
                'h': h1,
                's': speed,
                'l': obj['l'],
                'w': obj['w'],
            }
    return newwstate

def dump_wstate(wstate, egolocs):
    dump_wstate = {'t': wstate['t'], 'objs': {}}

    for objid,obj in wstate['objs'].items():
            dump_wstate['objs'][objid] = {'x': obj['x'], 'y': obj['y'], 'h': obj['h'],
                                        's': obj['s'], 'l': obj['l'], 'w': obj['w']}
    
    # collect ego plan
    dump_wstate['objs']['ego']['plan'] = egolocs

    return dump_wstate

def compute_splines(wstate, lane_graph, cdisttheta, xydistmax, lane_ds, lane_sig, sbuffer,
                    smax, tmax):
    assert(smax > 0), f'{smax}'
    for objid,obj in wstate['objs'].items():
        matches = get_lane_matches(obj['x'], obj['y'], obj['h'], lane_graph,
                                   cdistmax=1.0 - np.cos(np.radians(cdisttheta)),
                                   xydistmax=xydistmax)
        # update wstate in place
        obj['final_matches'] = cluster_matches_combine(obj['x'], obj['y'], matches, lane_graph)
        backdist = 1.0 if obj['s'] > 0 else 1.0+abs(obj['s'])*tmax
        fordist = 1.0+smax*tmax if obj['s'] < 0 else max(1.0+smax*tmax, 1.0+obj['s']*tmax)
        obj['splines'] = get_prediction_splines(obj['final_matches'], lane_graph, backdist=backdist, fordist=fordist,
                                                xydistmax=xydistmax, egoxy=np.array([obj['x'],obj['y']]),
                                                lane_ds=lane_ds, lane_sig=lane_sig, sbuffer=sbuffer,
                                                egoh=obj['h'])
    return wstate


def postprocess_act_for_speed(x0, y0, x1, y1, h1, s1, dt):
    news = xyh2speed(x0, y0, x1, y1, h1, dt)
    if np.abs(news) < 1e-6:
        assert(np.abs(s1) < 1e-6), f'{news} {s1}'
        return x1, y1, h1
    assert(np.sign(news) == np.sign(s1)), f'{news} {s1}'
    diff = np.array([x1 - x0, y1 - y0])
    diff /= np.linalg.norm(diff)
    return x0 + diff[0] * dt * np.abs(s1), y0 + diff[1] * dt * np.abs(s1), h1


def compute_speed_profile(s, stgt, acc, nsteps, preddt):
    """accelerate with magnitude no greater than acc from speed
    s to speed stgt. Returns profile of length nsteps+1 (eg the
    first index always has speed s).
    """
    if stgt > s:
        sprof = s + np.arange(nsteps+1) * acc * preddt
        sprof[sprof > stgt] = stgt
    elif stgt < s:
        sprof = s - np.arange(nsteps+1) * acc * preddt
        sprof[sprof < stgt] = stgt
    else:
        sprof = s + np.zeros(nsteps+1)
    return sprof


def sprof2dists(sprof, preddt):
    """Converts n+1 length sprof to distances
    """
    teval = np.zeros(len(sprof))
    teval[1:] = np.cumsum(sprof[1:]*preddt)
    return teval


def collect_other_trajs(wstate, egoid, nsteps, preddt, predsfacs,
                        predafacs, interacdist, maxacc):
    egoobj = wstate['objs'][egoid]
    egox,egoy,egoh = egoobj['x'],egoobj['y'],egoobj['h']
    other_trajs = []
    for otherid,other in wstate['objs'].items():
        if otherid == egoid or np.sqrt((egox-other['x'])**2+(egoy-other['y'])**2) > interacdist:
            continue
        sprofs = [compute_speed_profile(s=other['s'], stgt=other['s']*sfac,
                                        acc=maxacc*afac, nsteps=nsteps, preddt=preddt)
                                        for sfac in predsfacs for afac in predafacs]
        tevals = [sprof2dists(sprof, preddt) for sprof in sprofs]
        for interp in other['splines']:
            for teval in tevals:
                xyhh = interp(teval)
                traj = np.empty((nsteps+1, 5))
                traj[:, :2] = xyhh[:, :2]
                traj[:, 2] = np.arctan2(xyhh[:, 3], xyhh[:, 2])
                traj[:, 3] = other['l']
                traj[:, 4] = other['w']
                other_trajs.append(traj)

    if len(other_trajs) > 0:
        other_trajs = np.transpose(np.array(other_trajs), (1, 0, 2))
    else:
        other_trajs = np.empty((nsteps+1, 0, 5))

    return other_trajs


def score_dists(dists, score_wmin, score_wfac):
    assert(dists.ndim == 1), dists.shape
    w = score_wmin + np.arange(len(dists)) * score_wfac
    probs = 1.0 + np.tanh(-dists * w)
    probs[dists < 0] = 1.0
    return probs


def debug_planner(egotraj, otherobjs, lane_graph, debugname, egoobj, sprofi, x0, x1, dist, probs):
    egocircles = boxes2circles(egotraj)
    othercircles = boxes2circles(otherobjs)

    for ti in range(otherobjs.shape[0]):
        fig = plt.figure(figsize=(10, 10))
        gs = mpl.gridspec.GridSpec(1, 1, left=0.02, right=0.98, bottom=0.02, top=0.98)
        ax = plt.subplot(gs[0, 0])

        # plot lane graph
        plt.plot(lane_graph['edges'][:,0],lane_graph['edges'][:,1], '.', markersize=2)
        mag = 0.3
        plt.plot(lane_graph['edges'][:,0]+mag*lane_graph['edges'][:,2],
                 lane_graph['edges'][:,1]+mag*lane_graph['edges'][:,3], '.', markersize=1)

        for x,y,h,l,w in otherobjs[ti]:
            plot_car(x, y, h, l, w, color='b', alpha=0.2)
        plot_car(egotraj[ti,0,0],egotraj[ti,0,1], egotraj[ti,0,2],
                 egotraj[ti,0,3], egotraj[ti,0,4], color='g', alpha=0.2)

        # plot circles
        for circs in othercircles[ti]:
            for x,y,r in circs:
                ax.add_patch(plt.Circle((x, y), r, color='k'))
        for circs in egocircles[ti]:
            for x,y,r in circs:
                ax.add_patch(plt.Circle((x,y), r, color='k'))

        # plot closest distance
        plt.plot([x0[ti,0,0], x1[ti,0,0]],
                 [x0[ti,0,1], x1[ti,0,1]])
        plt.title(f'{dist[ti]:.2f} {probs[ti]:.2f}')

        ax.set_aspect('equal')
        plt.xlim((egoobj['x']-60.0, egoobj['x']+60.0))
        plt.ylim((egoobj['y']-60.0, egoobj['y']+60.0))
        imname = f'{debugname}_{sprofi:04}_{ti:03}.jpg'
        print('saving', imname)
        plt.savefig(imname)
        plt.close(fig)


def choose_sprof(otherobjs, sprofs, egoobj, egospline, lane_graph, nsteps, debugname,
                 score_wmin, score_wfac, debug):
    # if no other objects are around, just brake
    if otherobjs.shape[1] == 0:
        return sprofs[np.argmin([sprof['teval'][-1] for sprof in sprofs])]

    egotraj = np.empty((nsteps+1, 1, 5))
    egotraj[:, :, 3] = egoobj['l']
    egotraj[:, :, 4] = egoobj['w']

    all_probs = []
    for sprofi,sprof in enumerate(sprofs):
        egolocs = egospline(sprof['teval'])
        egotraj[:, 0, :2] = egolocs[:, :2]
        egotraj[:, 0, 2] = np.arctan2(egolocs[:, 3], egolocs[:, 2])

        # approximate distances
        x0, x1, dist = approx_bbox_distance(egotraj, otherobjs)
        dist = dist[:, 0]
        probs = score_dists(dist, score_wmin, score_wfac)
        prob = 1.0 - np.product(1.0 - probs)
        all_probs.append(prob)
        if debug:
            debug_planner(egotraj, otherobjs, lane_graph, debugname, egoobj, sprofi,
                        x0, x1, dist, probs)
    # always choose the route least likely to result in a collision
    # choose the high speed one
    all_probs.reverse()
    
    chosen_ix = np.argmin(all_probs)
    
    return sprofs[-chosen_ix-1]
    # return sprofs[chosen_ix]


def gen_sprofiles(s0, preddt, nsteps, planaccfacs, maxacc, smax, NS, debugname):
    n1 = nsteps // 2
    n2 = nsteps - n1
    sprofs = []
    for fac in planaccfacs:
        acc = fac*maxacc
        stop = min(smax, s0 + n1*preddt*acc)
        sbot = max(0.0, s0 - n1*preddt*acc)
        for s1 in np.linspace(sbot, stop, NS):
            sprof1 = compute_speed_profile(s0, s1, acc, n1, preddt)
            stop = min(smax, sprof1[-1] + n2*preddt*acc)
            sbot = max(0.0, sprof1[-1] - n2*preddt*acc)
            for s2 in np.linspace(sbot, stop, NS):
                sprof2 = compute_speed_profile(sprof1[-1], s2, acc, n2, preddt)
                sprof = np.concatenate((sprof1, sprof2[1:]))
                teval = sprof2dists(sprof, preddt)
                sprofs.append({'sprof': sprof,
                               'teval': teval,
                               'acc': acc,
                               's1': s1,
                               's2': s2})

    return sprofs


def compute_action(wstate, objid, dt, nsteps, preddt, predsfacs, predafacs,
                   interacdist, maxacc, debugname, lane_graph, planaccfacs, smax,
                   plannspeeds, debug, score_wmin, score_wfac):
    obj = wstate['objs'][objid]

    spline = obj['splines'][0]

    sprofs = gen_sprofiles(obj['s'], preddt, nsteps, planaccfacs, maxacc, smax,
                           plannspeeds, debugname)
    otherobjs = collect_other_trajs(wstate, objid, nsteps, preddt, predsfacs, predafacs,
                                    interacdist, maxacc)
    sprof = choose_sprof(otherobjs, sprofs, obj, spline, lane_graph, nsteps, debugname,
                         score_wmin, score_wfac, debug=debug)

    stgt = compute_speed_profile(obj['s'], sprof['s1'], sprof['acc'], 1, dt)[1]

    # stgt = min(smax, obj['s'] + maxacc*preddt)
    newx,newy,newhcos,newhsin = spline(dt*stgt)
    newh = np.arctan2(newhsin, newhcos)
    newx,newy,newh = postprocess_act_for_speed(obj['x'], obj['y'],
                                               newx, newy, newh, stgt, dt)

    obj['control'] = {
        'x': newx,
        'y': newy,
        'h': newh,
    }

def compute_action_with_prediction(wstate, objid, dt, nsteps, preddt, predsfacs, predafacs,
                   interacdist, maxacc, debugname, lane_graph, planaccfacs, smax,
                   plannspeeds, debug, score_wmin, score_wfac, other_objs=[]):
    obj = wstate['objs'][objid]

    spline = obj['splines'][0]

    sprofs = gen_sprofiles(obj['s'], preddt, nsteps, planaccfacs, maxacc, smax,
                           plannspeeds, debugname)

    if len(other_objs) > 0:
        other_objs = np.transpose(np.array(other_objs), (1, 0, 2))
    else:
        other_objs = np.empty((nsteps+1, 0, 5))


    sprof = choose_sprof(other_objs, sprofs, obj, spline, lane_graph, nsteps, debugname,
                         score_wmin, score_wfac, debug=debug)

    stgt = compute_speed_profile(obj['s'], sprof['s1'], sprof['acc'], 1, dt)[1]

    # stgt = min(smax, obj['s'] + maxacc*preddt)
    newx,newy,newhcos,newhsin = spline(dt*stgt)
    newh = np.arctan2(newhsin, newhcos)
    newx,newy,newh = postprocess_act_for_speed(obj['x'], obj['y'],
                                               newx, newy, newh, stgt, dt)

    obj['control'] = {
        'x': newx,
        'y': newy,
        'h': newh,
        'sprof': sprof,
    }


def boxes2circles(b):
    B,NA,_ = b.shape

    XY,Hi,Li,Wi = b[:,:,[0,1]],b[:,:,2],b[:,:,3],b[:,:,4]
    L = np.maximum(Li, Wi)
    W = np.minimum(Li, Wi)
    kept = Li < Wi
    H = np.copy(Hi)
    H[kept] = H[kept] + np.pi/2.0

    v0 = ((L-W)/2 + W/4)[:,:,np.newaxis] * np.stack((np.cos(H), np.sin(H)), 2)
    v1 = (W/4)[:,:,np.newaxis] * np.stack((-np.sin(H), np.cos(H)), 2)

    circles = np.empty((B, NA, 5, 3))
    circles[:, :, 0, [0,1]] = XY + v0 + v1
    circles[:, :, 1, [0,1]] = XY - v0 + v1
    circles[:, :, 2, [0,1]] = XY - v0 - v1
    circles[:, :, 3, [0,1]] = XY + v0 - v1
    circles[:, :, 4, [0,1]] = XY
    circles[:, :, 4, 2] = W/2
    circles[:, :, :4, 2] = W[:,:,np.newaxis] / 4

    return circles


def approx_bbox_distance(b0, b1):
    B,NA0,_ = b0.shape
    _,NA1,_ = b1.shape

    bc0 = boxes2circles(b0)
    bc1 = boxes2circles(b1)

    diff = bc1[:, :, :, [0,1]].reshape((B, 1, 1, NA1, 5, 2))\
           - bc0[:, :, :, [0,1]].reshape((B, NA0, 5, 1, 1, 2))
    dc = np.linalg.norm(diff, axis=5)
    direc = diff / dc.reshape((B, NA0, 5, NA1, 5, 1))
    dist = np.maximum(0.0, dc - bc0[:,:,:,2].reshape((B, NA0, 5, 1, 1)) - bc1[:,:,:,2].reshape((B, 1, 1, NA1, 5)))

    direc = direc.reshape((B, NA0, 5*NA1*5, 2))
    dist = dist.reshape((B, NA0, 5*NA1*5))

    cix = dist.argmin(axis=2)
    direc = np.take_along_axis(direc, cix[:,:,np.newaxis,np.newaxis], axis=2)[:,:,0,:]
    dist = np.take_along_axis(dist, cix[:,:,np.newaxis], axis=2)
    c0 = np.take_along_axis(bc0, cix[:,:,np.newaxis, np.newaxis]//(NA1*5), axis=2)[:,:,0,:]

    x0 = c0[:,:,[0,1]] + c0[:,:,[2]] * direc
    x1 = x0 + dist * direc

    return x0,x1,dist[:,:,0]


def read_conf(f):
    print('reading', f)
    with open(f, 'r') as reader:
        conf = yaml.safe_load(reader)
    return conf


def state_to_corners(state, xix, yix, hix, lix, wix):
    """returns B x NA x 4 x 2 tensor of corner coordinates
    """
    xs = np.stack((state[:, :, lix],
                 state[:, :, lix],
                 -state[:, :, lix],
                 -state[:, :, lix],), 2)/2.0
    ys = np.stack((state[:, :, wix],
                  -state[:, :, wix],
                 -state[:, :, wix],
                 state[:, :, wix],), 2)/2.0

    xrot = xs * np.cos(state[:, :, [hix]]) - ys * np.sin(state[:, :, [hix]]) + state[:, :, [xix]]
    yrot = xs * np.sin(state[:, :, [hix]]) + ys * np.cos(state[:, :, [hix]]) + state[:, :, [yix]]

    return np.stack((xrot, yrot), 3)


def state_corner_collisions(state, corners, xix, yix, hix, lix, wix):
    """Runs collision-checking between state and corners
    """
    B, NA, _ = state.shape
    BC, NC, _, _ = corners.shape

    xt = corners[:, :, :, 0].reshape((BC, NC, 1, 4)) - state[:, :, xix].reshape((B, 1, NA, 1))
    yt = corners[:, :, :, 1].reshape((BC, NC, 1, 4)) - state[:, :, yix].reshape((B, 1, NA, 1))
    xrot = xt * np.cos(state[:, :, hix].reshape((B, 1, NA, 1))) + yt * np.sin(state[:, :, hix].reshape((B, 1, NA, 1)))
    yrot = -xt * np.sin(state[:, :, hix].reshape((B, 1, NA, 1))) + yt * np.cos(state[:, :, hix].reshape((B, 1, NA, 1)))

    collidex = np.logical_and(xrot.max(3) > -state[:, :, lix].reshape((B, 1, NA))/2.0,
                              xrot.min(3) < state[:, :, lix].reshape((B, 1, NA))/2.0)
    collidey = np.logical_and(yrot.max(3) > -state[:, :, wix].reshape((B, 1, NA))/2.0,
                              yrot.min(3) < state[:, :, wix].reshape((B, 1, NA))/2.0)

    return collidex, collidey


def query_to_collisions(query, obstacles):
    """query is B x N x _
       obstacles is B x M x _

       returns B x N indicating which of the queries
       collided with the M obstacles, vectorized over B
    """
    obstacle_corners = state_to_corners(obstacles, 0, 1, 2, 3, 4)
    query_corners = state_to_corners(query, 0, 1, 2, 3, 4)
    collidex0, collidey0 = state_corner_collisions(query, obstacle_corners, 0, 1, 2, 3, 4)
    collidex1, collidey1 = state_corner_collisions(obstacles, query_corners, 0, 1, 2, 3, 4)
    collide0 = np.logical_and(collidex0, collidey0)
    collide1 = np.transpose(np.logical_and(collidex1, collidey1), (0, 2, 1))
    collide = np.logical_and(collide0, collide1)

    # note we could easily return number of collisions
    colcounts = collide.sum(axis=1)
    return colcounts > 0


def check_collisions(wstate, egoid):
    query = np.array([[[wstate['objs'][egoid]['x'], wstate['objs'][egoid]['y'], wstate['objs'][egoid]['h'],
                        wstate['objs'][egoid]['l'], wstate['objs'][egoid]['w']]]])
    obstacles = []
    for objid,obj in wstate['objs'].items():
        if objid == egoid:
            continue
        obstacles.append([obj['x'], obj['y'], obj['h'], obj['l'], obj['w']])
    obstacles = np.array(obstacles).reshape((1, len(obstacles), 5))
    didcollide = query_to_collisions(query, obstacles)
    return didcollide.sum()


def read_traj_data(version):
    fname = f'./storage/traj_{version}.pkl'
    print('reading cached trajectory splines', fname)
    with open(fname, 'rb') as reader:
        scene2data = pickle.load(reader)
    return scene2data


def rollout_scene_collision(k, v, lane_graphs, conf, Tsteps):
    t0 = v['objs']['ego']['traj'][0]['t']
    t1 = t0 + conf['dt']
    lane_graph = lane_graphs[v['map_name']]
    wstate = get_init_wstate(t0, t1, v, conf['keptclasses'])
    compute_splines(wstate, lane_graph, conf['cdisttheta'], conf['xydistmax'],
                    conf['lane_ds'], conf['lane_sig'], conf['sbuffer'], conf['smax'], conf['nsteps']*conf['preddt'])
    compute_action(wstate, 'ego', conf['dt'], conf['nsteps'], conf['preddt'], conf['predsfacs'], conf['predafacs'],
                    conf['interacdist'], conf['maxacc'], f'plan{k}_{0:04}', lane_graph, conf['planaccfacs'], conf['smax'],
                    conf['plannspeeds'], debug=False, score_wmin=conf['score_wmin'], score_wfac=conf['score_wfac'])
    for istep in range(Tsteps):
        wstate = update_wstate(wstate, v, conf['dt'])
        compute_splines(wstate, lane_graph, conf['cdisttheta'], conf['xydistmax'],
                        conf['lane_ds'], conf['lane_sig'], conf['sbuffer'], conf['smax'], conf['nsteps']*conf['preddt'])
        compute_action(wstate, 'ego', conf['dt'], conf['nsteps'], conf['preddt'], conf['predsfacs'], conf['predafacs'],
                    conf['interacdist'], conf['maxacc'], f'plan{k}_{0:04}', lane_graph, conf['planaccfacs'], conf['smax'],
                    conf['plannspeeds'], debug=False, score_wmin=conf['score_wmin'], score_wfac=conf['score_wfac'])
        didcollide = check_collisions(wstate, 'ego')
        if didcollide > 0:
            return True
    return False

# pprint dictionary
import json
def pprint(s_dict, indent = 4):
    p_dict = json.dumps(s_dict, indent=indent)
    print(p_dict)