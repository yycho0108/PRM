#!/usr/bin/env python3

from typing import Tuple, List, Dict, Hashable
from enum import Enum
import numpy as np
from dataclasses import dataclass
from simple_parsing import Serializable

from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Qt5agg')
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from base import *
from prm import PRM
from util import *
from draw import *
from plan import LinearAngleLocalPlanner, LinearLocalPlanner
from maps import *
from sampler import UniformGridSampler
from time_limit import time_limit, TimeoutException


class RobotType(Enum):
    PLANAR_ARM = 'PLANAR_ARM'
    MOBILE_2D_BASE = 'MOBILE_2D_BASE'


@dataclass
class Settings(Serializable):
    bounds: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (-10.0, -10.0), (10.0, 10.0))  # (lo, hi)
    num_obstacles: int = 32
    obs_area_range: Tuple[float, float] = (0.01, 0.03)

    # PRM settings
    num_waypoints: int = 32
    num_nodes: int = 512
    num_neighbors: int = 8

    robot_type: RobotType = RobotType.PLANAR_ARM

    # N-Link-Arm Settings
    link_length_range: Tuple[float, float] = (0.05, 0.2)
    num_links: int = 4

    # Visualization/debugging
    plot_margin: float = 1.25
    seed: int = -1
    show: bool = True
    debug_cspace: bool = False


def main():
    opts = Settings()

    # seed = 9 # --> triggers nodenotfound
    # seed = 15  # semi-interesting path
    # seed = None
    # seed = 18
    seed = np.random.randint(low=0, high=65536)
    # seed = 24229
    # seed = 8019 # interesting result in both robot arm + holonomic settings
    # seed = 41429
    # seed = 41851
    # seed = 4279
    # seed = 23682
    # seed = 52885
    # seed = 28639
    # seed =
    print('seed', seed)
    np.random.seed(seed)

    # (Common) World settings
    bounds = np.asarray([[-10.0, -10.0], [10.0, 10.0]])  # (lo, hi)

    # Obstacles settings
    num_obstacles = 32
    min_obs_area = 0.01
    max_obs_area = 0.03  # expressed in ratio w.r.t total map area
    max_gen_plan_iter = 64

    # PRM Settings
    num_waypoints = 32
    num_vertices = 512
    num_neighbors = 8
    # robot_radius # << dilate obstacle by this much
    min_link_length = 0.20 * np.linalg.norm(bounds[1] - bounds[0], axis=-1)
    max_link_length = 0.05 * np.linalg.norm(bounds[1] - bounds[0], axis=-1)
    num_links = 4
    plot_margin: float = 1.25
    robot_type = RobotType.MOBILE_2D_BASE

    anim = False
    anim_vert = anim
    anim_edge = anim
    anim_exec = anim

    np.random.seed(seed)

    balls_map = BallsMap.generate(bounds, min_obs_area,
                                  max_obs_area, num_obstacles)

    #fig = plt.figure()
    #ax = fig.gca()
    #ax.set_xlim(plot_margin * bounds[0, 0], plot_margin * bounds[1, 0])
    #ax.set_ylim(plot_margin * bounds[0, 1], plot_margin * bounds[1, 1])
    #ax.set_axisbelow(True)
    #ax.grid(True)
    #ax.set_aspect('equal')
    # plotted_map = PlottedMap(balls_map, ax=ax)
    tracked_map = TrackedMap(balls_map)

    fig = plt.figure()
    ax = fig.gca()

    if robot_type == RobotType.PLANAR_ARM:
        link_length = np.random.uniform(
            min_link_length,
            max_link_length,
            size=num_links)
        query_map = PlanarArmsMap(tracked_map, link_length)
        sampler = UniformGridSampler(np.full(len(link_length), -np.pi),
                                     np.full(len(link_length), np.pi))
        local_planner = LinearAngleLocalPlanner(query_map, num_waypoints)
    else:
        query_map = tracked_map
        sampler = UniformGridSampler(bounds[0], bounds[1])
        local_planner = LinearLocalPlanner(query_map, num_waypoints)

    prm = PRM(sampler, query_map, local_planner, num_vertices, num_neighbors)
    aux_edges = {}
    try:
        with time_limit(10):
            # prm.construct()
            prm._construct_vertices()
            tracked_map.set_track(False)
            prm._construct_edges(aux_edges)
            prm._construct_graph()
    except TimeoutException as e:
        print('aborting due to time-out: {}'.format(e))
        return

    def _generate_feasible_config():
        q = None
        while True:
            q = sampler.sample(())
            if query_map.query(q):
                continue
            q_xy = query_map._map_to_xy(q)
            if not in_bound(q_xy, bounds[0], bounds[1]).all():
                continue
            break
        return q

    for _ in range(1):
        # Generate begin-end configurations.
        for _ in tqdm(range(max_gen_plan_iter)):
            q0 = _generate_feasible_config()
            q1 = _generate_feasible_config()
            path = prm.query(q0, q1)
            if path is not None:
                break
        if path is None:
            continue

        # NOTE(ycho): map visualization
        if opts.show:

            if anim_vert:
                fig = plt.figure()
                ax = fig.gca()
                ax.set_xlim(
                    plot_margin * bounds[0, 0],
                    plot_margin * bounds[1, 0])
                ax.set_ylim(
                    plot_margin * bounds[0, 1],
                    plot_margin * bounds[1, 1])
                ax.set_axisbelow(True)
                ax.grid(True)
                ax.set_aspect('equal')
                draw_ballsmap_2d(balls_map, ax)
                ax.set_title('PRM Vertex Construction')
                tracked_map.animate(fig)
                plt.close(fig)

            if anim_edge:
                fig = plt.figure()
                ax = fig.gca()
                ax.set_xlim(
                    plot_margin * bounds[0, 0],
                    plot_margin * bounds[1, 0])
                ax.set_ylim(
                    plot_margin * bounds[0, 1],
                    plot_margin * bounds[1, 1])
                ax.set_axisbelow(True)
                ax.grid(True)
                ax.set_aspect('equal')
                draw_ballsmap_2d(balls_map, ax)
                ax.set_title('PRM Edge Construction')
                animate_prm_edges(
                    aux_edges, lambda p: query_map._map_to_xy(p)
                    [..., -1, :],
                    fig)
                plt.close(fig)

            plt.clf()
            fig = plt.gcf()

            def _get_point(q):
                if robot_type == RobotType.PLANAR_ARM:
                    return query_map._map_to_xy(q)[..., -1, :]
                else:
                    return query_map._map_to_xy(q)

            if opts.debug_cspace:
                ax = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)
            else:
                ax = fig.add_subplot(1, 1, 1)

            # NOTE(ycho): prepare axis.
            ax.set_xlim(plot_margin * bounds[0, 0], plot_margin * bounds[1, 0])
            ax.set_ylim(plot_margin * bounds[0, 1], plot_margin * bounds[1, 1])
            ax.set_axisbelow(True)
            ax.grid(True)
            ax.set_aspect('equal')

            draw_ballsmap_2d(balls_map, ax)

            # V = prm.V
            V = _get_point(prm.V)

            # show prm
            ax.plot(V[..., 0], V[..., 1], '.', linewidth=4, markersize=8)

            # show PRM roadmap
            col = LineCollection(V[prm.E, :], alpha=0.25)
            ax.add_collection(col)

            # show init-goal pairs
            q0 = _get_point(q0)
            q1 = _get_point(q1)
            ax.plot(q0[0], q0[1], 'go', markersize=8, label='init')
            ax.plot(q1[0], q1[1], 'bo', markersize=8, label='goal')

            # show path (if available)
            if path is not None:
                if robot_type == RobotType.PLANAR_ARM:
                    path_xy = query_map._map_to_xy(path)
                    path_end = path_xy[:, -1]
                    ax.plot(path_end[:, 0], path_end[:, 1], 'rx')

                    path_up = interp_axis(path, axis=-2,
                                          residual_fn=lambda a, b: adiff(b, a)
                                          )
                    path_xy = query_map._map_to_xy(path_up)
                    path_end = path_xy[:, -1]
                    ax.plot(path_end[:, 0], path_end[:, 1], 'r--')
                else:
                    ax.plot(path[:, 0], path[:, 1], 'r--', label='path')

            if robot_type == RobotType.PLANAR_ARM:
                if path is not None:
                    path_up = interp_axis(path, axis=-2,
                                          residual_fn=lambda a, b: adiff(b, a)
                                          )
                    path_xy = query_map._map_to_xy(path_up)
                    path_xy = np.insert(path_xy, 0, 0, axis=1)
                    path_i = 0
                    h, = ax.plot(
                        path_xy[path_i, ..., 0],
                        path_xy[path_i, ..., 1],
                        'm.-')
                    ax.set_title('PRM Execution')

                    def _init():
                        return h,

                    def _update(i):
                        h.set_data(
                            path_xy[i, ..., 0],
                            path_xy[i, ..., 1])

                        if opts.debug_cspace:
                            ax2.cla()
                            draw_cspace_slice(
                                path_up[i],
                                sampler.min_bound, sampler.max_bound,
                                query_map, ax2)
                        return h,

                    ax.legend()
                    if anim_exec:
                        anim = FuncAnimation(
                            fig, _update, frames=np.arange(
                                len(path_xy)), init_func=_init, blit=True)
                        anim.save('/tmp/prm_exec.mp4')
                    plt.show()

            else:
                path_up = interp_axis(path, axis=-2)
                path_xy = query_map._map_to_xy(path_up)
                path_i = 0
                h, = ax.plot(
                    path_xy[path_i, ..., 0],
                    path_xy[path_i, ..., 1],
                    'mo', label='exec')
                ax.set_title('PRM Execution')

                def _init():
                    return h,

                def _update(i):
                    h.set_data(
                        path_xy[i, ..., 0],
                        path_xy[i, ..., 1])
                    return h,

                ax.legend()
                if anim_exec:
                    anim = FuncAnimation(
                        fig, _update, frames=np.arange(
                            len(path_xy)), init_func=_init, blit=True)
                    anim.save('/tmp/prm_exec.mp4')
                plt.show()

                #for _ in range(2 * path_xy.shape[0]):
                #    path_i = (path_i + 1) % path_xy.shape[0]
                #    h.set_data(
                #        path_xy[path_i, ..., 0],
                #        path_xy[path_i, ..., 1])
                #    plt.pause(0.001)


if __name__ == '__main__':
    main()
