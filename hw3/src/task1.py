import heapq
import json
import math
import os
import random
import sys
import time
from queue import PriorityQueue

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


def plot_locations(locations: dict, color: "str" = "black"):
    for key, value in locations.items():
        plt.plot(locations[key][0], locations[key][1], marker="o", markersize=10, markeredgecolor="red")
        plt.text(
            locations[key][0],
            locations[key][1] - 15,
            s=key,
            fontsize="x-large",
            fontweight="bold",
            c=color,
            ha="center",
        )
    return


def dilate(image):
    """get the dilated grid map to consider the configuration space"""
    height, width = image.shape
    dilated_image = image.copy()
    neighbors = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]

    for y in range(height):
        for x in range(width):
            if image[y, x] == 0:
                for dy, dx in neighbors:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        dilated_image[yy, xx] = 0

    return dilated_image


def plot_path(
    map_image, walkable_map, locations, start_key, end_key, path=None, visited=None, save=False, save_path=""
):
    """plot the derived path on the map image"""
    plt.imshow(map_image, cmap="gray")
    plot_locations(locations, color="black")

    if path:
        x_coords, y_coords = zip(*path)
        plt.plot(x_coords, y_coords, color="red", linewidth=2)

    if visited:
        x_v, y_v = zip(*visited)
        plt.scatter(x_v, y_v, s=1, alpha=0.5, color="lightgreen")

    plt.xlim(0, walkable_map.shape[1])
    plt.ylim(0, walkable_map.shape[0])
    plt.gca().invert_yaxis()
    plt.title(f"Path from {start_key} to {end_key}")
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"path_{start_key}_{end_key}.png"))
        plt.close()


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


class Planner:
    """Base class for path planning algorithms"""

    def __init__(self, grid_map: np.ndarray, map_resulution=0.2):
        self.grid_map = grid_map.copy()
        self.map_resulution = map_resulution

    def euclidean_distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def calcu_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += (
                self.euclidean_distance(Node(path[i][0], path[i][1]), Node(path[i + 1][0], path[i + 1][1]))
                * self.map_resulution
            )
        return length

    def get_path(self, goal_node):
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append((current_node.x, current_node.y))
            current_node = current_node.parent
        return path[::-1]

    def plan(self):
        raise NotImplementedError


class GraphSearchPlanner(Planner):
    def __init__(
        self,
        grid_map,
        plot_process=False,
        heuristic="analytic",
        heuristic_ratio=1,
        tie_breaking=False,
        search_strategy="astar",
        map_resulution=0.2,
    ):
        """

        Args:
            plot_process (bool, optional): whether to plot the planning process. Defaults to False.
            heuristic (str, optional): heuristic function to estimate the cost to the goal. Defaults to "analytic".
            heuristic_ratio (int, optional): weight of heuristic in the cost function. Defaults to 1.
            tie_breaking (bool, optional): whether to enable tie-breaking. Defaults to False.
            search_prefer (str, optional): search strategy. Defaults to "astar". Options: astar, dijkstra, greedy.
        """
        super().__init__(grid_map, map_resulution=map_resulution)
        self.plot_process = plot_process
        self.heuristic = heuristic
        self.heuristic_ratio = heuristic_ratio
        self.enable_tie_breaking = tie_breaking
        self.search_strategy = search_strategy  # astar, dijkstra, greedy

    def get_heuristic_value(self, current, target):
        heur_value = 0
        if self.heuristic == "analytic":  # analytic heuristic for 8-connected grid
            dy = abs(current[1] - target[1])
            dx = abs(current[0] - target[0])
            heur_value = 0.2 * (dy + dx + (math.sqrt(2) - 2) * min(dx, dy))
        elif self.heuristic == "euclidean":
            heur_value = math.sqrt((current[0] - target[0]) ** 2 + (current[1] - target[1]) ** 2)
        elif self.heuristic == "manhattan":
            heur_value = abs(current[0] - target[0]) + abs(current[1] - target[1])
        elif self.heuristic == "chebyshev":
            heur_value = max(abs(current[0] - target[0]), abs(current[1] - target[1]))
        else:
            raise ValueError("heuristic should be euclidean, manhattan or chebyshev")
        if self.enable_tie_breaking:
            # refer to Shen Lan Xue Yuan's lecture notes
            dx1 = abs(current[0] - target[0])
            dy1 = abs(current[1] - target[1])
            dx2 = abs(current[0] - target[0])
            dy2 = abs(current[1] - target[1])
            cross = abs(dx1 * dy2 - dx2 * dy1)
            heur_value += cross * 0.001
        return heur_value

    def plan(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        closed_list = set()  # store visited nodes
        open_list = PriorityQueue()  # store nodes to be visited
        open_list.put((0, start))

        node_cost_g = {start: 0}  # store cost from start to each node
        node_parent = {}  # store parent node of each node to trace back the path

        if self.plot_process:
            img_show = cv2.cvtColor((self.grid_map).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.circle(img_show, (start[1], start[0]), 10, (0, 255, 0), -1)
            cv2.circle(img_show, (goal[1], goal[0]), 10, (0, 0, 255), -1)

        while open_list:
            current_node = open_list.get()[1]
            closed_list.add(current_node)
            if self.plot_process:
                cv2.circle(img_show, (current_node[1], current_node[0]), 1, (0, 255, 0), -1)
                img_render = cv2.resize(img_show, (800, 800))
                cv2.imshow("A*", img_render)
                cv2.waitKey(1)
            current_cost = node_cost_g[current_node]

            if current_node == goal:  # reach the goal, reconstruct the path
                path = []
                while current_node in node_parent:
                    path.append(current_node)
                    current_node = node_parent[current_node]
                return path, current_cost, closed_list

            for move in moves:
                dx, dy = move
                neighbor = (current_node[0] + dx, current_node[1] + dy)
                if 0 <= neighbor[0] < len(self.grid_map) and 0 <= neighbor[1] < len(self.grid_map[0]):
                    if self.grid_map[neighbor[0]][neighbor[1]] == 0 or neighbor in closed_list:
                        continue
                    cost_g = (move[0] ** 2 + move[1] ** 2) ** 0.5 * self.map_resulution + current_cost
                    cost_h = self.get_heuristic_value(neighbor, goal)
                    if self.search_strategy == "greedy":
                        cost_f = cost_h
                    elif self.search_strategy == "astar":
                        cost_f = cost_g + cost_h * self.heuristic_ratio
                    elif self.search_strategy == "dijkstra":
                        cost_f = cost_g

                    if neighbor not in node_cost_g or cost_g < node_cost_g[neighbor]:  # update cost and parent
                        node_cost_g[neighbor] = cost_g
                        node_parent[neighbor] = current_node
                        open_list.put((cost_f, neighbor))

        return None, None, closed_list


class RRTStarPlanner(Planner):
    def __init__(
        self,
        grid_map,
        max_iter,
        step_size,
        search_radius,
        goal_toler_th,
        plot_process=False,
        map_resulution=0.2,
        enable_star=True,
    ):
        super().__init__(grid_map, map_resulution=map_resulution)
        self.max_iter = max_iter
        self.step_size = step_size  # max step distance when steering
        self.search_radius = search_radius  # search radius for finding near nodes
        self.goal_toler_th = goal_toler_th  # goal tolerance threshold
        self.plot_process = plot_process  # whether to plot the planning process
        self.enable_star = enable_star

    def find_nearest_node(self, nodes, rand_node):
        min_dist = float("inf")
        nearest_node = None
        for node in nodes:
            dist = self.euclidean_distance(node, rand_node)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def steer(self, from_node, to_node, max_extend_length=1.0):
        """extend from_node towards to_node with max_extend_length"""
        dist = self.euclidean_distance(from_node, to_node)
        if dist > max_extend_length:
            ratio = max_extend_length / dist
            x = int(from_node.x + ratio * (to_node.x - from_node.x))
            y = int(from_node.y + ratio * (to_node.y - from_node.y))
            new_node = Node(x, y)
            new_node.parent = from_node
            new_node.cost = from_node.cost + max_extend_length
        else:
            new_node = to_node
            new_node.parent = from_node
            new_node.cost = from_node.cost + dist
        return new_node

    def line_on_collision(self, node1, node2):
        """return True if there is collision between node1 and node2, False otherwise"""
        theta = math.atan2(node2.y - node1.y, node2.x - node1.x)
        dist = self.euclidean_distance(node1, node2)
        for i in range(int(dist)):
            x = int(node1.x + i * math.cos(theta))
            y = int(node1.y + i * math.sin(theta))
            if self.grid_map[x, y] == 0:
                return True
        return False

    def find_near_nodes(self, nodes, new_node, radius):
        """find nodes in the radius of new_node"""
        near_nodes = []
        for node in nodes:
            if self.euclidean_distance(node, new_node) <= radius:
                near_nodes.append(node)
        return near_nodes

    def plan(self, start, goal):
        start_node = Node(start[0], start[1])
        nodes = [start_node]
        goal_node = Node(goal[0], goal[1])
        if self.plot_process:
            img_show = cv2.cvtColor((self.grid_map).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.circle(img_show, (start_node.y, start_node.x), 10, (0, 255, 0), -1)
            cv2.circle(img_show, (goal_node.y, goal_node.x), 10, (0, 0, 255), -1)

        for i in range(self.max_iter):
            rand_node = Node(random.randint(0, len(self.grid_map[0]) - 1), random.randint(0, len(self.grid_map) - 1))
            nearest_node = self.find_nearest_node(nodes, rand_node)
            new_node = self.steer(nearest_node, rand_node, self.step_size)
            if self.grid_map[new_node.x, new_node.y] == 0:  # the node is on obstacle
                continue

            if not self.line_on_collision(
                nearest_node, new_node
            ):  # the edge connecting nearest_node and new_node is not on obstacle
                nodes.append(new_node)

                if self.euclidean_distance(new_node, goal_node) <= self.goal_toler_th:  # reach the goal
                    goal_node.parent = new_node
                    nodes.append(goal_node)
                    path = self.get_path(goal_node)
                    cost = self.calcu_path_length(path)
                    return path, cost, set([(node.x, node.y) for node in nodes])
                if self.enable_star:
                    near_nodes = self.find_near_nodes(nodes, new_node, self.search_radius)
                    for near_node in near_nodes:
                        new_cost = near_node.cost + self.euclidean_distance(near_node, new_node)
                        if new_cost < new_node.cost:  # find better parent
                            new_node.parent = near_node
                            new_node.cost = new_cost
                        else:  # rewire
                            new_cost = new_node.cost + self.euclidean_distance(near_node, new_node)
                            if new_cost < near_node.cost:
                                near_node.parent = new_node
                                near_node.cost = new_cost
                else:
                    new_node.parent = nearest_node
            if self.grid_map is not None and self.plot_process:
                cv2.circle(img_show, (rand_node.y, rand_node.x), 1, (0, 0, 255), -1)
                cv2.circle(img_show, (new_node.y, new_node.x), 1, (0, 255, 0), -1)
                cv2.line(img_show, (nearest_node.y, nearest_node.x), (new_node.y, new_node.x), (255, 0, 0), 2)
                img_render = cv2.resize(img_show, (800, 800))
                cv2.imshow("RRT*", img_render)
                cv2.waitKey(1)

        return None, -1, None


def get_all_paths(locations, planner: Planner):
    all_distances = {}
    all_visited_cells = {}
    all_paths = {}
    all_times = {}

    for start_key, start_pos in locations.items():
        all_distances[start_key] = {}
        all_visited_cells[start_key] = {}
        all_paths[start_key] = {}
        all_times[start_key] = {}

        for end_key, end_pos in locations.items():
            st = time.time()
            print(f"get path between {start_key} and {end_key}... ", end="")
            if start_key == end_key:
                cost = 0
                visited_cells = set()
                path = [start_pos]
            else:
                path, cost, visited_cells = planner.plan(start_pos, end_pos)
            dt = time.time() - st
            print(f"time {dt:.2f}s. distance: {cost:.2f}m.")

            all_distances[start_key][end_key] = cost
            all_visited_cells[start_key][end_key] = visited_cells
            all_paths[start_key][end_key] = path
            all_times[start_key][end_key] = dt

    return all_paths, all_distances, all_visited_cells, all_times


def write_log(
    log_path,
    raw_map,
    grid_map,
    all_paths,
    all_distances,
    all_visited_cells,
    all_times,
    locations,
    params=None,
):
    result_dict = {}
    for start_key in locations.keys():
        result_dict[start_key] = {}
        for end_key in locations.keys():
            result_dict[start_key][end_key] = {
                "distance": all_distances[start_key][end_key],
                "time": all_times[start_key][end_key],
            }
            visited_cells = all_visited_cells[start_key][end_key]
            path = all_paths[start_key][end_key]
            if path:
                with open(os.path.join(log_path, f"path_{start_key}_{end_key}.txt"), "w") as f:  # export path
                    for pos in path:
                        f.write(f"{pos[0]},{pos[1]}\n")
                plot_path(
                    raw_map,
                    grid_map,
                    locations,
                    start_key,
                    end_key,
                    path,
                    visited_cells,
                    save=True,
                    save_path=log_path,
                )
    with open(os.path.join(log_path, "0result.json"), "w") as f:
        json.dump(result_dict, f, indent=4)
    with open(os.path.join(log_path, "1params.json"), "w") as f:
        json.dump(params, f, indent=4)


def run(raw_map, grid_map, locations, MAP_RES, log_path, method, params):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if method == "rrt":
        planner = RRTStarPlanner(
            grid_map,
            map_resulution=MAP_RES,
            max_iter=params["max_iter"],
            step_size=params["step_size"],
            search_radius=params["search_radius"],
            goal_toler_th=params["goal_toler_th"],
            plot_process=False,
        )
    elif method == "graph_search":
        planner = GraphSearchPlanner(
            grid_map,
            map_resulution=MAP_RES,
            heuristic=params["heuristic"],
            heuristic_ratio=params["heuristic_ratio"],
            tie_breaking=params["tie_breaking"],
            search_strategy=params["search_strategy"],
            plot_process=False,
        )
    all_paths, all_distances, all_visited_cells, all_times = get_all_paths(locations, planner)
    print("All paths found. Saving results... ")
    write_log(
        log_path,
        raw_map,
        grid_map,
        all_paths,
        all_distances,
        all_visited_cells,
        all_times,
        locations,
        params=params,
    )
    print("Results saved.")


def main():

    raw_map = imageio.imread(sys.path[0] + "/../map/vivocity.png")
    grid_map_img = imageio.imread(sys.path[0] + "/../map/vivocity_dilate.png")[:, :, 0]
    grid_map = grid_map_img.transpose()

    MAP_RES = 0.2

    # occupied_cells = np.count_nonzero(grid_map == 0)
    # free_cells = np.count_nonzero(grid_map == 255)

    locations = {
        "start": [345, 95],
        "snacks": [470, 475],
        "store": [20, 705],
        "movie": [940, 545],
        "food": [535, 800],
    }

    if 0:  # XXX get the dilated grid map
        grid_map_walkable = dilate(grid_map)
        grid_map_walkable = np.rot90(grid_map_walkable, k=-1)
        grid_map_walkable = np.fliplr(grid_map_walkable)
        plt.imsave(sys.path[0] + "/map/vivocity_dilate.png", grid_map_walkable, cmap="gray")

    if 0:  # XXX single run
        # settings need to modify
        method = "rrt"  # "graph_search" or "rrt"
        log_prefix = "test"
        rrt_params = {
            "map_resulution": MAP_RES,
            "max_iter": 10000,
            "step_size": 20,
            "search_radius": 30,
            "goal_toler_th": 50,
        }
        graph_search_params = {
            "map_resulution": MAP_RES,
            "heuristic": "analytic",  # analytic, euclidean, manhattan, chebyshev
            "heuristic_ratio": 1,  # 0.25,0.5,0.75,1
            "tie_breaking": True,
            "search_strategy": "astar",  # astar, dijkstra, greedy
        }
        params = graph_search_params if method == "graph_search" else rrt_params
        log_path = os.path.join(sys.path[0], f"../results/{log_prefix}/{method}")
        run(raw_map, grid_map, locations, MAP_RES, log_path, method, params)

    if 0:  # XXX ablation experiments for graph search
        heuristic_ratio_set = [0.25, 0.5, 0.75, 1]
        heuristic_set = ["analytic", "euclidean", "manhattan", "chebyshev"]
        search_strategy_set = ["astar", "dijkstra", "greedy"]
        tie_breaking_set = [True, False]

        if 0:  # exp - heuristic
            params = {"heuristic_ratio": 1, "tie_breaking": False, "search_strategy": "astar"}
            for i, heuristic in enumerate(heuristic_set):
                log_path = os.path.join(sys.path[0], f"../results/graph_search/heuristic/{heuristic}")
                print(f"Experiment {i+1}: heuristic: {heuristic}")
                params["heuristic"] = heuristic
                run(raw_map, grid_map, locations, MAP_RES, log_path, "graph_search", params)
                print("-" * 50)
        if 0:  # exp - search strategy
            params = {"heuristic_ratio": 1, "tie_breaking": False, "heuristic": "analytic"}
            for i, search_strategy in enumerate(search_strategy_set):
                print(f"Experiment {i+1}: search strategy: {search_strategy}")
                log_path = os.path.join(sys.path[0], f"../results/graph_search/search_strategy/{search_strategy}")
                params["search_strategy"] = search_strategy
                run(raw_map, grid_map, locations, MAP_RES, log_path, "graph_search", params)
                print("-" * 50)
        if 0:  # exp - tie breaking
            params = {"heuristic_ratio": 1, "search_strategy": "astar", "heuristic": "analytic"}
            for i, tie_breaking in enumerate(tie_breaking_set):
                print(f"Experiment {i+1}: tie breaking: {tie_breaking}")
                log_path = os.path.join(sys.path[0], f"../results/graph_search/tie_breaking/{tie_breaking}")
                params["tie_breaking"] = tie_breaking
                run(raw_map, grid_map, locations, MAP_RES, log_path, "graph_search", params)
                print("-" * 50)
        if 0:  # exp - heuristic ratio
            params = {"heuristic": "analytic", "tie_breaking": False, "search_strategy": "astar"}
            for i, heuristic_ratio in enumerate(heuristic_ratio_set):
                print(f"Experiment {i+1}: heuristic ratio: {heuristic_ratio}")
                log_path = os.path.join(sys.path[0], f"../results/graph_search/heuristic_ratio/{heuristic_ratio}")
                params["heuristic_ratio"] = heuristic_ratio
                run(raw_map, grid_map, locations, MAP_RES, log_path, "graph_search", params)
                print("-" * 50)

    if 0:  # XXX ablation experiments for rrt*
        step_size_set = [5, 10, 20, 30]
        search_radius_set = [10, 20, 30, 40]
        rrt_params = {
            "map_resulution": MAP_RES,
            "max_iter": 20000,
            "step_size": 20,
            "search_radius": 30,
            "goal_toler_th": 50,
        }
        if 0:  # exp: step size
            for i, step_size in enumerate(step_size_set):
                print(f"Experiment {i+1}: step size: {step_size}")
                log_path = os.path.join(sys.path[0], f"../results/rrt_star/step_size/{step_size}")
                rrt_params["step_size"] = step_size
                run(raw_map, grid_map, locations, MAP_RES, log_path, "rrt", rrt_params)
                print("-" * 50)
        if 0:  # exp: search radius
            for i, search_radius in enumerate(search_radius_set):
                print(f"Experiment {i+1}: search radius: {search_radius}")
                log_path = os.path.join(sys.path[0], f"../results/rrt_star/search_radius/{search_radius}")
                rrt_params["search_radius"] = search_radius
                run(raw_map, grid_map, locations, MAP_RES, log_path, "rrt", rrt_params)
                print("-" * 50)
        if 1: # exp: rrt vs rrt*
            use_flag = [True, False]
            for flag in use_flag:
                print(f"Experiment: RRT* enabled: {flag}")
                log_path = os.path.join(sys.path[0], f"../results/rrt_star/enable_star/{flag}")
                rrt_params["enable_star"] = flag
                run(raw_map, grid_map, locations, MAP_RES, log_path, "rrt", rrt_params)
                print("-" * 50)

if __name__ == "__main__":
    main()
