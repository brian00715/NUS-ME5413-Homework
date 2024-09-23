# %matplotlib inline
import itertools
import json
import os
import sys
import time
from queue import PriorityQueue

import imageio
import matplotlib.pyplot as plt


# Util functions -----------------------------------
def transform_distance_dict(all_distances):
    keys = list(all_distances.keys())
    size = len(keys)
    distances_matrix = [[0] * size for _ in range(size)]
    key_index = {key: i for i, key in enumerate(keys)}

    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            distances_matrix[i][j] = all_distances[key1][key2]["distance"]

    return distances_matrix, key_index, keys


def plot_tour_on_map(raw_map, grid_map, locations, tour, all_paths, save=False, show=True, file_name=""):
    plt.imshow(raw_map, cmap="gray")

    # Draw the paths
    for i in range(len(tour) - 1):
        location = tour[i]
        path = all_paths[location][tour[i + 1]]
        path_on_floor_plan = transform_path_to_floor_plan(path, grid_map.shape, raw_map.shape)
        plt.plot([p[0] for p in path_on_floor_plan], [p[1] for p in path_on_floor_plan], "r-")
        store_pos = locations[location]
        plt.text(store_pos[0] - 30, store_pos[1] - 20, str(i + 1), color="black", fontsize=14)

    # Plot the locations and tour order
    for loc_name, (x, y) in locations.items():
        floor_x, floor_y = transform_path_to_floor_plan([(x, y)], grid_map.shape, raw_map.shape)[0]
        plt.scatter(floor_x, floor_y, c="blue", label=loc_name)
        plt.text(floor_x, floor_y, loc_name, color="black", fontsize=12)

    plt.axis("off")
    plt.title(f"Optimal Tour - {file_name}")
    if save:
        file_path = os.path.join(sys.path[0], f"../results/task2/{file_name}.png")
        plt.savefig(file_path)
    if show:
        plt.show()


def transform_path_to_floor_plan(path, grid_shape, floor_plan_shape):
    # Compute the scaling factors
    scale_x = floor_plan_shape[1] / grid_shape[1]
    scale_y = floor_plan_shape[0] / grid_shape[0]

    # Apply the scaling factors to the path coordinates
    transformed_path = [(int(x * scale_x), int(y * scale_y)) for x, y in path]

    return transformed_path


def calculate_total_distance(route, all_distances):
    total_distance = 0
    # Start from the 'start' and go to the first location
    total_distance += all_distances["start"][route[0]]
    # Sum the distances between the consecutive locations in the route
    for i in range(len(route) - 1):
        total_distance += all_distances[route[i]][route[i + 1]]
    # Return to start from the last location
    total_distance += all_distances[route[-1]]["start"]
    return total_distance


# Core functions -----------------------------------


def greedy_search(locations, all_distances):
    unvisited = set(locations.keys())
    current_location = "start"
    unvisited.remove(current_location)
    tour = [current_location]
    total_distance = 0

    while unvisited:
        min_dist = float("inf")
        next_loc = None
        for loc in unvisited:
            if all_distances[current_location][loc]["distance"] < min_dist:
                min_dist = all_distances[current_location][loc]["distance"]
                next_loc = loc
        total_distance += all_distances[current_location][loc]["distance"]
        unvisited.remove(next_loc)
        tour.append(next_loc)
        current_location = next_loc

    # Return to start
    total_distance += all_distances[current_location]["start"]["distance"]
    tour.append("start")

    return tour, total_distance


def held_karp(distances_matrix):
    n = len(distances_matrix)
    C = {}  # used to store the sub-problems' solutions

    # set the start point as 0
    for k in range(1, n):
        C[(1 << k, k)] = (distances_matrix[0][k], 0)

    # resolve using dynamic programming
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + distances_matrix[m][k], m))
                C[(bits, k)] = min(res)

    bits = (2**n - 1) - 1
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + distances_matrix[k][0], k))
    opt, parent = min(res)

    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    path.append(0)  #  add terminal
    path = list(reversed(path))
    path.append(0)  # add start
    return path, opt


def tsp_branch_and_bound(distances_matrix, key_index, keys):
    n = len(distances_matrix)
    pq = PriorityQueue()
    pq.put((0, [key_index["start"]], set(range(n)) - {key_index["start"]}))
    min_cost = float("inf")
    best_path = []

    while not pq.empty():
        cost, path, remaining = pq.get()
        if cost >= min_cost:
            continue
        if not remaining:
            final_cost = cost + distances_matrix[path[-1]][key_index["start"]]
            if final_cost < min_cost:
                min_cost = final_cost
                best_path = path
        else:
            for next_node in remaining:
                new_path = path + [next_node]
                new_cost = cost + distances_matrix[path[-1]][next_node]
                heuristic = min([distances_matrix[i][j] for i in new_path for j in remaining if i != j])
                if new_cost + heuristic < min_cost:
                    pq.put((new_cost, new_path, remaining - {next_node}))

    best_path = [keys[i] for i in best_path] + ["start"]
    return best_path, min_cost


def run(raw_map, grid_map, locations, all_paths, all_distances, method="greedy", save=False):
    st = time.time()
    if method == "greedy":
        optimal_tour, optimal_distance = greedy_search(locations, all_distances)
    elif method == "held_karp":
        distances_matrix, key_index, keys = transform_distance_dict(all_distances)
        optimal_path, optimal_distance = held_karp(distances_matrix)
        optimal_tour = [keys[i] for i in optimal_path]
    elif method == "tsp_branch_and_bound":
        distances_matrix, key_index, keys = transform_distance_dict(all_distances)
        optimal_tour, optimal_distance = tsp_branch_and_bound(distances_matrix, key_index, keys)
    duration = (time.time() - st) * 1000
    print("Optimal Tour:", optimal_tour)
    print("Optimal Distance:", optimal_distance)
    print(f"Time: {duration:.2f} ms")
    # plot
    file_path = os.path.join(sys.path[0], f"../results/task2/")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    plot_tour_on_map(raw_map, grid_map, locations, optimal_tour, all_paths, file_name=method, save=save, show=True)
    if save:
        tour_file = os.path.join(sys.path[0], f"../results/task2/{method}.json")
        tour_json = {"tour": optimal_tour, "distance": optimal_distance, "time": duration}
        with open(tour_file, "w") as f:
            json.dump(tour_json, f, indent=4)


def main():
    raw_map = imageio.imread(os.path.join(sys.path[0], "../map/vivocity.png"))
    grid_map_img = imageio.imread(os.path.join(sys.path[0], "../map/vivocity_dilate.png"))[:, :, 0]
    grid_map = grid_map_img.transpose()

    # load the path search results
    locations = {
        "start": [345, 95],
        "snacks": [470, 475],
        "store": [20, 705],
        "movie": [940, 545],
        "food": [535, 800],
    }
    dist_data_file = os.path.join(sys.path[0], "../results/task1/graph_search/search_strategy/astar/0result.json")
    path_data_dir = os.path.join(sys.path[0], "../results/task1/graph_search/search_strategy/astar/")
    all_paths = {}
    for file in os.listdir(path_data_dir):
        if file.endswith(".txt"):
            loc_from = file.split("_")[1]
            loc_to = file.split("_")[2].split(".")[0]
            with open(os.path.join(path_data_dir, file), "r") as f:
                path = f.readlines()
                path = [tuple(map(int, p.strip().split(","))) for p in path]
                all_paths[loc_from] = all_paths.get(loc_from, {})
                all_paths[loc_from][loc_to] = path
    all_distances = json.load(open(dist_data_file, "r"))

    # get and save the tour path
    run(raw_map, grid_map, locations, all_paths, all_distances, method="greedy", save=True)
    run(raw_map, grid_map, locations, all_paths, all_distances, method="held_karp", save=True)
    run(raw_map, grid_map, locations, all_paths, all_distances, method="tsp_branch_and_bound", save=True)


if __name__ == "__main__":
    main()