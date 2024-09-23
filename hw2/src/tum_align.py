#!/usr/bin/env python

import numpy as np


# Read text files in TUM format
def read_tum_file(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items = line.split()
            timestamp = float(items[0])
            x, y, z, qx, qy, qz, qw = map(float, items[1:])
            data.append((timestamp, x, y, z, qx, qy, qz, qw))
    return data


# Save data to text files in TUM format
def save_tum_file(data, file_path):
    with open(file_path, "w") as file:
        for item in data:
            timestamp, x, y, z, qx, qy, qz, qw = item
            file.write(
                "{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(timestamp, x, y, z, qx, qy, qz, qw)
            )


# Process data using rotation matrix and translation vector
def transform_data(data, rotation_matrix, translation_vector):
    transformed_data = []
    for item in data:
        timestamp, x, y, z, qx, qy, qz, qw = item
        position = np.array([x, y, z])
        rotated_position = np.dot(rotation_matrix, position) + translation_vector
        transformed_data.append(
            (timestamp, rotated_position[0], rotated_position[1], rotated_position[2], qx, qy, qz, qw)
        )
    return transformed_data


if __name__ == "__main__":
    # Read text files in TUM format
    input_file_path = "fast"
    data = read_tum_file(input_file_path + ".txt")

    # Given rotation matrix and translation vector
    rotation_matrix = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    translation_vector = np.array([0, 0.08, -0.27])

    # Process data using rotation matrix and translation vector
    transformed_data = transform_data(data, rotation_matrix, translation_vector)

    # Save processed data to text files in TUM format
    output_file_path = input_file_path + "_aligned.txt"
    save_tum_file(transformed_data, output_file_path)
