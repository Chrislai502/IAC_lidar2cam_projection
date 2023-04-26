import numpy as np
import matplotlib.pyplot as plt
import math
import random
# import open3d as o3d
import pickle
import os

print(os.getcwd())


class RectangleData():

    def __init__(self):
        # 4 lines of ax + by + c = 0
        self.a = [None] * 4
        self.b = [None] * 4
        self.c = [None] * 4

        self.rect_c_x = [None] * 4
        self.rect_c_y = [None] * 4


    def calc_rect_contour(self):

        self.rect_c_x[0], self.rect_c_y[0] = self.find_intersection(self.a[0:2], self.b[0:2], self.c[0:2])

        self.rect_c_x[1], self.rect_c_y[1] = self.find_intersection(self.a[1:3], self.b[1:3], self.c[1:3])
        
        self.rect_c_x[2], self.rect_c_y[2] = self.find_intersection(self.a[2:4], self.b[2:4], self.c[2:4])
        
        self.rect_c_x[3], self.rect_c_y[3] = self.find_intersection([self.a[3], self.a[0]], [self.b[3], self.b[0]], [self.c[3], self.c[0]])
        

    def find_intersection(self, a, b, c):
        x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
        y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
        return x, y

    def rect_corners_to_bbox(self):
        self.calc_rect_contour()
        corners = np.array([
            [self.rect_c_x[0], self.rect_c_y[0], 0],
            [self.rect_c_x[1], self.rect_c_y[1], 0],
            [self.rect_c_x[2], self.rect_c_y[2], 0],
            [self.rect_c_x[3], self.rect_c_y[3], 0]
        ])

        # return corners

        # Compute the center of the rectangle
        center = corners.mean(axis=0)

        # Compute the principal axes of the covariance matrix of the corners
        covariance = np.cov(corners.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        axes = eigenvectors.T

        # Compute the dimensions of the rectangle
        distances = np.abs(corners - center)
        dimensions = 2 * np.max(distances.dot(axes), axis=0)

        # # Create the OrientedBoundingBox object
        # bbox = o3d.geometry.OrientedBoundingBox(center=center, R=axes, extent=dimensions)
        # return center, axes, dimensions
        return center, corners, dimensions
    
    

def main(args=None):


    # Function to fit an L-shaped model to a point cloud
    def lshapedfitting(X):

            
        # cluster = o3d_pcd.select_by_index(list(np.where(labels == i)[0]))

        # cluster_bbox = cluster.get_oriented_bounding_box()
        dtheta = np.deg2rad(1.0)
        min_dist_of_closeness_crit = 0.01 #[m]
        minp = (-float('inf'), None)
        for theta in np.arange(0.0, np.pi / 2.0 - dtheta, dtheta):
            e1 = np.array([np.cos(theta), np.sin(theta)])
            e2 = np.array([-np.sin(theta), np.cos(theta)])
            c1 = X @ e1.T
            
            c2 = X @ e2.T

            c1_max = max(c1)
            c2_max = max(c2)
            c1_min = min(c1)
            c2_min = min(c2)

            D1 = [min([np.linalg.norm(c1_max - ic1),
                    np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
            D2 = [min([np.linalg.norm(c2_max - ic2),
                    np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]

            cost = 0
            for i, _ in enumerate(D1):
                d = max(min([D1[i], D2[i]]), min_dist_of_closeness_crit)
                cost += (1.0 / d)

            if minp[0] < cost:
                minp = (cost, theta)
        print(minp[1])
        # calculate best rectangle
        sin_s = np.sin(minp[1])
        cos_s = np.cos(minp[1])

        c1_s = X @ np.array([cos_s, sin_s]).T
        c2_s = X @ np.array([-sin_s, cos_s]).T

        rect = RectangleData()
        rect.a[0] = cos_s
        rect.b[0] = sin_s
        rect.c[0] = min(c1_s)
        rect.a[1] = -sin_s
        rect.b[1] = cos_s
        rect.c[1] = min(c2_s)
        rect.a[2] = cos_s
        rect.b[2] = sin_s
        rect.c[2] = max(c1_s)
        rect.a[3] = -sin_s
        rect.b[3] = cos_s
        rect.c[3] = max(c2_s)
        return rect.rect_corners_to_bbox()

    # Set the path to your .pkl file
    file_path = "chris_to_sean(Close).pkl"

    # Define the output file format
    combined_output_file = "combined_output.txt"

    # Load the point cloud data from the file
    with open(file_path, 'rb') as f:
        pointcloud_data = pickle.load(f)
    dim1 = len(pointcloud_data)
    dim2 = len(pointcloud_data[0])
    dim3 = len(pointcloud_data[0][0])
    print(dim1, dim2, dim3)
    # print(pointcloud_data[0][0])


    # Loop over each time step and point cloud, and apply the lshapedfitting() function
    with open(combined_output_file, 'w') as f:
        for i in range(len(pointcloud_data)):
            labels = pointcloud_data[i][0]
            # print(labels)
            for current_label in range(0, max(labels) + 1):
                print(f'time {i} label {current_label}')
                current_pointcloud = pointcloud_data[i][1]

                cluster = [point for (index, point) in enumerate(current_pointcloud) if labels[index] == current_label]

                cluster_xy = np.array(cluster)[:, [1, 2]]  # Remove the z-axis
                # print(cluster_xy)
                # print(cluster_xy)
                # bbox = lshapedfitting(pointcloud_xyz)
                # f.write(f"Time {time[i]}, Label {label}, Center {bbox.get_center()}, Corners {bbox.get_box_points()}")
                center, corners, dimensions = lshapedfitting(cluster_xy)
                f.write(f"Time {i}, Label {current_label}:\n Center {center}\n Corners {corners}\n Dimensions {dimensions}\n")
                # corners = lshapedfitting(cluster_xy)
                # f.write(f"Time {i}, Label {current_label}\n Corners {corners}\n")
                

            

if __name__ == '__main__':
    main()