import open3d as o3d
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
# from osgeo import gdal
import time
import copy
from skimage import data,exposure
from skimage import measure
from skimage.morphology import opening,closing,erosion
from skimage.morphology import square
# from osgeo import gdal, gdal_array
import cv2
from scipy.signal import medfilt2d
from skimage.feature import peak_local_max
from scipy.ndimage import label
from skimage.segmentation import watershed
from skimage.measure import regionprops,find_contours
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from skopt.space import Space
from skopt.utils import use_named_args


#*******************Definition of basic functions***********************
def draw_geometries(result,batch):
    if batch:
        for i in range(len(result)):
            o3d.visualization.draw_geometries([result[i]], "result", 800, 600			
				, 50, 50, False, False, True)
    else:
        o3d.visualization.draw_geometries(result, "result", 800, 600
			, 50, 50, False, False, True)

#*******************Display of registration results***********************
def draw_registration_result(source, target, transformation):
    source_temp = o3d.geometry.PointCloud(source)
    target_temp = o3d.geometry.PointCloud(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) 
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],"result", 1000, 800,
                                      50, 50, False, False, True,
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def FPFH_Compute(pcd, voxel_size):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    #Parameter tuning: search neighborhood radius and maximum number of neighbors.
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_fpfh

#*******************Iterative least squares method***********************
def Iterative_least_squares(source_down, target_down, threshold):
    target_kdtree = o3d.geometry.KDTreeFlann(target_down)
    source_points = np.asarray(source_down.points)
    total_squared_distance = 0
    inlier_count = 0

    for point in source_points:
        [_, idx, distances] = target_kdtree.search_knn_vector_3d(point, 1)
        nearest_distance = np.sqrt(distances[0])
        
        if nearest_distance <= threshold:
            total_squared_distance += nearest_distance**2
            inlier_count += 1

    if inlier_count == 0:
        return float('inf')  
    rmse = np.sqrt(total_squared_distance / inlier_count)
    return rmse

def compute_inlier_rmse(source, target, transformation, voxel_size):
    
    source_temp = source.transform(transformation)  
    threshold = voxel_size * 2.0  
    inlier_rmse = Iterative_least_squares(source_temp, target, threshold)  
    return inlier_rmse

#*******************Truncated least squares method***********************
def Truncated_least_squares(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5 # Parameter setting: maximum allowable distance between two points.
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)) #Parameter tuning: maximum number of iterations (default: 100,000) and confidence level.
    
    
    return ransac_result

#**********************Feature matching***********************
def feature_matching(source, target, voxel_size, max_iterations=50, tolerance=1e-6):
    # Construct a KD-Tree for efficient nearest neighbor search in the target point cloud.
    target_kdtree = o3d.geometry.KDTreeFlann(target)

    # Initialize the transformation matrix as the identity matrix.
    transformation = np.eye(4)

    # Extract the original coordinates of the source point cloud.
    source_points = np.asarray(source.points)
    previous_transformation = np.eye(4)

    for iteration in range(max_iterations):
        # Step 1: Apply the current transformation to the source point cloud.
        source_points_transformed = np.dot(transformation[:3, :3], source_points.T).T + transformation[:3, 3]

        # Step 2: For each transformed point, identify its nearest neighbor in the target point cloud.
        indices = []
        for point in source_points_transformed:
            _, idx, _ = target_kdtree.search_knn_vector_3d(point, 1)
            indices.append(idx[0])
        target_points_matched = np.asarray(target.points)[indices]

        # Step 3: Compute the centroids and center both point clouds accordingly.
        source_centroid = np.mean(source_points_transformed, axis=0)
        target_centroid = np.mean(target_points_matched, axis=0)
        source_centered = source_points_transformed - source_centroid
        target_centered = target_points_matched - target_centroid

        # Step 4: Compute the cross-covariance matrix and solve for the optimal rotation and translation using singular value decomposition (SVD).
        H = np.dot(source_centered.T, target_centered)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = target_centroid - np.dot(R, source_centroid)

        # Step 5: Construct the updated transformation matrix based on the estimated rigid motion.
        delta_transform = np.eye(4)
        delta_transform[:3, :3] = R
        delta_transform[:3, 3] = t
        transformation = delta_transform @ transformation  # 累计变换

        # Step 6: Evaluate convergence based on the change in transformation between iterations.
        delta_translation = np.linalg.norm(previous_transformation[:3, 3] - transformation[:3, 3])
        if delta_translation < tolerance:
            break

        previous_transformation = transformation.copy()

   
    aligned_source = copy.deepcopy(source)
    aligned_source.transform(transformation)

    return aligned_source, transformation

#*******************The selection of optimal voxel size *****************************************
def compute_rmse_with_bo(source, target, n_calls=10, n_random_starts=5):
    result_df = pd.DataFrame(columns=['Voxel Size', 'Inlier RMSE'])

    def objective(voxel_size):
        voxel_size = voxel_size[0]  # 解包 skopt 传入的 list
        
        print(f"Processing voxel_size = {voxel_size:.4f}")

        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        source_fpfh = FPFH_Compute(source_down, voxel_size)
        target_fpfh = FPFH_Compute(target_down, voxel_size)

        result_ransac = Truncated_least_squares(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        transformation = result_ransac.transformation

        inlier_rmse = compute_inlier_rmse(source, target, transformation, voxel_size)
        print(f"Voxel Size: {voxel_size:.4f}, Inlier RMSE: {inlier_rmse:.4f}")

        new_data = pd.DataFrame({
            'Voxel Size': [voxel_size],
            'Inlier RMSE': [inlier_rmse]
        })
        nonlocal result_df
        result_df = pd.concat([result_df, new_data], ignore_index=True)

        return inlier_rmse
    
    search_space = [Real(1.2, 2.5, name='voxel_size')]

    result = gp_minimize(
        objective,
        search_space,
        acq_func="EI",
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        random_state=42
    )

    optimal_voxel_size = round(result.x[0], 1)
    min_inlier_rmse = result.fun

    print(f"\nOptimal Voxel Size: {optimal_voxel_size:.1f} with Inlier RMSE: {min_inlier_rmse:.4f}")

    return optimal_voxel_size, result_df

#*******************Path setting and data reading***********************
input_path = 'PATH_TO_YOUR_DATA'
output_path = 'PATH_TO_YOUR_OUTPUT_PATH'
HMLS_name = 'HMLS_NAME'
ALS_name = 'UAV_NAME'
HMLS_pcd = o3d.io.read_point_cloud(os.path.join(input_path, HMLS_name+'.pcd'))  # HMLS data
ALS_pcd = o3d.io.read_point_cloud(os.path.join(input_path, ALS_name+'.pcd'))    # UAV data
source = HMLS_pcd  # Source point cloud
target = ALS_pcd   # Target point cloud

#*******************First stage optimal voxel selection and feature extraction method*****************************************
optimal_voxel_size, _ = compute_rmse_with_bo(HMLS_pcd, ALS_pcd)


# Downsampling
source_down = source.voxel_down_sample(optimal_voxel_size)  
target_down = target.voxel_down_sample(optimal_voxel_size) 

# FPFH feature extraction
source_fpfh = FPFH_Compute(source_down,optimal_voxel_size)  
target_fpfh = FPFH_Compute(target_down, optimal_voxel_size)  

#*******************Second stage gross error elimination and feature matching method*****************************************
result_ransac = Truncated_least_squares(source_down, target_down, source_fpfh, target_fpfh, optimal_voxel_size)

draw_geometries([source,target],False)
draw_registration_result(source, target, result_ransac.transformation)

source_temp = copy.deepcopy(source)
source_temp.transform(result_ransac.transformation) 
o3d.io.write_point_cloud(os.path.join(output_path, HMLS_name+'_coarse_registration.pcd'), source_temp)  

source_down_temp = copy.deepcopy(source_down)
source_down_temp.transform(result_ransac.transformation)

aligned_source, final_transformation = feature_matching(source_down_temp, target_down, optimal_voxel_size)

source_temp2 = copy.deepcopy(source_temp)
source_temp2.transform(final_transformation)

o3d.io.write_point_cloud(os.path.join(output_path, HMLS_name+'_fine_registration.pcd'), source_temp2)

draw_geometries([source_temp2, target], False)

#*******************Merged point cloud*****************************************
pcd1 = o3d.io.read_point_cloud('PATH_TO_UAV_DATA')
pcd2 = o3d.io.read_point_cloud('PATH_TO_FINE_REGISTRATION_HMLS_DATA')
merged_pcd = pcd1 + pcd2
o3d.io.write_point_cloud('PATH_TO_MERGED_DATA', merged_pcd)