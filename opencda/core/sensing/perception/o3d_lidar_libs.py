# -*- coding: utf-8 -*-
"""
Utility functions for 3d lidar visualization
and processing by utilizing open3d.
"""

# Author: CARLA Team, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import time

import numpy as np
import open3d as o3d
from matplotlib import cm
from scipy.stats import mode
from opencda.customize.v2x.aux import PLDMentry
import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.sensing.perception.obstacle_vehicle import \
    is_vehicle_cococlass, ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import StaticObstacle

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255),  # None
    (70, 70, 70),  # Building
    (100, 40, 40),  # Fences
    (55, 90, 80),  # Other
    (220, 20, 60),  # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),  # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),  # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),  # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),  # Water
    (145, 170, 100),  # Terrain
]) / 255.0  # normalize each channel [0-1] since is what Open3D uses


def o3d_pointcloud_encode(raw_data, point_cloud):
    """
    Encode the raw point cloud(np.array) to Open3d PointCloud object.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw lidar points, (N, 4).

    point_cloud : o3d.PointCloud
        Open3d PointCloud.

    """

    # Isolate the intensity and compute a color for it
    intensity = raw_data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = np.array(raw_data[:, :-1], copy=True)
    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(int_color)


def o3d_visualizer_init(actor_id):
    """
    Initialize the visualizer.

    Parameters
    ----------
    actor_id : int
        Ego vehicle's id.

    Returns
    -------
    vis : o3d.visualizer
        Initialize open3d visualizer.

    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=str(actor_id),
                      width=800,
                      height=800,
                      left=480,
                      top=270)
    vis.get_render_option().background_color = [0, 0, 0]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True

    return vis


def o3d_visualizer_show(vis, count, point_cloud, objects, LDM=False):
    """
    Visualize the point cloud at runtime.

    Parameters
    ----------
    LDM
    vis : o3d.Visualizer
        Visualization interface.

    count : int
        Current step since simulation started.

    point_cloud : o3d.PointCloud
        Open3d point cloud.

    objects : dict
        The dictionary containing objects.

    Returns
    -------

    """
    point_cloud.paint_uniform_color([1, 1, 0])
    if count == 2:
        vis.add_geometry(point_cloud)

    vis.update_geometry(point_cloud)

    LDM_geometries = []
    for key, object_list in objects.items():
        # we only draw vehicles for now
        if key != 'vehicles':
            continue
        for object_ in object_list:
            if LDM is True:
                # min_arr = object_.min.reshape((3, 1))
                # max_arr = object_.max.reshape((3, 1))
                # min_bound = min_arr.astype(np.float64)
                # max_bound = max_arr.astype(np.float64)
                # geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                geometry = object_.o3d_bbx
                if object_.detected:
                    geometry.color = (0, 1, 0)
                else:
                    geometry.color = (0, 1, 1)
                vis.add_geometry(geometry)
                LDM_geometries.append(geometry)
            else:
                if object_.o3d_obb is not None:
                    object_.o3d_obb.color = (1, 0, 0)
                    vis.add_geometry(object_.o3d_obb)
                else:
                    aabb = object_.o3d_bbx
                    vis.add_geometry(aabb)

    min = np.array([1, 1, 1])
    max = np.array([2, 2, 2])

    vis.poll_events()
    vis.update_renderer()
    # # This can fix Open3D jittering issues:
    time.sleep(0.001)

    for key, object_list in objects.items():
        if key != 'vehicles':
            continue
        for object_ in object_list:
            if object_.o3d_obb is not None:
                vis.remove_geometry(object_.o3d_obb)
            else:
                aabb = object_.o3d_bbx
                vis.remove_geometry(aabb)

    for geometry in LDM_geometries:
        vis.remove_geometry(geometry)


def o3d_visualizer_showLDM(vis, count, point_cloud, objects, groundTruth):
    """
    Visualize the point cloud at runtime.

    Parameters
    ----------
    groundTruth
    LDM
    vis : o3d.Visualizer
        Visualization interface.

    count : int
        Current step since simulation started.

    point_cloud : o3d.PointCloud
        Open3d point cloud.

    objects : dict
        The dictionary containing objects.

    Returns
    -------

    """
    point_cloud.paint_uniform_color([0, 1, 1])
    if count == 10:
        vis.add_geometry(point_cloud)

    vis.update_geometry(point_cloud)

    LDM_geometries = []
    for key, object_list in objects.items():
        # we only draw vehicles for now
        if key != 'vehicles':
            continue
        for object_ in object_list:
            if object_.perception.line_set is not None:
                geometry = object_.perception.line_set
                if not object_.tracked:
                    continue
                if object_.detected and object_.onSight:
                    if object_.CPM:
                        colors = [[1, 0.6, 0] for _ in range(12)]
                        geometry.colors = o3d.utility.Vector3dVector(colors)
                    else:
                        colors = [[1, 0, 0] for _ in range(12)]
                        geometry.colors = o3d.utility.Vector3dVector(colors)
                elif not object_.detected:
                    colors = [[0, 1, 0] for _ in range(12)]
                    geometry.colors = o3d.utility.Vector3dVector(colors)
                elif object_.CPM:
                    colors = [[1, 1, 0] for _ in range(12)]
                    geometry.colors = o3d.utility.Vector3dVector(colors)
                elif isinstance(object_, PLDMentry) and object_.assignedPM is not None:
                    colors = [[1, 0, 0] for _ in range(12)]
                    geometry.colors = o3d.utility.Vector3dVector(colors)
                else:
                    colors = [[0.5, 0, 0] for _ in range(12)]
                    geometry.colors = o3d.utility.Vector3dVector(colors)
            else:
                geometry = object_.perception.o3d_bbx
                if not object_.tracked:
                    continue
                if object_.detected and object_.onSight:
                    geometry.color = (1, 0, 0)
                elif not object_.detected:
                    geometry.color = (0, 1, 0)
                elif object_.CPM:
                    geometry.color = (0.7, 0, 0)
                elif isinstance(object_, PLDMentry) and object_.assignedPM is not None:
                    geometry.color = (1, 0, 0)
                else:
                    geometry.color = (0.5, 0, 0)
            vis.add_geometry(geometry)

            LDM_geometries.append(geometry)

    # vis.add_geometry(test_rotation())

    for key, object_list in groundTruth.items():
        # we only draw vehicles for now
        if key != 'vehicles':
            continue
        for object_ in object_list:
            geometry = object_.o3d_bbx
            geometry.color = (0.3, 0.3, 0.3)
            vis.add_geometry(geometry)
            LDM_geometries.append(geometry)

    vis.poll_events()
    vis.update_renderer()
    # # This can fix Open3D jittering issues:
    time.sleep(0.001)

    for geometry in LDM_geometries:
        vis.remove_geometry(geometry)


def test_rotation():
    box = [0, 0, 0, 3, 5, 2, np.deg2rad(45)]
    corner_boxes = np.zeros((8, 3))

    translation = box[0:3]
    h, w, l = box[3], box[4], box[5]
    rotation = box[6]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    corner_box = corner_box.transpose()

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner_box)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def o3d_camera_lidar_fusion(objects,
                            yolo_bbx,
                            lidar_3d,
                            projected_lidar,
                            lidar_sensor,
                            ego_pos = None):
    """
    Utilize the 3D lidar points to extend the 2D bounding box
    from camera to 3D bounding box under world coordinates.

    Parameters
    ----------
    objects : dict
        The dictionary contains all object detection results.

    yolo_bbx : torch.Tensor
        Object detection bounding box at current photo from yolov5,
        shape (n, 5)->(n, [x1, y1, x2, y2, label])

    lidar_3d : np.ndarray
        Raw 3D lidar points in lidar coordinate system.

    projected_lidar : np.ndarray
        3D lidar points projected to the camera space.

    lidar_sensor : carla.sensor
        The lidar sensor.

    ego_pos : carla.transform
        The ego vehicle's transform.
    Returns
    -------
    objects : dict
        The update object dictionary that contains 3d bounding boxes.
    """

    # convert torch tensor to numpy array first
    if yolo_bbx.is_cuda:
        yolo_bbx = yolo_bbx.cpu().detach().numpy()
    else:
        yolo_bbx = yolo_bbx.detach().numpy()

    for i in range(yolo_bbx.shape[0]):
        detection = yolo_bbx[i]
        # 2d bbx coordinates
        x1, y1, x2, y2 = int(detection[0]), int(detection[1]), \
            int(detection[2]), int(detection[3])
        label = int(detection[5])
        confidence = float(detection[4])

        # choose the lidar points in the 2d yolo bounding box
        points_in_bbx = \
            (projected_lidar[:, 0] > x1) & (projected_lidar[:, 0] < x2) & \
            (projected_lidar[:, 1] > y1) & (projected_lidar[:, 1] < y2) & \
            (projected_lidar[:, 2] > 0.0)
        # ignore intensity channel
        select_points = lidar_3d[points_in_bbx][:, :-1]

        if select_points.shape[0] == 0:
            continue

        # filter out the outlier
        x_common = mode(np.array(np.abs(select_points[:, 0]),
                                 dtype=np.int), axis=0)[0][0]
        y_common = mode(np.array(np.abs(select_points[:, 1]),
                                 dtype=np.int), axis=0)[0][0]
        points_inlier = (np.abs(select_points[:, 0]) > x_common - 3) & \
                        (np.abs(select_points[:, 0]) < x_common + 3) & \
                        (np.abs(select_points[:, 1]) > y_common - 3) & \
                        (np.abs(select_points[:, 1]) < y_common + 3)
        select_points = select_points[points_inlier]

        if select_points.shape[0] < 4:
            continue

        # to visualize 3d lidar points in o3d visualizer, we need to
        # revert the x coordinates
        select_points[:, :1] = -select_points[:, :1]

        # create o3d.PointCloud object
        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(select_points)
        o3d_pointcloud.paint_uniform_color([0, 1, 1])
        # add o3d bounding box
        aabb = o3d_pointcloud.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)

        obb = None
        if np.asarray(o3d_pointcloud.points).shape[0] >= 4:
            try:
                obb = o3d_pointcloud.get_oriented_bounding_box()
                obb.color = (0, 1, 0)
            except RuntimeError as e:
                # print("Unable to compute the oriented bounding box:", e)
                pass

        # get the eight corner of the bounding boxes.
        corner = np.asarray(aabb.get_box_points())
        # covert back to unreal coordinate
        corner[:, :1] = -corner[:, :1]
        corner = corner.transpose()
        # extend (3, 8) to (4, 8) for homogenous transformation
        corner = np.r_[corner, [np.ones(corner.shape[1])]]
        # project to world reference
        corner = st.sensor_to_world(corner, lidar_sensor.get_transform())
        corner = corner.transpose()[:, :3]

        if is_vehicle_cococlass(label):
            obstacle_vehicle = ObstacleVehicle(corner, aabb, confidence=confidence)
            if obb is not None:
                obstacle_vehicle.o3d_obb = obb
                obstacle_vehicle.bounding_box.extent.x = obb.extent[0]/2
                obstacle_vehicle.bounding_box.extent.y = obb.extent[1]/2
                yaw = ego_pos.rotation.yaw - np.degrees(np.arctan2(np.array(obb.R)[1, 0], np.array(obb.R)[0, 0]))
                obstacle_vehicle.yaw = yaw
            if 'vehicles' in objects:
                objects['vehicles'].append(obstacle_vehicle)
            else:
                objects['vehicles'] = [obstacle_vehicle]
        # todo: refine the category
        # we regard or other obstacle rather than vehicle as static class
        else:
            static_obstacle = StaticObstacle(corner, aabb)
            if 'static' in objects:
                objects['static'].append(static_obstacle)
            else:
                objects['static'] = [static_obstacle]

    return objects
