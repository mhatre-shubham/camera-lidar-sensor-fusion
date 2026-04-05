#ifndef DBSCAN_HPP_
#define DBSCAN_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <vector>

std::vector<std::vector<int>> dbscan(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree,
    float base_eps,
    int minPts);

#endif  // DBSCAN_HPP_