#include "clustering/dbscan.hpp"
#include <queue>
#include <cmath>

std::vector<std::vector<int>> dbscan(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree,
    float base_eps,
    int minPts)

{
    const int UNVISITED = -1;
    const int NOISE = -2;

    std::vector<int> labels(cloud->size(), UNVISITED);
    int cluster_id = 0;

    std::vector<std::vector<int>> clusters;

    for (size_t i = 0; i < cloud->size(); ++i) {
        if (labels[i] != UNVISITED)
            continue;

        const auto& p = cloud->points[i];
        float distance = std::sqrt(p.x*p.x + p.y*p.y);

        float eps = base_eps * (1.0f + distance / 50.0f);
        eps = std::max(0.5f, std::min(eps, 1.5f));

        std::vector<int> neighbors;
        std::vector<float> distances;
        tree->radiusSearch(i, eps, neighbors, distances);

        if (neighbors.size() < static_cast<size_t>(minPts)) {
            labels[i] = NOISE;
            continue;
        }

        clusters.emplace_back();
        auto& cluster = clusters.back();

        std::queue<int> q;
        q.push(i);
        labels[i] = cluster_id;

        while (!q.empty())
        {
            int idx = q.front(); // Returns first point of queue without removing it
            q.pop(); // Removes front point from queue

            cluster.push_back(idx);

            const auto& pt = cloud->points[idx];
            float d = std::sqrt(pt.x * pt.x + pt.y * pt.y);
            float local_eps = base_eps * (1.0f + d / 50.0f);
            local_eps = std::max(0.5f, std::min(local_eps, 1.5f));

            std::vector<int> nbrs;
            std::vector<float> nbr_dist;
            tree->radiusSearch(idx, local_eps, nbrs, nbr_dist);

            if (nbrs.size() >= static_cast<size_t>(minPts)) {
                for (int n_idx : nbrs) {
                    if (labels[n_idx] == UNVISITED || labels[n_idx] == NOISE) {
                        if (labels[n_idx] == UNVISITED) {
                            q.push(n_idx);
                        }
                        labels[n_idx] = cluster_id;
                    }
                }
            }
        }

        cluster_id++;
    }

    return clusters;
}