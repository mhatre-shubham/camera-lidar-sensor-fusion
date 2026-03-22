#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

class Track{
public:
    int track_id;
    int age; // how many frames the object has existed
    int hits; // how many times the object has been matched to a detection
    int time_since_update; // frames since the last match
    std::vector<double> size; // 3D bounding box

    // Kalman Filter matrices
    Eigen::VectorXd x; // State: [x, y, z, vx, vy, vz]
    Eigen::MatrixXd P; // State covariance
    Eigen::MatrixXd F; // State transition matrix. Constant velocity model
    Eigen::MatrixXd H; // Measurement matrix
    Eigen::MatrixXd R; // Measurement noise covariance
    Eigen::MatrixXd Q; // Process noise covariance
    Eigen::MatrixXd I; // Identity matrix

    // Constructor declaration. Defines parameters that are passed when creating a new Track object
    Track(const Eigen::Vector3d& detection, const std::vector<double>& dimensions, int id) 
        : track_id(id), age(1), hits(1), time_since_update(0), size(dimensions) 

    {
        x = Eigen::VectorXd::Zero(6);
        x.head(3) = detection;

        P = Eigen::MatrixXd::Identity(6,6) * 10.0;

        F = Eigen::MatrixXd::Identity(6,6); // State Matrix. dt added during predict

        H = Eigen::MatrixXd::Zero(3,6);
        H(0,0) = 1.0;
        H(1,1) = 1.0;
        H(2,2) = 1.0;

        R = Eigen::MatrixXd::Identity(3,3) * 0.5; // From Sensor specs 
        Q = Eigen::MatrixXd::Identity(6, 6) * 0.1;
        I = Eigen::MatrixXd::Identity(6, 6);
    }

    Eigen::Vector3d predict(double dt){
        // Update physics model with actual dt
        F(0, 3) = dt;
        F(1, 4) = dt;
        F(2, 5) = dt;

        x = F * x;
        P = F * P * F.transpose() + Q;

        age += 1;
        time_since_update += 1;
        return x.head(3);
    }

    void update(const Eigen::Vector3d& detection, const std::vector<double>& dimensions){
        Eigen::Vector3d y = detection - (H * x);// Innvation Residual
        Eigen::MatrixXd S = H * P * H.transpose() + R; //Innovation Covariance
        Eigen::MatrixXd K = P * H.transpose() * S.inverse(); // Kalman Gain

        x = x + (K * y); // Update State
        P = (I - K * H) * P; // Update Covariance

        size = dimensions;
        time_since_update = 0;
        hits += 1;
    }
};

class ObjectTrackingNode : public rclcpp::Node{
public:
    ObjectTrackingNode() : Node("lidar_tracking_node"), next_track_id_(0), last_timestamp_(-1.0) {

        this->declare_parameter<int>("max_age", 3); // How long a track survives without updates
        this->get_parameter("max_age", max_age_);

        this->declare_parameter<double>("distance_threshold", 4.0);
        this->get_parameter("distance_threshold", distance_threshold_);

        subscription_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/lidar/bounding_boxes",
            10,
            std::bind(&ObjectTrackingNode::detection_callback, this, std::placeholders::_1));

        tracked_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/lidar/tracked_objects", 10);

        RCLCPP_INFO(this->get_logger(), "Lidar Tracker Node Started...");
}

private:
    std::vector<Track> tracks_;
    int next_track_id_;
    double last_timestep_;

    // Tuned Parameters
    int max_age_;
    double distance_threshold_;

    void detection_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg){
        if (msg->markers.empty())
            return;

        double current_time_float = 0.0;
        std::string current_frame = "map";
        builtin_interfaces::msg::Time stamp;
        bool stamp_found = false;

        // Extract raw detections and timestamp
        std::vector<Eigen::Vector3d> detections;
        std::vector<std::vector<double>> dimensions;

        for (const auto& marker : msg->markers){
            if (marker.action == visualization_msgs::msg::Marker::ADD){
                if (!stamp_found){
                    stamp = marker.header.stamp;
                    current_frame = marker.header.frame_id;
                    current_time_float = stamp.sec + (stamp.nanosec * 1e-9);
                    stamp_found = true;
                }
                detections.push_back(Eigen::Vector3d(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z));
                dimensions.push_back({marker.scale.x, marker.scale.y, marker.scale.z});
            }
        }

        if (!stamp_found)
            return;

        // Calcualte dynamic dt
        double dt = 0.1;
        if (last_timestep_ > 0.0){
            dt = current_time_float - last_timestep_;
            if (dt <= 0.0 || dt > 2.0) 
                dt = 0.1; // Fallback for loops/pauses
        }
        last_timestep_ = current_time_float;
        
        // Predict
        std::vector<Eigen::Vector3d> predictions;
        for (auto& track : tracks_) {
            predictions.push_back(track.predict(dt));
        }

        // Data Association (Greedy Nearest Neighbor)
        /*
        -- Compute distance between each predicted track and detection
        -- Apply distance threshold (gating)
        -- Store matches in MatchCost struct
        -- Assign tracks to detections using Greedy or Hungarian
        -- Identify unmatched tracks and unmatched detections
        */
        
        std::vector<std::pair<int, int>> matched_indices;
        std::vector<int> unmatched_detections;
        std::vector<int> unmatched_tracks;
            
        // If we have existing tracks and new detections
        if (!tracks_.empty() && !detections.empty()){
            struct MatchCost{
                int trk_idx;
                int det_idx;
                double cost;
            };
            std::vector<MatchCost> costs;

            for (size_t t = 0; t < predictions.size(); ++t) {
                for (size_t d = 0; d < detections.size(); ++d) {
                    double dist = (predictions[t] - detections[d]).norm();
                    if (dist < distance_threshold_) {
                        costs.push_back({(int)t, (int)d, dist});
                    } // Only consider nearby detections for matching
                }
            }

            // Sort by lowest cost (closest pair first)
            std::sort(costs.begin(), costs.end(), [](const MatchCost& a, const MatchCost& b) {
                return a.cost < b.cost;
            });

            std::vector<bool> trk_assigned(tracks_.size(), false);
            std::vector<bool> det_assigned(detections.size(), false);

            for (const auto& c : costs){
                if (!trk_assigned[c.trk_idx] && !det_assigned[c.det_idx]){
                    matched_indices.push_back({c.trk_idx, c.det_idx});
                    trk_assigned[c.trk_idx] = true;
                    det_assigned[c.det_idx] = true;
                }
            }

            for (size_t t = 0; t < tracks_.size(); t++){
                if (!trk_assigned[t]){
                    unmatched_tracks.push_back(t);
                }
            }

            for (size_t d = 0; d < detections.size(); d++){
                if (!det_assigned[d]){
                    unmatched_detections.push_back(d);
                }
            }
        }

        else{
            for(size_t d = 0; detections.size(); d++) unmatched_detections.push_back(d);
            for(size_t t = 0; tracks_.size(); t++) unmatched_tracks.push_back(t);
        }

        // Update matched tracks
        for (const auto& match : matched_indices){
            tracks_[match.first].update(detections[match.second], dimensions[match.second]);
        }

        // Create new tracks
        for (int det_idx : unmatched_detections){
            tracks_.push_back(Track(detections[det_idx], dimensions[det_idx], next_track_id_++));
        }

        // Delete dead tracks
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const Track& t) { return t.time_since_update > max_age_; }), tracks_.end());

    }












}