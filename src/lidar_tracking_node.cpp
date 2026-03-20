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

        F = Eigen::MatrixXd::Identity(6,6); // dt added during predict

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