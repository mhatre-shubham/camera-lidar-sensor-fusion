#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>

class LidarCameraFusionNode : public rclcpp::Node
{
public:
    LidarCameraFusionNode() : Node("lidar_camera_fusion_node")
    {
        this->declare_parameter("overlap_threshold", 0.15);
        overlap_threshold_ = this->get_parameter("overlap_threshold").as_double();

        setup_calibration_matrices();

        yolo_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
            "/camera/object_detection", 10, std::bind(&LidarCameraFusionNode::yolo_callback, this, std::placeholders::_1));

        mot_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/lidar/tracked_objects", 10, std::bind(&LidarCameraFusionNode::mot_callback, this, std::placeholders::_1));
            
        pub_identified_ = this->create_publisher<visualization_msgs::msg::MarkerArray(
            "/fusion/identified_objects", 10);

        pub_unknown_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/fusion/unknown_objects", 10);
    
        RCLCPP_INFO(this->get_logger(), "Lidar Camera Fusion Node Started...");
    }

private:
    double overlap_threshold_;
    cv::Mat projection_matrix_;
    vision_msgs::msg::Detection2DArray latest_yolo_detections_;

    std::map<int, std::string> label_memory_;
    std::set<int> active_ids_;

    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr yolo_sub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr mot_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_identified_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_unknown_;

    void setup_calibration_matrices()
    {
        cv::Mat Tr = cv::Mat::eye(4, 4, CV_64F);
        Tr.at<double>(0,0) = 7.533745e-03; Tr.at<double>(0,1) = -9.999714e-01; Tr.at<double>(0,2) = -6.166020e-04; Tr.at<double>(0,3) = -4.069766e-03;
        Tr.at<double>(1,0) = 1.480249e-02; Tr.at<double>(1,1) = 7.280733e-04;  Tr.at<double>(1,2) = -9.998902e-01; Tr.at<double>(1,3) = -7.631618e-02;
        Tr.at<double>(2,0) = 9.998621e-01; Tr.at<double>(2,1) = 7.523790e-03;  Tr.at<double>(2,2) = 1.480755e-02;  Tr.at<double>(2,3) = -2.717806e-01;

        cv::Mat R_rect = cv::Mat::eye(4, 4, CV_64F);
        R_rect.at<double>(0,0) = 9.999239e-01; R_rect.at<double>(0,1) = 9.837760e-03;  R_rect.at<double>(0,2) = -7.445048e-03;
        R_rect.at<double>(1,0) = -9.869795e-03; R_rect.at<double>(1,1) = 9.999421e-01; R_rect.at<double>(1,2) = -4.278459e-03;
        R_rect.at<double>(2,0) = 7.402527e-03; R_rect.at<double>(2,1) = 4.351614e-03;  R_rect.at<double>(2,2) = 9.999631e-01;

        cv::Mat P_rect = cv::Mat::zeros(3, 4, CV_64F);
        P_rect.at<double>(0,0) = 7.215377e+02; P_rect.at<double>(0,2) = 6.095593e+02; P_rect.at<double>(0,3) = 4.485728e+01;
        P_rect.at<double>(1,1) = 7.215377e+02; P_rect.at<double>(1,2) = 1.728540e+02; P_rect.at<double>(1,3) = 2.163791e-01;
        P_rect.at<double>(2,2) = 1.000000e+00; P_rect.at<double>(2,3) = 2.745884e-03;

        projection_matrix_ = P_rect * R_rect * Tr;
    }

    cv::Point2f project_3d(double x, double y, double z) {
        cv::Mat pt_3d = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);
        cv::Mat pt_2d = projection_matrix_ * pt_3d;
        double w = pt_2d.at<double>(2, 0);
        if (w <= 0) return cv::Point2f(-1, -1);
        return cv::Point2f(pt_2d.at<double>(0, 0) / w, pt_2d.at<double>(1, 0) / w);
    }

    double calculate_iou(const cv::Rect2f& boxA, const cv::Rect2f& boxB) {
        cv::Rect2f intersection = boxA & boxB;
        double inter_area = intersection.area();
        double union_area = boxA.area() + boxB.area() - inter_area;
        return (union_area <= 0) ? 0.0 : (inter_area / union_area);
    }

    std_msgs::msg::ColorRGBA get_semantic_color(std::string label) {
        std::transform(label.begin(), label.end(), label.begin(), ::tolower);
        std_msgs::msg::ColorRGBA color;
        color.a = 0.7;

        if (label == "car" || label == "truck" || label = "bus" || label = "bicycle" || label = "motorcycle") {
            color.r = 1.0, color.g = 0.5, color.b = 0.0; // Orange
        } else if (label = "Person") {
            color.r = 1.0; color.g = 0.0; color.b = 0.0; // Red
        } else if (label == "traffic light" || label == "stop sign") {
            color.r = 1.0; color.g = 1.0; color.b = 0.0; // Yellow
        } else if (label == "unknown") {
            color.r = 0.0; color.g = 1.0; color.b = 0.0; // Green
        } else {
            color.r = 0.0; color.g = 1.0; color.b = 1.0; // Cyan
        }
        return color;
    }

    void yolo_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
    latest_yolo_detections_ = *msg;
    }

    void mot_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
        visualization_msgs::msg::MarkerArray identified_array;
        visualization_msgs::msg::MarkerArray unknown_array;

        // Extract YOLO
        std::vector<cv::Rect2f> yolo_boxes;
        std::vector<std::string> yolo_labels;
        for (const auto& det : latest_yolo_detections_.detections) {
            float x = det.bbox.center.position.x - det.bbox.size_x / 2.0;
            float y = det.bbox.center.position.y - det.bbox.size_y / 2.0;
            yolo_boxes.push_back(cv::Rect2f(x, y, det.bbox.size_x, det.bbox.size_y));
            yolo_labels.push_back(det.results[0].hypothesis.class_id);
        }

        std::set<int> current_live_ids;
        std::vector<visualization_msgs::msg::Marker> live_tracks;
        std::vector<cv::Rect2f> valid_mot_boxes;
        std::vector<int> valid_mot_indices;

        std_msgs::msg::Header ref_header;
        if (!msg->markers.empty()) ref_header = msg->markers[0].header;

        // Parse Live Tracks
        for (const auto& m : msg->markers) {
            if (m.action == visualization_msgs::msg::Marker::ADD && m.type == visualization_msgs::msg::Marker::CUBE) {
                int real_id = m.id / 2;
                current_live_ids.insert(real_id);
                live_tracks.push_back(m);

                auto pos = m.pose.position;
                auto dim = m.scale;
                std::vector<cv::Point2f> corners;

                // Cube Corners
                double dx_vals[] = {-dim.x/2, dim.x/2};
                double dy_vals[] = {-dim.y/2, dim.y/2};
                double dz_vals[] = {-dim.z/2, dim.z/2};

                for (double dx : dx_vals) {
                    for (double dy : dy_vals) {
                        for (double dz : dz_vals) {
                            cv::Point2f p = project_3d(pos.x + dx, pos.y + dy, pos.z + dz);
                            if (p.x != -1)  corners.push_back(p);
                        } 
                    }
                }

                if (corners.size() >= 4 ) {
                    valid_mot_boxes.push_back(cv::boundingRect(corners));
                    valid_mot_indices.push_back(live_tracks.size() - 1);
                }
            }
        }







    }








}   