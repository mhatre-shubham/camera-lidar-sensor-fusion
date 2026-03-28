#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class LidarCameraProjectionNode : public rclcpp::Node
{
public:
    using ImageMsg = sensor_msgs::msg::Image;
    using CloudMsg = sensor_msgs::msg::PointCloud2;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, CloudMsg>;

    LidarCameraProjectionNode() : Node("lidar_camera_projection_node")
    {
        setup_calibration();

        // Subscribers
        image_sub_.subscribe(this, "/kitti/image/color/left");
        cloud_sub_.subscribe(this, "/lidar/clustered_obstacles_pcd");

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), image_sub_, cloud_sub_);

        sync_->registerCallback(
            std::bind(&LidarCameraProjectionNode::fusion_callback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // Publisher
        image_pub_ = this->create_publisher<ImageMsg>("/fusion/lidar_camera_projection", 10);

        RCLCPP_INFO(this->get_logger(), " Lidar-Camera Fusion Node Startedd");
    }

private:
    // Subscribers
    message_filters::Subscriber<ImageMsg> image_sub_;
    message_filters::Subscriber<CloudMsg> cloud_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Publisher
    rclcpp::Publisher<ImageMsg>::SharedPtr image_pub_;

    // Projection matrix
    cv::Mat projection_matrix_;

    void setup_calibration()
    {
        cv::Mat Tr = cv::Mat::eye(4, 4, CV_64F);
        Tr.at<double>(0,0)=7.533745e-03; Tr.at<double>(0,1)=-9.999714e-01; Tr.at<double>(0,2)=-6.166020e-04; Tr.at<double>(0,3)=-4.069766e-03;
        Tr.at<double>(1,0)=1.480249e-02; Tr.at<double>(1,1)=7.280733e-04;  Tr.at<double>(1,2)=-9.998902e-01; Tr.at<double>(1,3)=-7.631618e-02;
        Tr.at<double>(2,0)=9.998621e-01; Tr.at<double>(2,1)=7.523790e-03;  Tr.at<double>(2,2)=1.480755e-02;  Tr.at<double>(2,3)=-2.717806e-01;

        cv::Mat R_rect = cv::Mat::eye(4, 4, CV_64F);
        R_rect.at<double>(0,0)=9.999239e-01; R_rect.at<double>(0,1)=9.837760e-03;  R_rect.at<double>(0,2)=-7.445048e-03;
        R_rect.at<double>(1,0)=-9.869795e-03;R_rect.at<double>(1,1)=9.999421e-01; R_rect.at<double>(1,2)=-4.278459e-03;
        R_rect.at<double>(2,0)=7.402527e-03; R_rect.at<double>(2,1)=4.351614e-03;  R_rect.at<double>(2,2)=9.999631e-01;

        cv::Mat P_rect = cv::Mat::zeros(3, 4, CV_64F);
        P_rect.at<double>(0,0)=721.5377; P_rect.at<double>(0,2)=609.5593; P_rect.at<double>(0,3)=44.85728;
        P_rect.at<double>(1,1)=721.5377; P_rect.at<double>(1,2)=172.8540; P_rect.at<double>(1,3)=0.2163791;
        P_rect.at<double>(2,2)=1.0;

        projection_matrix_ = P_rect * R_rect * Tr;
    }

    cv::Point2f project_point(double x, double y, double z)
    {
        cv::Mat pt = (cv::Mat_<double>(4,1) << x, y, z, 1.0);
        cv::Mat img_pt = projection_matrix_ * pt;

        double w = img_pt.at<double>(2,0);
        if (w <= 0) return {-1, -1};

        return cv::Point2f(
            static_cast<float>(img_pt.at<double>(0,0) / w),
            static_cast<float>(img_pt.at<double>(1,0) / w)
        );
    }

    void fusion_callback(const ImageMsg::ConstSharedPtr img_msg,
                         const CloudMsg::ConstSharedPtr cloud_msg)
    {
        // Convert ROS → OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        cv::Mat img = cv_ptr->image;

        // Iterate LiDAR points
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");

        int count = 0;

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            //if (count++ % 5 != 0) continue; // downsample

            double x = *iter_x;
            double y = *iter_y;
            double z = *iter_z;

            if (x < 0.1) continue;

            cv::Point2f pt = project_point(x, y, z);

            if (pt.x < 0 || pt.y < 0 || pt.x >= img.cols || pt.y >= img.rows)
                continue;

            double dist = std::sqrt(x*x + y*y + z*z);
            int intensity = std::min(255, int(255.0 * dist / 50.0));

            cv::circle(img, pt, 2, cv::Scalar(0, intensity, 255 - intensity), -1);
        }

        // Publish result
        image_pub_->publish(
            *cv_bridge::CvImage(img_msg->header, "bgr8", img).toImageMsg()
        );
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarCameraProjectionNode>());
    rclcpp::shutdown();
    return 0;
}