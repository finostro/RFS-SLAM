
#include "measurement_models/isInFrustum.hpp"

namespace rfs  {

gtsam::Point3 to_gtsam(const Landmark3d &landmark){
	return landmark.get();
}

gtsam::Pose3 to_gtsam(const Pose3d &pose){
    return gtsam::Pose3(gtsam::Rot3(Eigen::Quaterniond(Eigen::AngleAxis(pose.getRot().norm(), pose.getRot().normalized()))), pose.getPos());
}

gtsam::Pose3 to_gtsam(const Pose6d &pose){
    return gtsam::Pose3(gtsam::Rot3(Eigen::Quaterniond(pose.getRot())), pose.getPos());
}

bool isInFrustum(const Landmark3d &landmark, const Pose6d &pose, const Camera &camera, double * predictedScale){
        
    auto lm = to_gtsam(landmark);
    auto pose_gtsam = to_gtsam(pose);

    return isInFrustum( lm , pose_gtsam, camera   , predictedScale);
}

bool isInFrustum(const gtsam::Point3 &landmark, const gtsam::Pose3 &pose, const Camera &camera, double * predictedScale){
    gtsam::Point3  point_in_camera_frame = pose.transformTo(landmark);
            // std::cout <<"point_in_camera_frame   " <<point_in_camera_frame << "\n"
            //     << "z  " << point_in_camera_frame(2)  << "\n";
            // check depth
            if (point_in_camera_frame(2) <= 0)
            {
                return false;
            }
            
            gtsam::StereoPoint2  stereoPoint = camera.camera.project( point_in_camera_frame );
           

            // check image bounds
            
            if (stereoPoint.uL() < camera.bounds.mnMinX || stereoPoint.uL() > camera.bounds.mnMaxX)
            {
                return false;
            }

            if (stereoPoint.v() < camera.bounds.mnMinY || stereoPoint.v() > camera.bounds.mnMaxY)
            {
                return false;
            }

            if (stereoPoint.uR() < camera.bounds.mnMinX || stereoPoint.uR() > camera.bounds.mnMaxX)
            {
                return false;
            }

    // TODO: implement the following checks
            // // Check distance is in the scale invariance region of the MapPoint
            // const float maxDistance = pMP->mfMaxDistance;
            // const float minDistance = pMP->mfMinDistance;
            // const float dist = point_in_camera_frame.norm();
            //
            // if (dist < minDistance || dist > maxDistance)
            //     return false;
            //
            // // Check viewing angle
            // Eigen::Vector3d Pn = pMP->normalVector;
            // Eigen::Vector3d viewingVector =pMP->position-pose.translation();
            // 
            // const float viewCos = viewingVector.dot(Pn) / dist;
            //
            // if (viewCos < viewingCosLimit)
            //     return false;
            //
            // // Predict scale in the image
            // const int nPredictedLevel = pMP->predictScale(dist, this);
            // if (predictedScale!=NULL){
            //     *predictedScale = nPredictedLevel;
            // }
            return true;
}
}
