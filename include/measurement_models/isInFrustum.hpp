#pragma once 


#include <Pose.hpp>
#include <Landmark.hpp>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>



namespace rfs  {
struct CameraBounds{
  float mnMinX;
  float mnMaxX;
  float mnMinY;
  float mnMaxY;
};
struct Camera{
  gtsam::StereoCamera camera;
  CameraBounds bounds;
  float viewingCosLimit;
};


gtsam::Point3 to_gtsam(const rfs::Landmark3d &landmark);

gtsam::Pose3 to_gtsam(const rfs::Pose6d &pose);

bool isInFrustum(const Landmark3d &landmark, const Pose6d &pose, const Camera &camera, double * predictedScale=NULL);
bool isInFrustum(const gtsam::Point3 &landmark, const gtsam::Pose3 &pose, const Camera &camera, double * predictedScale=NULL);
}
