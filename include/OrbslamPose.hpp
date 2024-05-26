/*
 * Software License Agreement (New BSD License)
 *
 * Copyright (c) 2013, Keith Leung, Felipe Inostroza
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Advanced Mining Technology Center (AMTC), the
 *       Universidad de Chile, nor the names of its contributors may be
 *       used to endorse or promote products derived from this software without
 *       specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AMTC, UNIVERSIDAD DE CHILE, OR THE COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 * THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "g2o/types/sba/types_six_dof_expmap.h" // se3 poses
#include <opencv2/core/core.hpp>
#include <vector>
#include <external/ORBextractor.h>

namespace rfs
{
 class OrbslamMapPoint;

    class OrbslamPose
    {

    public:
        typedef g2o::VertexSBAPointXYZ PointType;
        typedef g2o::VertexSE3Expmap PoseType;
        typedef g2o::EdgeProjectXYZ2UV MonocularMeasurementEdge;
        typedef g2o::EdgeStereoSE3ProjectXYZ StereoMeasurementEdge;

        PoseType *pPose;

        // Scale
         int mnScaleLevels;
         float mfScaleFactor;
         float mfLogScaleFactor;
         std::vector<float> mvScaleFactors;
         std::vector<float> mvLevelSigma2;
         std::vector<float> mvInvLevelSigma2;

        // image bounds

        double mnMinX;
        double mnMaxX;
        double mnMinY;
        double mnMaxY;

        //ORB

        // keypoints detected
        std::vector<cv::KeyPoint> keypoints_left, keypoints_right;

        cv::Mat descriptors_left, descriptors_right;

        std::vector<float> uRight;
        std::vector<float> depth;
        std::vector<Eigen::Vector3d> point_camera_frame;

        std::vector<cv::DMatch> matches_left_to_right;




        std::vector<int> fov_; /**< indices of landmarks in field of view at time k */

        std::vector<StereoMeasurementEdge *> Z_; /**< Measurement edges stored, in order to set data association and add to graph later */

        /**
         * @brief estimate if map point should be measured, as a stereo pair
         *
         * @param pMP  point to be measured (o not)
         * @param viewingCosLimit  view angle requirement
         * @return true  point is in field of view
         * @return false point should not be measured
         */
        bool isInFrustum(OrbslamMapPoint *pMP, float viewingCosLimit, g2o::CameraParameters *cam_params);

        /**
         *
         */
        OrbslamPose( );



        /**
         *  clone this pose , generating a new instance of each dynamically alocated object inside
         *
         * @return
         */
        OrbslamPose clone();

        /**
         *
         * @param other
         */
        OrbslamPose( const OrbslamPose &other);

    };

}
