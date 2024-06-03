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

#include "OrbslamPose.hpp"
#include "OrbslamMapPoint.hpp"

namespace rfs
{


        bool OrbslamPose::isInFrustum(OrbslamMapPoint *pMP, float viewingCosLimit, g2o::CameraParameters *cam_params)
        {

        	Eigen::Vector3d  point_in_camera_frame = pPose->estimate().map(pMP->pPoint->estimate());

            // check depth
            if (point_in_camera_frame(2) <= 0)
            {
                return false;
            }
            Eigen::Vector3d uvu = cam_params->stereocam_uvu_map(point_in_camera_frame);

            // check image bounds
            if (uvu(0) < mnMinX || uvu(0) > mnMaxX)
            {
                return false;
            }

            if (uvu(1) < mnMinY || uvu(1) > mnMaxY)
            {
                return false;
            }

            if (uvu(2) < mnMinX || uvu(2) > mnMaxX)
            {
                return false;
            }

            // Check distance is in the scale invariance region of the MapPoint
            const float maxDistance = pMP->mfMinDistance;
            const float minDistance = pMP->mfMinDistance;
            const float dist = point_in_camera_frame.norm();

            if (dist < minDistance || dist > maxDistance)
                return false;

            // Check viewing angle
            Eigen::Vector3d Pn = pMP->normalVector;
            const float viewCos = point_in_camera_frame.dot(Pn) / dist;

            if (viewCos < viewingCosLimit)
                return false;

            // Predict scale in the image
            const int nPredictedLevel = pMP->predictScale(dist, this);
            return true;
        }


}
