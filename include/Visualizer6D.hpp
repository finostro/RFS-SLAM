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

#ifndef VISUALIZER6D_HPP
#define VISUALIZER6D_HPP

#include "RBPHDFilter.hpp"
#include "RBLMBFilter.hpp"
#include "ProcessModel_Odometry6D.hpp"
#include "measurement_models/MeasurementModel_3D_stereo_orb.hpp"
#include "measurement_models/MeasurementModel_6D.hpp"
#include <thread>
#include <mutex>

#include <vtkCommand.h>
#include <vtkSphereSource.h>
#include <vtkAxesActor.h>
#include <vtkTransform.h>
#include <vtkPolyData.h>
#include <vtkPolyLine.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkOpenGLRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkGlyph3D.h>
#include <vtkPointData.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNamedColors.h>

// #include "VectorGLMBComponent6D.hpp"

namespace rfs{
/**
 * \class Visualizer6D
 * \brief A visualization class to visualize 6D slam
 * \author Felipe Inostroza
 */

class Visualizer6D{

public:

	Visualizer6D();

	void setup(const std::vector<MeasurementModel_6D::TLandmark> &groundtruth_landmark_,
			const std::vector<MotionModel_Odometry6d::TState> &groundtruth_pose_,
			const std::vector<MotionModel_Odometry6d::TState> &deadreckoning_pose_,
				RBPHDFilter<MotionModel_Odometry6d, StaticProcessModel<Landmark3d>,
                        MeasurementModel_3D_stereo_orb,
                        KalmanFilter<StaticProcessModel<Landmark3d>, MeasurementModel_3D_stereo_orb> > *pFilter_= NULL);
	void setup();


        void update(RBPHDFilter<MotionModel_Odometry6d, StaticProcessModel<Landmark3d>,
                        MeasurementModel_6D,
                        KalmanFilter<StaticProcessModel<Landmark3d>, MeasurementModel_6D> > *pFilter_);
        void update(RBPHDFilter<MotionModel_Odometry6d, StaticProcessModel<Landmark3d>,
                        MeasurementModel_3D_stereo_orb,
                        KalmanFilter<StaticProcessModel<Landmark3d>, MeasurementModel_3D_stereo_orb> > *pFilter_);
        void update(RBLMBFilter<MotionModel_Odometry6d, StaticProcessModel<Landmark3d>,
                        MeasurementModel_6D,
                        KalmanFilter<StaticProcessModel<Landmark3d>, MeasurementModel_6D> > *pFilter_);
        // void update(const VectorGLMBComponent6D &c);

        void initFrustum( MeasurementModel_3D_stereo_orb &measurementModel);
	void updateFrustum(const Pose6d &x_i, MeasurementModel_3D_stereo_orb &measurementModel);

	void start();

	void run();

	void pause();


	vtkSmartPointer<vtkRenderer> renderer_ ;
	vtkSmartPointer<vtkRenderWindow> renderWindow_ ;
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor_;
	vtkSmartPointer<vtkSphereSource> sphereSource_;
	vtkSmartPointer<vtkPoints> mapPoints_, gtmapPoints_, particlePoints_, gtTrajectoryPoints_, estTrajectoryPoints_, drTrajectoryPoints_, measurementPoints_, frustumPoints_;
	vtkSmartPointer<vtkCellArray> gtTrajectoryCells_, estTrajectoryCells_, drTrajectoryCells_, measurementCells_, frustumCells_;
	vtkSmartPointer<vtkUnsignedCharArray> mapColors_, gtmapColors_, particleColors_;
	vtkSmartPointer<vtkGlyph3D> mapGlyph3D_, gtmapGlyph3D_, particleGlyph3D_;
	vtkSmartPointer<vtkPolyData> mapPolydata_, gtmapPolydata_, particlePolydata_, gtTrajectoryPolydata_, estTrajectoryPolydata_, drTrajectoryPolydata_, measurementPolydata_, frustumPolydata_;
	vtkSmartPointer<vtkActor> mapActor_, gtmapActor_, particleActor_, gtTrajectoryActor_, estTrajectoryActor_, drTrajectoryActor_, measurementActor_, frustumActor_;
	vtkSmartPointer<vtkAxesActor>  axesActor_;
	vtkSmartPointer<vtkPolyDataMapper> mapMapper_, gtmapMapper_, particleMapper_, gtTrajectoryMapper_, estTrajectoryMapper_, drTrajectoryMapper_, measurementMapper_, frustumMapper_;


  	vtkSmartPointer<vtkTransform> origin_transform;


  	vtkSmartPointer<vtkNamedColors> colors_;

	std::vector<MotionModel_Odometry6d::TState> const *groundtruth_pose_;
	double sphere_radius_ = 2.1;
	bool stopped=false;
	bool init_trajectory=false;
	std::thread *display_thread_;
	std::mutex *display_mutex_;
	int i_trajectory = 0; // number of steps to show

	std::vector<gtsam::Point3> frustum_points_in_camera_frame_;
};






}



#endif


