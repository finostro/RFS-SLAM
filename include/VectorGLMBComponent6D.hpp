
/*
 * Software License Agreement (New BSD License)
 *
 * Copyright (c) 2022, Felipe Inostroza
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

#include <vector>
#include <boost/bimap.hpp>
#include <boost/container/allocator.hpp>


#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/types/sba/types_six_dof_expmap.h" // se3 poses



#include "OrbslamMapPoint.hpp"
#include "OrbslamPose.hpp"


namespace rfs
{


/**
 *  The weight and index of a landmarks from which a measurement can come from
 */
struct AssociationProbability {
	int i; /**< index of a landmark */
	double l; /**< log probability of association*/
};
struct AssociationProbabilities {
	std::vector<int> i; /**< index of a landmark */
	std::vector<double> l; /**< log probability of association*/
};




/**
 * Struct to store a single component of a VGLMB , with its own g2o optimizer
 */
struct VectorGLMBComponent6D {
	typedef g2o::VertexSBAPointXYZ PointType;
	typedef g2o::VertexSE3Expmap PoseType;
	typedef g2o::EdgeProjectXYZ2UV MonocularMeasurementEdge;
	typedef g2o::EdgeProjectXYZ2UVU StereoMeasurementEdge;
	typedef g2o::EdgeSE3Expmap OdometryEdge;


	typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> > SlamBlockSolver;
	typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	g2o::SparseOptimizer *optimizer_;

	g2o::OptimizationAlgorithmLevenberg *solverLevenberg_;
	SlamLinearSolver *linearSolver_;
	SlamBlockSolver *blockSolver_;

	std::vector<boost::bimap<int, int, boost::container::allocator<int>>>
			DA_bimap_, prevDA_bimap_; /**< Bimap containing data association hypothesis at time k  */

	std::vector<std::vector<StereoMeasurementEdge*> > Z_; /**< Measurement edges stored, in order to set data association and add to graph later */
	std::vector<std::vector<AssociationProbabilities> > DAProbs_; /**< DAProbs_ [k][nz] are is the association probabilities of measurement
	 nz at time k, used for switching using gibbs sampling*/

	std::vector<OrbslamPose> poses_;
	std::vector<OrbslamMapPoint> landmarks_;
	std::vector<double> landmarksResetProb_, landmarksInitProb_;

	double logweight_ = -std::numeric_limits<double>::infinity(),
			prevLogWeight_ = -std::numeric_limits<double>::infinity();
	int numPoses_, numPoints_;
	bool reverted_ = false;
	std::vector<std::pair<int, int>> tomerge_; /**< proportional to probability of merge heuristic*/

    int maxpose_; /**< maximum pose to optimize to */
};

}
