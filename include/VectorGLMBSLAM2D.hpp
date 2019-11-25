/*
 * Software License Agreement (New BSD License)
 *
 * Copyright (c) 2014, Keith Leung, Felipe Inostroza
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
#ifndef VECTORGLMBSLAM_HPP
#define VECTORGLMBSLAM_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Timer.hpp"
#include <Eigen/Core>
#include <math.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include "ceres/ceres.h"
#include <unordered_map>
#include <math.h>
#include "GaussianGenerators.hpp"
#include "AssociationSampler.hpp"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/core/robust_kernel_impl.h"

#include "g2o/types/slam2d/vertex_point_xy.h"
#include "g2o/types/slam2d/vertex_se2.h"
#include "g2o/types/slam2d/edge_pointxy.h"
#include "g2o/types/slam2d/edge_se2_pointxy.h"
#include "g2o/types/slam2d/edge_se2.h"
#include <boost/random/uniform_real.hpp>
#include <boost/bimap.hpp>
#include <yaml-cpp/yaml.h>

#include "misc/EigenYamlSerialization.hpp"


#ifdef _PERFTOOLS_CPU
#include <gperftools/profiler.h>
#endif
#ifdef _PERFTOOLS_HEAP
#include <gperftools/heap-profiler.h>
#endif

namespace rfs {

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
struct VectorGLMBComponent2D {

	typedef g2o::VertexPointXY PointType;
	typedef g2o::VertexSE2 PoseType;
	typedef g2o::EdgeSE2PointXY MeasurementEdge;
	typedef g2o::EdgePointXY PointAnchorEdge;

	typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> > SlamBlockSolver;
	typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	g2o::SparseOptimizer *optimizer_;
	g2o::OptimizationAlgorithmLevenberg *solverLevenberg_;
	SlamLinearSolver *linearSolver_;

	std::vector<boost::bimap<int, int>> DA_bimap_; /**< Bimap containing data association hypothesis at time k  */

	std::vector<std::vector<MeasurementEdge*> > Z_; /**< Measurement edges stored, in order to set data association and add to graph later */
	std::vector<std::vector<AssociationProbabilities> > DAProbs_; /**< DAProbs_ [k][nz] are is the association probabilities of measurement
	 nz at time k, used for switching using gibbs sampling*/
	std::vector<std::vector<int> > fov_; /**< indices of landmarks in field of view at time k */

	std::vector<PoseType*> poses_;
	std::vector<PointType*> landmarks_;

	double logweight_,prevLogWeight_;
	int numPoses_, numPoints_;
};

/**
 *  \class VectorGLMBSLAM2D
 *  \brief Random Finite Set  optimization using ceres solver for  feature based SLAM
 *
 *
 *  \author  Felipe Inostroza
 */
class VectorGLMBSLAM2D {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef g2o::VertexPointXY PointType;
	typedef g2o::VertexSE2 PoseType;
	typedef g2o::EdgeSE2 OdometryEdge;
	typedef g2o::EdgeSE2PointXY MeasurementEdge;
	typedef g2o::EdgePointXY PointAnchorEdge;
	typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> > SlamBlockSolver;
	typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	/**
	 * \brief Configurations for this RFSBatchPSO optimizer
	 */
	struct Config {

		/** The threshold used to determine if a possible meaurement-landmark
		 *  pairing is significant to worth considering
		 */
		double MeasurementLikelihoodThreshold_;


		double logKappa_; /**< intensity of false alarm poisson model*/

		double PE_; /**<  landmark existence probability*/

		double PD_; /**<  landmark detection probability*/

		double maxRange_; /**< maximum sensor range */

		int numComponents_;

		std::vector<double> xlim_, ylim_;

		int numLandmarks_; /**< number of landmarks per dimension total landmarks will be numlandmarks^2 */

        int numGibbs_; /**< number of gibbs samples of the data association */
        int numLevenbergIterations_; /**< number of gibbs samples of the data association */

		int lmExistenceProb_;
		Eigen::Matrix2d anchorInfo_; /** information for anchor edges, should be low*/

	} config;

	/**
	 * Constructor
	 */
	VectorGLMBSLAM2D();

	/** Destructor */
	~VectorGLMBSLAM2D();

	/**
	 *  Load a g2o style file , store groundtruth data association.
	 * @param filename g2o file name
	 */
	void
	load(std::string filename);

	/**
	 *  Load a yaml style config file
	 * @param filename filename of the yaml config file
	 */
	void
	loadConfig(std::string filename);

	/**
	 * initialize the components , set the initial data associations to all false alarms
	 */
	void initComponents();

	/**
	 * run the optimization over the possible data associations.
	 * @param numsteps number of iterations in algorithm.
	 */
	void run(int numsteps);
	/**
	 * Do n iterations
	 * @param ni number of iterations of the optimizer
	 */
	void optimize(int ni);

	/**
	 * Use the data association stored in DA_ to create the graph.
	 * @param c the GLMB component
	 */
	void constructGraph(VectorGLMBComponent2D &c);
	/**
	 *
	 * Initialize a VGLMB component , setting the data association to all false alarms
	 * @param c the GLMB component
	 */
	void init(VectorGLMBComponent2D &c);

	/**
	 * Calculate the probability of each measurement being associated with a specific landmark
	 * @param c the GLMB component
	 */
	void updateDAProbs(VectorGLMBComponent2D &c);

	/**
	 * Calculate the FoV at each time
	 * @param c the GLMB component
	 */
	void updateFoV(VectorGLMBComponent2D &c);

	/**
	 * Use the probabilities calculated using updateDAProbs to sample a new data association through gibbs sampling
	 * @param c the GLMB component
	 */
	void sampleDA(VectorGLMBComponent2D &c);

	/**
	 * Use the data association hipothesis and the optimized state to calculate the component weight.
	 * @param c the GLMB component
	 */
	void calculateWeight(VectorGLMBComponent2D &c);
	/**
	 * Use the new sampled data association to update the g2o graph
	 * @param c the GLMB component
	 */
	void updateGraph(VectorGLMBComponent2D &c);

	/**
	 * Calculate the range between a pose and a landmark, to calculate the probability of detection.
	 * @param pose A 2D pose
	 * @param lm A 2D landmark
	 * @return the distance between pose and landmark
	 */
	static double distance(PoseType *pose, PointType *lm);

	int nThreads_; /**< Number of threads  */

	VectorGLMBComponent2D gt_graph;

	std::vector<VectorGLMBComponent2D> components_; /**< VGLMB components */

};

//////////////////////////////// Implementation ////////////////////////

VectorGLMBSLAM2D::VectorGLMBSLAM2D() {
	nThreads_ = 1;

#ifdef _OPENMP
      nThreads_ = omp_get_max_threads();
#endif

}

VectorGLMBSLAM2D::~VectorGLMBSLAM2D() {

}

void VectorGLMBSLAM2D::load(std::string filename) {
	std::ifstream ifs (filename, std::ifstream::in);

	gt_graph.optimizer_->load(ifs);
	ifs.close();
}

void VectorGLMBSLAM2D::loadConfig(std::string filename) {

	YAML::Node node = YAML::LoadFile(filename);

	config.MeasurementLikelihoodThreshold_ = node["MeasurementLikelihoodThreshold"].as<double>();
	config.lmExistenceProb_ = node["lmExistenceProb"].as<double>();
	config.logKappa_ = node["logKappa"].as<double>();
	config.PE_ = node["PE"].as<double>();
	config.PD_ = node["PD"].as<double>();
	config.maxRange_ = node["maxRange"].as<double>();
	config.numComponents_ = node["numComponents"].as<int>();
	config.numLandmarks_ = node["numLandmarks"].as<int>();
	config.numGibbs_ = node["numGibbs"].as<int>();
	config.xlim_.push_back(node["xlim"][0].as<double>()) ;
	config.xlim_.push_back(node["xlim"][1].as<double>()) ;
	config.ylim_.push_back(node["ylim"][0].as<double>()) ;
	config.ylim_.push_back(node["ylim"][1].as<double>()) ;



	if(!YAML::convert<Eigen::Matrix2d>::decode(node["anchorInfo"], config.anchorInfo_)){
		std::cerr << "could not load anchor info matrix \n";
		exit(1);
	}

}

 double VectorGLMBSLAM2D::distance(PoseType *pose, PointType *lm) {

	Eigen::Vector3d posemean;
	pose->getEstimateData(posemean.data());
	Eigen::Vector2d pointmean;
	lm->getEstimateData(pointmean.data());
	return sqrt((pointmean - posemean.head(2)).squaredNorm());

}

inline void VectorGLMBSLAM2D::initComponents() {
	components_.resize(config.numComponents_);

	for (auto &c : components_) {
		init(c);
		constructGraph(c);
	}

}
inline void VectorGLMBSLAM2D::run(int numSteps) {
    initComponents();
    for( int i =0; i < numSteps; i++){
        optimize(config.numLevenbergIterations_);
    }
}
inline void VectorGLMBSLAM2D::optimize(int ni) {
	for (auto &c : components_) {
		updateFoV(c);
		updateDAProbs(c);
		for(int i=0; i< config.numGibbs_ ; i++){
			sampleDA(c);
		}
		updateGraph(c);
		c.poses_[0]->fixed();
		c.optimizer_->initializeOptimization(c.optimizer_->edges());
		c.optimizer_->optimize(ni);
		calculateWeight(c);
		std::cout << "weight: " << c.logweight_ << "\n";

	}
}
inline void VectorGLMBSLAM2D::calculateWeight(VectorGLMBComponent2D &c) {
	double logw = 0;


	for (int k = 0; k < c.poses_.size(); k++) {
		for (int nz = 0; nz < c.Z_[k].size(); nz++) {
			auto it = c.DA_bimap_[k].left.find(nz);
			int selectedDA = -2;
			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}
			if(selectedDA <0){
				logw+=config.logKappa_;
			}else{
				logw+= -0.5*(c.Z_[k][nz]->dimension()*std::log(2*M_PI) - std::log(c.Z_[k][nz]->information().determinant()) );
			}
		}


		for(int lm=0; lm< c.fov_[k].size() ; lm++){


			if(c.DA_bimap_[k].right.count(c.fov_[k][lm]) >0){
				logw+=std::log(config.PD_);
			}else{
				bool exists = c.optimizer_->vertex(c.fov_[k][lm])->edges().size()>1;
				if (exists){
				logw+=std::log(1-config.PD_);
				}
			}
		}
	}

	for(int lm=0; lm < c.landmarks_.size() ; lm++){
		bool exists = c.landmarks_[lm]->edges().size()>1;
		if (exists){
			logw+=std::log(config.PE_);
		}else{
			logw+=std::log(1-config.PE_);
		}
	}
	logw+= -0.5*(c.optimizer_->chi2() + c.linearSolver_->_determinant);
	c.prevLogWeight_ = c.logweight_;
	c.logweight_ = logw;
}



inline void VectorGLMBSLAM2D::updateGraph(VectorGLMBComponent2D &c) {
	for (int k = 0; k < c.poses_.size(); k++) {
		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
			auto it = c.DA_bimap_[k].left.find(nz);
			int selectedDA = -2;
			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}
			int previd = c.Z_[k][nz]->vertex(1)? c.Z_[k][nz]->vertex(1)->id():-2; /**< previous data association */
			if(previd == selectedDA){
				continue;
			}
			if(selectedDA>=0){

				c.Z_[k][nz]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(selectedDA)->second));

				// if edge was already in graph, remove it before inserting it again
				 if(previd>=0){
					 c.optimizer_->removeEdge(c.Z_[k][nz]);
				 }
				 c.optimizer_->addEdge(c.Z_[k][nz]);
			}else{
				c.Z_[k][nz]->setVertex(1,NULL);
				c.optimizer_->removeEdge(c.Z_[k][nz]);

			}

		}
	}
}
inline void VectorGLMBSLAM2D::sampleDA(VectorGLMBComponent2D &c) {
	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif

	AssociationProbabilities probs;
	for (int k = 0; k < c.DAProbs_.size(); k++) {

		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
			probs.i.clear();
			probs.l.clear();
			double maxprob = -std::numeric_limits<double>::infinity();
			auto it = c.DA_bimap_[k].left.find(nz);
			int selectedDA = -2;
			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}
			for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++) {
				if (c.DAProbs_[k][nz].i[a] == -2 || c.DAProbs_[k][nz].i[a] == selectedDA) { // is false alarm probability or is the already selected association
					probs.i.push_back(c.DAProbs_[k][nz].i[a]);
					probs.l.push_back(c.DAProbs_[k][nz].l[a]);
					if (c.DAProbs_[k][nz].l[a] > maxprob)
						maxprob = c.DAProbs_[k][nz].l[a];
				} else {
					if (c.DA_bimap_[k].right.count(c.DAProbs_[k][nz].i[a]) == 0) {  // landmark is not already associated to another measurement
						probs.i.push_back(c.DAProbs_[k][nz].i[a]);
						probs.l.push_back(c.DAProbs_[k][nz].l[a]);
						if (c.DAProbs_[k][nz].l[a] > maxprob)
							maxprob = c.DAProbs_[k][nz].l[a];
					}
				}
			}

			for (auto &p : probs.l) {
				p = std::exp(p - maxprob);
			}
			size_t sample = GibbsSampler::sample(randomGenerators_[threadnum], probs.l);

			if (probs.i[sample] != selectedDA) { // if selected association, change bimap

				if (probs.i[sample] >= 0) {
					if (selectedDA < 0) {
						c.DA_bimap_[k].insert( { nz, probs.i[sample] });
					} else {
						c.DA_bimap_[k].left.replace_data(it, probs.i[sample]);
					}
				} else { // if a change has to be made and new DA is false alarm, we need to remove the association
					c.DA_bimap_[k].left.erase(it);

				}

			}

		}
	}

}

inline void VectorGLMBSLAM2D::updateFoV(VectorGLMBComponent2D &c) {
	for (int k = 0; k < c.fov_.size(); k++) {
		for (auto lm : c.landmarks_) {
			if (distance(c.poses_[k], lm) <= config.maxRange_) {
				c.fov_[k].push_back(lm->id());
			}
		}
	}
}

inline void VectorGLMBSLAM2D::updateDAProbs(VectorGLMBComponent2D &c) {

	for (int k = 0; k < c.DAProbs_.size(); k++) {

		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {

			// setting the topology of DAProbs to include all measurements in current FoV
			c.DAProbs_[k][nz].i = c.fov_[k];
			c.DAProbs_[k][nz].l.resize(c.DAProbs_[k][nz].i.size());

			auto it = c.DA_bimap_[k].left.find(nz);
			int selectedDA = -2;
			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}

			for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++) {
				if (c.DAProbs_[k][nz].i[a] == -2) { // set measurement to false alarm
					c.DAProbs_[k][nz].l[a] = config.logKappa_
							+ 0.5 * (c.Z_[k][nz]->dimension() * std::log(2 * M_PI) - std::log(c.Z_[k][nz]->information().determinant()));
				} else {
					bool isNew; /**< does selecting this landmark imply creating it*/
					int numMeasurements = c.optimizer_->vertex(c.DAProbs_[k][nz].i[a])->edges().size()-1;
					isNew = numMeasurements == 0 || (numMeasurements == 1 && selectedDA == c.DAProbs_[k][nz].i[a]);
					c.DAProbs_[k][nz].l[a] = 0;
					if (isNew)
						c.DAProbs_[k][nz].l[a] += std::log(config.PE_)-std::log(1-config.PE_);

					c.DAProbs_[k][nz].l[a] +=  std::log(config.PD_)-std::log(1-config.PD_);
					c.Z_[k][nz]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(c.DAProbs_[k][nz].i[a])->second));
					c.Z_[k][nz]->computeError();

					c.DAProbs_[k][nz].l[a] += -0.5 * c.Z_[k][nz]->chi2();
				}
			};
			if(selectedDA>=0){
				c.Z_[k][nz]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(selectedDA)->second));
			}

		}

	}

}

inline void VectorGLMBSLAM2D::constructGraph(VectorGLMBComponent2D &c) {

	c.numPoses_ = 0;
	c.numPoints_ = 0;
	//Copy Vertices from optimizer with data association
	for (auto pair : gt_graph.optimizer_->vertices()) {
	    g2o::HyperGraph::Vertex *v =pair.second;
		PoseType *pose = dynamic_cast<PoseType*>(v);
		if (pose != NULL) {
			PoseType *poseCopy = new PoseType();
			double poseData[3];
			pose->getEstimateData(poseData);
			poseCopy->setEstimateData(poseData);
			poseCopy->setId(pose->id());
			c.optimizer_->addVertex(poseCopy);
			c.poses_.push_back(poseCopy);
			c.numPoses_++;
		}
		/*
		 PointType* point = dynamic_cast<PoseType>(v);
		 if (point != NULL) {
		 PointType*  pointCopy= new PointType();
		 double pointData[2];
		 point->getEstimateData(pointData);
		 pointCopy->setEstimateData(pointData);
		 pointCopy->setId(point->id());
		 c.optimizer_->addVertex(pointCopy);
		 c.landmarks_.push_back(pointCopy);
		 c.numPoints_++;
		 }
		 */
	}

	int lmid = 1;
	for (double x = config.xlim_[0]; x <= config.xlim_[1]; x += (config.xlim_[1] - config.xlim_[2]) / config.numLandmarks_) {
		for (double y = config.ylim_[0]; y <= config.ylim_[1]; y += (config.ylim_[1] - config.ylim_[2]) / config.numLandmarks_) {
			PointType *lm = new PointType();
			PointAnchorEdge *anchor = new PointAnchorEdge();
			Eigen::Vector2d xy(x, y);
			lm->setEstimateData(xy.data());
			lm->setId(lmid++);
			c.optimizer_->addVertex(lm);
			anchor->setVertex(0, lm);
			anchor->setMeasurement(xy);
			anchor->setInformation(config.anchorInfo_);
			c.optimizer_->addEdge(anchor);
		}

	}

	//Copy odometry measurements, Copy and save landmark measurements

	c.DA_bimap_.resize(c.numPoses_);

	c.Z_.resize(c.numPoses_);
	c.DAProbs_.resize(c.numPoses_);

	for (g2o::HyperGraph::Edge *e : gt_graph.optimizer_->edges()) {
		OdometryEdge *odo = dynamic_cast<OdometryEdge*>(e);
		if (odo != NULL) {
			OdometryEdge *odocopy = new OdometryEdge();
			int firstvertex = odo->vertex(0)->id();
			odocopy->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(firstvertex)->second));
			int secondvertex = odo->vertex(1)->id();
			odocopy->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(secondvertex)->second));
			double measurementData[3];
			odo->getMeasurementData(measurementData);
			odocopy->setMeasurementData(measurementData);
			odocopy->setInformation(odo->information());
			odocopy->setParameterId(0, 0);
			c.optimizer_->addEdge(odocopy);
		}

		MeasurementEdge *z = dynamic_cast<MeasurementEdge*>(e);
		if (z != NULL) {
			MeasurementEdge *zcopy = new MeasurementEdge();
			int firstvertex = z->vertex(0)->id();
			zcopy->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(firstvertex)->second));
			int secondvertex = z->vertex(1)->id();
			// zcopy->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(secondvertex)->second));
			double measurementData[2];
			z->getMeasurementData(measurementData);
			zcopy->setMeasurementData(measurementData);
			zcopy->setInformation(z->information());
			zcopy->setParameterId(0, 0);
			c.Z_[firstvertex - c.poses_[0]->id()].push_back(zcopy);
		}

	}

}

inline void VectorGLMBSLAM2D::init(VectorGLMBComponent2D &c) {
	auto linearSolver = g2o::make_unique<SlamLinearSolver>();
	linearSolver->setBlockOrdering(false);
	c.linearSolver_ = linearSolver.get();
	c.solverLevenberg_ = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<SlamBlockSolver>(std::move(linearSolver)));
	c.optimizer_->setAlgorithm(c.solverLevenberg_);
}

}
#endif
