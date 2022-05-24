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
#include "g2o/types/slam2d/edge_xy_prior.h"
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>

#include <boost/bimap.hpp>
#include <boost/container/allocator.hpp>
#include <yaml-cpp/yaml.h>

#include "misc/EigenYamlSerialization.hpp"
#include <misc/termcolor.hpp>

#ifdef _PERFTOOLS_CPU
#include <gperftools/profiler.h>
#endif
#ifdef _PERFTOOLS_HEAP
#include <gperftools/heap-profiler.h>
#endif

namespace rfs {

struct bimap_less {
	bool operator()(const boost::bimap<int, int, boost::container::allocator<int>> x,
			const boost::bimap<int, int, boost::container::allocator<int>> y) const {

		return x.left < y.left;
	}
};

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
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	typedef g2o::VertexPointXY PointType;
	typedef g2o::VertexSE2 PoseType;
	typedef g2o::EdgeSE2PointXY MeasurementEdge;
	typedef g2o::EdgeXYPrior PointAnchorEdge;

	typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> > SlamBlockSolver;
	typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	g2o::SparseOptimizer *optimizer_;

	g2o::OptimizationAlgorithmLevenberg *solverLevenberg_;
	SlamLinearSolver *linearSolver_;
	SlamBlockSolver *blockSolver_;

	std::vector<boost::bimap<int, int, boost::container::allocator<int>>> DA_bimap_, prevDA_bimap_; /**< Bimap containing data association hypothesis at time k  */

	std::vector<std::vector<MeasurementEdge*> > Z_; /**< Measurement edges stored, in order to set data association and add to graph later */
	std::vector<std::vector<AssociationProbabilities> > DAProbs_; /**< DAProbs_ [k][nz] are is the association probabilities of measurement
	 nz at time k, used for switching using gibbs sampling*/
	std::vector<std::vector<int> > fov_; /**< indices of landmarks in field of view at time k */

	std::vector<PoseType*> poses_;
	std::vector<PointType*> landmarks_;
	std::vector<int> landmarks_numDetections_, prevlandmarks_numDetections_,
			landmarks_numFoV_;
	std::vector<double> landmarksResetProb_, landmarksInitProb_;

	double logweight_ = -std::numeric_limits<double>::infinity(),
			prevLogWeight_ = -std::numeric_limits<double>::infinity();
	int numPoses_, numPoints_;
	bool reverted_ = false;
	std::vector<std::pair<int, int>> tomerge_; /**< proportional to probability of merge heuristic*/
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
	typedef g2o::EdgeXYPrior PointAnchorEdge;
	typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> > SlamBlockSolver;
	typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	/**
	 * \brief Configurations for this  optimizer
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

		int birthDeathNumIter_; /**< apply birth and death every n iterations */

		std::vector<double> xlim_, ylim_;

		int numLandmarks_; /**< number of landmarks per dimension total landmarks will be numlandmarks^2 */

		int numGibbs_; /**< number of gibbs samples of the data association */
		int numLevenbergIterations_; /**< number of gibbs samples of the data association */
		int crossoverNumIter_;
		int numPosesToOptimize_; /**< number of poses to optimize data associations */

		int lmExistenceProb_;
		int numIterations_; /**< number of iterations of main algorithm */
		double initTemp_;
		double tempFactor_;
		Eigen::Matrix2d anchorInfo_; /** information for anchor edges, should be low*/

		std::string finalStateFile_;

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
	 * Select nearest neighbor at the last included time step (maxpose_)
	 * @param c the GLMB component
	 */
	void  selectNN(VectorGLMBComponent2D &c);
	/**
	 * Sample n data associations from the already visited group, in order to perform gibbs sampler on each.
	 * @param ni number of iterations of the optimizer
	 */
	void sampleComponents();

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
	void updateDAProbs(VectorGLMBComponent2D &c, int minpose, int maxpose);

	/**
	 * Calculate the FoV at each time
	 * @param c the GLMB component
	 */
	void updateFoV(VectorGLMBComponent2D &c);

	/**
	 * Use the probabilities calculated using updateDAProbs to sample a new data association through gibbs sampling
	 * @param c the GLMB component
	 */
	double sampleDA(VectorGLMBComponent2D &c);

	/**
	 * Merge two data associations into a third one, by selecting a random merge time.
	 */
	std::vector<boost::bimap<int, int, boost::container::allocator<int>> > sexyTime(VectorGLMBComponent2D &c1, VectorGLMBComponent2D &c2);

	/**
	 * Use the probabilities calculated in sampleDA to reset all the detections of a single landmark to all false alarms.
	 * @param c the GLMB component
	 */
	double sampleLMDeath(VectorGLMBComponent2D &c);

	/**
	 * Randomly merge landmarks in order to improve the sampling algorithm
	 * @param c the GLMB component
	 */
	double mergeLM(VectorGLMBComponent2D &c);

	/**
	 * Use the probabilities calculated in sampleDA to initialize landmarks from  false alarms.
	 * @param c the GLMB component
	 */
	double sampleLMBirth(VectorGLMBComponent2D &c);

	/**
	 * Change the data association in component c to the one stored on da.
	 */
	void changeDA(VectorGLMBComponent2D &c,
			const std::vector<boost::bimap<int, int, boost::container::allocator<int>> > &da);

	/**
	 * Revert the current data association to keep the last one
	 * @param c the GLMB component
	 */
	void revertDA(VectorGLMBComponent2D &c);

	/**
	 * print the data association in component c
	 * @param c the GLMB component
	 */
	void printDA(VectorGLMBComponent2D &c, std::ostream &s = std::cout);
	/**
	 * print the data association in component c
	 * @param c the GLMB component
	 */
	void printDAProbs(VectorGLMBComponent2D &c);
	/**
	 * print the data association in component c
	 * @param c the GLMB component
	 */
	void printFoV(VectorGLMBComponent2D &c);
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

	/**
	 * @brief Calculate an angle from a pose to a landmark
	 * 
	 * @param pose A 2D pose
	 * @param lm A 2D landmark
	 * @return double angle in radians
	 */
	static double angle(const PoseType *pose, const PointType *lm);

	int nThreads_; /**< Number of threads  */

	VectorGLMBComponent2D gt_graph;

	std::vector<VectorGLMBComponent2D> components_; /**< VGLMB components */
	double bestWeight_ = -std::numeric_limits<double>::infinity();
	std::vector<boost::bimap<int, int, boost::container::allocator<int>> > best_DA_;
	int best_DA_max_detection_time_ = 0; /**< last association time */

	std::map<std::vector<boost::bimap<int, int, boost::container::allocator<int>> >, double> visited_;
	double temp_;
	int minpose_=0; /**< sample data association from this pose  onwards*/
	int maxpose_=0; /**< optimize only up to this pose */
	int maxpose_prev_ =0;
	int iteration_ = 0;
	int iterationBest_ = 0;
	double insertionP_ = 0.5;

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

void VectorGLMBSLAM2D::changeDA(VectorGLMBComponent2D &c,
		const std::vector<boost::bimap<int, int, boost::container::allocator<int>> > &da) {
	// update bimaps!!!
	c.DA_bimap_ = da;

	// update the numdetections
	std::fill(c.landmarks_numDetections_.begin(),
			c.landmarks_numDetections_.end(), 0);
	for (auto &bimap : da) {
		for (auto it = bimap.begin(), it_end = bimap.end(); it != it_end;
				it++) {
			c.landmarks_numDetections_[it->right - c.landmarks_[0]->id()]++;
		}
	}
	updateGraph(c);
	c.poses_[0]->setFixed(true);
	c.optimizer_->initializeOptimization();
	//c.optimizer_->computeInitialGuess();
	c.optimizer_->setVerbose(false);
	c.optimizer_->optimize(config.numLevenbergIterations_);

}
void VectorGLMBSLAM2D::sampleComponents() {

	// do logsumexp on the components to calculate probabilities

	std::vector<double> probs(visited_.size(), 0);
	double maxw = -std::numeric_limits<double>::infinity();
	for (auto it = visited_.begin(), it_end = visited_.end(); it != it_end;
			it++) {
		if (it->second > maxw)
			maxw = it->second;
	}
	int i = 1;
	probs[0] = std::exp((visited_.begin()->second - maxw) / temp_);
	for (auto it = std::next(visited_.begin()), it_end = visited_.end();
			it != it_end; i++, it++) {
		if (it->second > maxw)
			maxw = it->second;
		probs[i] = probs[i - 1] + std::exp((it->second - maxw) / temp_);
	}
/*
	 std:: cout << "maxw " << maxw <<  "  temp " << temp_ << "\n";
	 std::cout << "probs " ;
	 for (auto p:probs){std::cout <<"  " << p;}
	 std::cout <<"\n";*/
	 
	boost::uniform_real<> dist(0.0,
			probs[probs.size() - 1] / components_.size());

	double r = dist(randomGenerators_[0]);
	int j = 0;
	auto it = visited_.begin();
	for (int i = 0; i < components_.size(); i++) {
		while (probs[j] < r) {
			j++;
			it++;
		}

		// add data association j to component i
		changeDA(components_[i], it->first);
		components_[i].logweight_ = it->second;

		//std::cout  << "sample w: " << it->second << " j " << j  << " r " << r <<" prob "  << probs[j]<< "\n";

		r += probs[probs.size() - 1] / components_.size();

	}

}

void VectorGLMBSLAM2D::load(std::string filename) {
	std::ifstream ifs(filename, std::ifstream::in);

	gt_graph.optimizer_->load(ifs);

	ifs.close();

	gt_graph.numPoses_ = 0;
	gt_graph.numPoints_ = 0;
	//Copy Vertices from optimizer with data association
	int maxid = 0;
	for (auto pair : gt_graph.optimizer_->vertices()) {
		g2o::HyperGraph::Vertex *v = pair.second;
		PoseType *pose = dynamic_cast<PoseType*>(v);
		if (pose != NULL) {

			gt_graph.poses_.push_back(pose);
			gt_graph.numPoses_++;

			if (maxid < pose->id()) {
				maxid = pose->id();
			}
		}
		//sort by id

		PointType *point = dynamic_cast<PointType*>(v);
		if (point != NULL) {

			gt_graph.landmarks_.push_back(point);
			gt_graph.numPoints_++;
		}

	}
	std::sort(gt_graph.poses_.begin(), gt_graph.poses_.end(),
			[](const auto &lhs, const auto &rhs) {
				return lhs->id() < rhs->id();
			});
	std::sort(gt_graph.landmarks_.begin(), gt_graph.landmarks_.end(),
			[](const auto &lhs, const auto &rhs) {
				return lhs->id() < rhs->id();
			});

	gt_graph.DA_bimap_.resize(gt_graph.numPoses_);

	gt_graph.Z_.resize(gt_graph.numPoses_);
	gt_graph.fov_.resize(gt_graph.numPoses_);
	gt_graph.landmarks_numDetections_.resize(gt_graph.landmarks_.size(), 1);
	for (g2o::HyperGraph::Edge *e : gt_graph.optimizer_->edges()) {

		MeasurementEdge *z = dynamic_cast<MeasurementEdge*>(e);
		if (z != NULL) {
			int firstvertex = z->vertex(0)->id();
			gt_graph.Z_[firstvertex - gt_graph.poses_[0]->id()].push_back(z);
			gt_graph.fov_[firstvertex - gt_graph.poses_[0]->id()].push_back(
					z->vertex(1)->id());
		}

	}

	gt_graph.poses_[0]->setFixed(true);
	gt_graph.optimizer_->initializeOptimization();
	gt_graph.optimizer_->optimize(10);

}

void VectorGLMBSLAM2D::loadConfig(std::string filename) {

	YAML::Node node = YAML::LoadFile(filename);

	config.MeasurementLikelihoodThreshold_ =
			node["MeasurementLikelihoodThreshold"].as<double>();
	config.lmExistenceProb_ = node["lmExistenceProb"].as<double>();
	config.logKappa_ = node["logKappa"].as<double>();
	config.PE_ = node["PE"].as<double>();
	config.PD_ = node["PD"].as<double>();
	config.maxRange_ = node["maxRange"].as<double>();
	config.numComponents_ = node["numComponents"].as<int>();
	config.birthDeathNumIter_ = node["birthDeathNumIter"].as<int>();
	config.numLandmarks_ = node["numLandmarks"].as<int>();
	config.numGibbs_ = node["numGibbs"].as<int>();
	config.numIterations_ = node["numIterations"].as<int>();
	config.numLevenbergIterations_ = node["numLevenbergIterations"].as<int>();
	config.xlim_.push_back(node["xlim"][0].as<double>());
	config.xlim_.push_back(node["xlim"][1].as<double>());
	config.ylim_.push_back(node["ylim"][0].as<double>());
	config.ylim_.push_back(node["ylim"][1].as<double>());
	config.initTemp_ = node["initTemp"].as<double>();
	config.tempFactor_ = node["tempFactor"].as<double>();

	config.crossoverNumIter_ = node["crossoverNumIter"].as<int>();
	config.numPosesToOptimize_ = node["numPosesToOptimize"].as<int>();
	config.finalStateFile_ = node["finalStateFile"].as<std::string>();

	if (!YAML::convert<Eigen::Matrix2d>::decode(node["anchorInfo"],
			config.anchorInfo_)) {
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

double VectorGLMBSLAM2D::angle(const PoseType *pose, const PointType *lm){
	Eigen::Vector3d posemean;
	pose->getEstimateData(posemean.data());
	Eigen::Vector2d pointmean;
	lm->getEstimateData(pointmean.data());
    double bearing=atan2(pointmean[1]-posemean[1] , pointmean[0]-posemean[0])-posemean[2];
	bearing= fmod(bearing + M_PI,2*M_PI);
	if (bearing <0){
		bearing+= 2*M_PI;
	}
	bearing-=M_PI;

	return bearing;

}
inline void VectorGLMBSLAM2D::initComponents() {
	components_.resize(config.numComponents_);
	gt_graph.optimizer_->computeInitialGuess();
	gt_graph.optimizer_->save("inittraj.g2o");

	for (auto &c : components_) {
		init(c);
		constructGraph(c);

		// optimize once at the start to calculate the hessian.
		c.poses_[0]->setFixed(true);
		c.optimizer_->initializeOptimization();
		//c.optimizer_->computeInitialGuess();
		c.optimizer_->setVerbose(true);
		std::cout << "niterations  " << c.optimizer_->optimize(1) << "\n";
	}

}
inline void VectorGLMBSLAM2D::run(int numSteps) {


	for (int i = 0; i < numSteps; i++) {
		maxpose_prev_ = maxpose_;
		maxpose_ = components_[0].poses_.size() * i / (numSteps*0.95);

		if (best_DA_max_detection_time_ + config.numPosesToOptimize_ < maxpose_ ){
		 	maxpose_ = best_DA_max_detection_time_ + config.numPosesToOptimize_;
		 }
		if (maxpose_ > components_[0].poses_.size())
			maxpose_ = components_[0].poses_.size();
		if(best_DA_max_detection_time_==components_[0].poses_.size()-1){
			minpose_ = 0;
		}else{
			minpose_ = std::max(0,std::min(maxpose_-2*config.numPosesToOptimize_ , best_DA_max_detection_time_-config.numPosesToOptimize_) );
		}
		//minpose_ = 0;
		std::cout << "maxpose: " << maxpose_ << " max det:  " << best_DA_max_detection_time_<< "  "<< minpose_ <<"\n";
		std::cout << "iteration: " << iteration_ << " / " << numSteps<< "\n";
		optimize(config.numLevenbergIterations_);

	}

}

void VectorGLMBSLAM2D::selectNN(VectorGLMBComponent2D &c)
{
	std::vector<boost::bimap<int, int, boost::container::allocator<int>>> out;
	out.resize(c.DA_bimap_.size());
	for (int i = 0; i < maxpose_; i++)
	{
		out[i] = c.DA_bimap_[i];
	}
	int max_detection_time = maxpose_-1;
	while(max_detection_time > 0 && c.DA_bimap_[max_detection_time].size()==0){
		max_detection_time--;
	}

	for (int k = max_detection_time+1 ; k < maxpose_; k++)
	{

		updateGraph(c);
		c.poses_[0]->setFixed(true);
		c.optimizer_->initializeOptimization();
		c.optimizer_->computeInitialGuess();
		c.optimizer_->setVerbose(false);
		c.optimizer_->optimize(30);
		calculateWeight(c);
		updateDAProbs(c, k, k+1);
		
		AssociationProbabilities probs;
		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
		{
			// std::cout << "selectNN\n";
			probs.i.clear();
			probs.l.clear();
			double maxprob = -std::numeric_limits<double>::infinity();
			int maxprobi = 0;
			auto it = c.DA_bimap_[k].left.find(nz);
			double selectedProb;
			int selectedDA = -2;
			if (it != c.DA_bimap_[k].left.end())
			{
				selectedDA = it->second;
			}
			double maxlikelihood = -std::numeric_limits<double>::infinity();
			int maxlikelihoodi = 0;
			for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++)
			{
				double likelihood = c.DAProbs_[k][nz].l[a];

				if (maxlikelihood < likelihood)
				{
					maxlikelihood = likelihood;
					maxlikelihoodi = a;
				}
				if (c.DAProbs_[k][nz].i[a] == -2)
				{
					probs.i.push_back(c.DAProbs_[k][nz].i[a]);
					probs.l.push_back(c.DAProbs_[k][nz].l[a]);
					if (c.DAProbs_[k][nz].l[a] > maxprob)
					{
						maxprob = c.DAProbs_[k][nz].l[a];
						maxprobi = a;
					}
				}
				else if (c.DAProbs_[k][nz].i[a] == selectedDA)
				{
					probs.i.push_back(c.DAProbs_[k][nz].i[a]);
					if (c.landmarks_numDetections_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0]->id()] == 1)
					{
						likelihood += std::log(config.PE_) - std::log(1 - config.PE_) + (c.landmarks_numFoV_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0]->id()]) * std::log(1 - config.PD_);
						// std::cout <<" single detection: increase:  " << std::log(config.PE_)-std::log(1-config.PE_) + (c.landmarks_numFoV_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0]->id()])*std::log(1-config.PD_) <<"\n";
						probs.l.push_back(likelihood);
					}
					else
					{
						probs.l.push_back(c.DAProbs_[k][nz].l[a]);
					}
					if (likelihood > maxprob)
					{
						maxprob = likelihood;
						maxprobi = a;
					}
				}
				else
				{
					if (c.DA_bimap_[k].right.count(c.DAProbs_[k][nz].i[a]) == 0)
					{ // landmark is not already associated to another measurement
						probs.i.push_back(c.DAProbs_[k][nz].i[a]);
						if (c.landmarks_numDetections_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0]->id()] == 0)
						{
							likelihood +=
								std::log(config.PE_) - std::log(1 - config.PE_) + (c.landmarks_numFoV_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0]->id()]) * std::log(1 - config.PD_);
							// std::cout <<" 0 detection: increase:  " << std::log(config.PE_)-std::log(1-config.PE_) + (c.landmarks_numFoV_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0]->id()])*std::log(1-config.PD_)<<"\n";
							probs.l.push_back(likelihood);
						}
						else
						{
							probs.l.push_back(c.DAProbs_[k][nz].l[a]);
						}
						if (likelihood > maxprob)
						{
							maxprob = likelihood;
							maxprobi = a;
						}
					}
				}
			}
			int newdai = c.DAProbs_[k][nz].i[maxprobi];

			if (newdai == -2)
			{
				//std::cout << "false alarm\n";
			}
			// std::cout << "ass\n";
			if (newdai != selectedDA)
			{ // if selected association, change bimap

				if (newdai >= 0)
				{
					c.landmarks_numDetections_[newdai - c.landmarks_[0]->id()]++;
					if (selectedDA < 0)
					{
						c.DA_bimap_[k].insert({nz, newdai});
					}
					else
					{

						c.landmarks_numDetections_[selectedDA - c.landmarks_[0]->id()]--;
						c.DA_bimap_[k].left.replace_data(it, newdai);
					}
				}
				else
				{ // if a change has to be made and new DA is false alarm, we need to remove the association
					c.DA_bimap_[k].left.erase(it);
					c.landmarks_numDetections_[selectedDA - c.landmarks_[0]->id()]--;
				}
			}
		}
	}
}

std::vector<boost::bimap<int, int, boost::container::allocator<int>> > VectorGLMBSLAM2D::sexyTime(VectorGLMBComponent2D &c1,
		VectorGLMBComponent2D &c2) {


	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif
if (maxpose_== 0){
	std::vector<boost::bimap<int, int, boost::container::allocator<int>> > out(c1.DA_bimap_);
	return out;
}
	boost::uniform_int<> random_merge_point(-maxpose_, maxpose_);
	std::vector<boost::bimap<int, int, boost::container::allocator<int>> > out;
	out.resize(c1.DA_bimap_.size());
	int merge_point = random_merge_point(rfs::randomGenerators_[threadnum]);

	if (merge_point >= 0) {
		// first half from da1 second from da2
		for (int i = merge_point; i < c2.DA_bimap_.size(); i++) {
			out[i] =c2.DA_bimap_[i];

		}
		for (int i = 0; i < merge_point; i++) {
			out[i] =c1.DA_bimap_[i];
		}
	} else {
		// first half from da2 second from da1
		for (int i = 0; i < -merge_point; i++) {
			out[i] =c2.DA_bimap_[i];
		}
		for (int i = -merge_point; i < c2.DA_bimap_.size(); i++) {
			out[i] =c1.DA_bimap_[i];

		}

	}
	return out;
}
inline void VectorGLMBSLAM2D::optimize(int ni) {

		std::cout << "visited  " << visited_.size() << "\n" ;
	if (visited_.size() > 0) {
		sampleComponents();
		std::cout << "sampling compos \n";
	}

	if (iteration_ % config.crossoverNumIter_ == 0 and false) {

		std::cout << termcolor::magenta
				<< " =========== SexyTime!============\n"
				<< termcolor::reset;
		for (int i = 0; i < components_.size(); i++) {
			auto &c = components_[i];

			boost::uniform_int<> random_component(0, components_.size() - 1);
			int secondComp;
			do {
				secondComp = random_component(rfs::randomGenerators_[0]);
			} while (secondComp == i);
			auto da = sexyTime(c, components_[secondComp]);
			changeDA(c,da);
		}
#pragma omp parallel for
		for (int i = 0; i < components_.size(); i++) {

			if (maxpose_prev_ != maxpose_) {
				auto &c = components_[i];
				//selectNN( c);

				updateFoV(c);
				//std::cout << "fov update: \n";

				updateDAProbs(c , minpose_ , maxpose_);
				//std::cout << "da update: \n";
				c.prevDA_bimap_ = c.DA_bimap_;
				c.prevlandmarks_numDetections_ = c.landmarks_numDetections_;
				double expectedChange = 0;
				bool inserted;
				selectNN(c);
			//	std::cout << "nn update: \n";
				updateGraph(c);
				c.poses_[0]->setFixed(true);
				c.optimizer_->initializeOptimization();
				c.optimizer_->computeInitialGuess();
				c.optimizer_->setVerbose(false);
				c.optimizer_->optimize(ni);
				calculateWeight(c);

				std::map<std::vector<boost::bimap<int, int, boost::container::allocator<int>> >, double>::iterator it;
#pragma omp critical(bestweight)
				{
					if (c.logweight_ > bestWeight_) {
						bestWeight_ = c.logweight_;
						best_DA_ = c.DA_bimap_;
						best_DA_max_detection_time_ = maxpose_-1;
						while(best_DA_max_detection_time_ > 0 && best_DA_[best_DA_max_detection_time_].size()==0){
							best_DA_max_detection_time_--;
						}
						std::stringstream filename;

						filename << "video/beststate_" << std::setfill('0')
								<< std::setw(5) << iterationBest_++ << ".g2o";
						c.optimizer_->save(filename.str().c_str(), 0);
						std::cout << termcolor::yellow << "========== newbest:"
								<< bestWeight_ << " ============\n"
								<< termcolor::reset;
					}
				}
				auto pair = std::make_pair(c.DA_bimap_, c.logweight_);
#pragma omp critical(insert)
			{
			std::tie(it, inserted) = visited_.insert(pair);
			insertionP_ = insertionP_*0.99;
			if (inserted)
				insertionP_ +=0.01;
			}
				it->second = c.logweight_;

			}
		}

	}
#pragma omp parallel for
	for (int i = 0; i < components_.size(); i++) {
		auto &c = components_[i];

		int threadnum = 0;
#ifdef _OPENMP
	threadnum = omp_get_thread_num();
	#endif

		updateFoV(c);
		if (!c.reverted_ )
			updateDAProbs(c, minpose_, maxpose_);
		for (int p=0 ; p< maxpose_; p++){
			c.prevDA_bimap_[p] = c.DA_bimap_[p];
		}
		c.prevlandmarks_numDetections_ = c.landmarks_numDetections_;
		double expectedChange = 0;
		bool inserted;
		std::map<std::vector<boost::bimap<int, int, boost::container::allocator<int>> >, double>::iterator it;

		 {
//do{
			selectNN(c);
			expectedChange += sampleDA(c);
			/*
			 std::cout << termcolor::magenta <<" =========== SexyTime!============\n" << termcolor::reset;
			 boost::uniform_int<> random_component(0, components_.size()-1);
			 int secondComp;
			 do{
			 secondComp=random_component(rfs::randomGenerators_[threadnum]);
			 }while(secondComp==i);
			 sexyTime(c,components_[secondComp]);
			 */

			if (iteration_ % config.birthDeathNumIter_ == 0) {
				switch ((iteration_ / config.birthDeathNumIter_) % 3) {
				case 0:
					expectedChange += sampleLMBirth(c);
					break;
				case 1:
					//expectedChange += sampleLMDeath(c);
					break;

				case 2:
					//
					break;

				}

			}

			for (int i = 1; i < config.numGibbs_; i++) {
				expectedChange += sampleDA(c);
				//
			}
			// if (iteration_ % config.birthDeathNumIter_ == 0) {
			// 	if ((iteration_ / config.birthDeathNumIter_) % 3 ==2) {
			// 		expectedChange += mergeLM(c);
			// 	}
			// }

		}
			//expectedChange += sampleLMDeath(c);
			//expectedChange += sampleLMBirth(c);
			//expectedChange += sampleLMDeath(c);
			std::pair<std::vector<boost::bimap<int, int, boost::container::allocator<int>> >, double> pair(c.DA_bimap_, c.logweight_);

#pragma omp critical(insert)
			{
			std::tie(it, inserted) =
			visited_.insert(pair);
			insertionP_ = insertionP_*0.99;
			if (inserted)
				insertionP_ +=0.01;
			}

			/*
			if (!inserted) {
				std::cout << "data association already inserted\n";
			}
			*/
//}while(!inserted);
			//printFoV(c);
			/*  print data association
			 if(i==0){
			 std::ofstream dafile;
			 std::stringstream filename;
			 filename << "DA__" << iteration_ << ".txt";
			 dafile.open(filename.str());
			 std::cout<<" iteraton " <<iteration_++  << " :::: \n";
			 printDA(c,dafile);
			 }
			 */
			//printDAProbs(c);
		updateGraph(c);
		c.poses_[0]->setFixed(true);
		c.optimizer_->initializeOptimization();
		c.optimizer_->computeInitialGuess();
		c.optimizer_->setVerbose(false);
		c.optimizer_->optimize(ni);
		//std::cout <<"niterations  " <<c.optimizer_->optimize(ni) << "\n";
		calculateWeight(c);

#pragma omp critical(bestweight)
		{
			if (c.logweight_ > bestWeight_) {
				bestWeight_ = c.logweight_;
				best_DA_ = c.DA_bimap_;
				std::stringstream filename;

				best_DA_max_detection_time_ = std::max(maxpose_-1 , 0 );
				while(best_DA_max_detection_time_ > 0 && best_DA_[best_DA_max_detection_time_].size()==0){
					best_DA_max_detection_time_--;
				}
				filename << "video/beststate_" << std::setfill('0')
						<< std::setw(5) << iterationBest_++ << ".g2o";
				c.optimizer_->save(filename.str().c_str(), 0);
				std::cout << termcolor::yellow << "========== newbest:"
						<< bestWeight_ << " ============\n" << termcolor::reset;
			}
		}
		it->second = c.logweight_;
		//double accept = std::min(1.0 ,  std::exp(c.logweight_-c.prevLogWeight_ - std::min(expectedChange, 0.0) ));

		/*
		 boost::uniform_real<> uni_dist(0, 1);

		 std::cout << "accept: " << accept << "thred " << threadnum<<"\n";
		 std::cout << "weight: " << c.logweight_ << " prevWeight: " << c.prevLogWeight_ << " expectedChange " << expectedChange << "   chi2:  " <<c.optimizer_->activeChi2() << "  determinant: " << c.linearSolver_->_determinant<< "\n";


		 if(uni_dist(rfs::randomGenerators_[threadnum]) > accept){
		 std::cout << "===================================================REVERT========================================================\n";
		 revertDA(c);
		 }else{
		 c.reverted_= false;
		 } */

	}

	iteration_++;

	//temp_ *= config.tempFactor_;

std::cout << "insertionp: " << insertionP_ << " temp: " << temp_ << "\n";

	if (insertionP_> 0.95 && temp_ > 5e-1){
		temp_*=0.5;
		std::cout << "reducing: \n";
	}
	if (insertionP_ < 0.05 && temp_ < 1e5){
		temp_*=2;
		std::cout << "augmenting: ";
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
			if (selectedDA < 0) {
				logw += config.logKappa_;
			} else {
				logw +=
						-0.5
								* (c.Z_[k][nz]->dimension() * std::log(2 * M_PI)
										- std::log(
												c.Z_[k][nz]->information().determinant()));
			}
		}

		for (int lm = 0; lm < c.fov_[k].size(); lm++) {

			if (c.DA_bimap_[k].right.count(c.fov_[k][lm]) > 0) {
				logw += std::log(config.PD_);
			} else {
				bool exists = c.landmarks_numDetections_[c.fov_[k][lm]
						- c.landmarks_[0]->id()] > 0;
				if (exists) {
					logw += std::log(1 - config.PD_);
				}
			}
		}
	}

	for (int lm = 0; lm < c.landmarks_.size(); lm++) {
		bool exists = c.landmarks_numDetections_[lm] > 0;
		if (exists) {
			logw += std::log(config.PE_);
		} else {
			logw += std::log(1 - config.PE_);
		}
	}
	logw += -0.5 * (c.optimizer_->activeChi2() + c.linearSolver_->_determinant);
	//std::cout << termcolor::blue << "weight: " <<logw << " det: " <<     c.linearSolver_->_determinant      <<termcolor::reset <<"\n";
	c.prevLogWeight_ = c.logweight_;
	c.logweight_ = logw;
}

inline void VectorGLMBSLAM2D::updateGraph(VectorGLMBComponent2D &c) {
	for (int k = 0; k < maxpose_; k++) {
		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
			int selectedDA = -2;
			auto it = c.DA_bimap_[k].left.find(nz);

			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}
			int previd =
					c.Z_[k][nz]->vertex(1) ? c.Z_[k][nz]->vertex(1)->id() : -2; /**< previous data association */
			if (previd == selectedDA) {
				continue;
			}
			if (selectedDA >= 0) {

				// if edge was already in graph, modify it
				if (previd >= 0) {
					c.optimizer_->setEdgeVertex(c.Z_[k][nz], 1,
							dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
									selectedDA)->second)); // this removes the edge from the list in both vertices
				} else {
					c.Z_[k][nz]->setVertex(1,
							dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
									selectedDA)->second));

					c.optimizer_->addEdge(c.Z_[k][nz]);
				}

			} else {

				c.optimizer_->removeEdge(c.Z_[k][nz]);
				c.Z_[k][nz]->setVertex(1, NULL);

			}

		}
	}
	for (auto lm : c.landmarks_) {
		// if landmark has only 1 edge then it is not detected we deactivate it
		if (lm->edges().size() == 1) {
			for (auto edge : lm->edges()) {
				dynamic_cast<g2o::OptimizableGraph::Edge*>(edge)->setLevel(1);
			}
		} else {
			for (auto edge : lm->edges()) {
				dynamic_cast<g2o::OptimizableGraph::Edge*>(edge)->setLevel(0);
			}

		}

	}
}
template<class MapType>
void print_map(const MapType &m, std::ostream &s = std::cout) {
	typedef typename MapType::const_iterator const_iterator;
	for (const_iterator iter = m.begin(), iend = m.end(); iter != iend;
			++iter) {
		s << iter->first << "-->" << iter->second << std::endl;
	}
}
inline void VectorGLMBSLAM2D::printFoV(VectorGLMBComponent2D &c) {
	std::cout << "FoV:\n";
	for (int k = 0; k < c.fov_.size(); k++) {
		std::cout << k << "  FoV at:   ";
		for (int lmid : c.fov_[k]) {
			std::cout << "  ,  " << lmid;
		}
		std::cout << "\n";
	}
}
inline void VectorGLMBSLAM2D::printDAProbs(VectorGLMBComponent2D &c) {
	for (int k = 0; k < c.DAProbs_.size(); k++) {
		if (k == 2)
			break;
		std::cout << k << "da probs:\n";
		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
			std::cout << "z =  " << nz << "  ;";
			for (double l : c.DAProbs_[k][nz].l) {
				std::cout << std::max(l, -100.0) << " , ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
}
inline void VectorGLMBSLAM2D::printDA(VectorGLMBComponent2D &c,
		std::ostream &s) {
	for (int k = 0; k < c.DA_bimap_.size(); k++) {
		s << k << ":\n";
		print_map(c.DA_bimap_[k].left, s);
	}
}

inline void VectorGLMBSLAM2D::revertDA(VectorGLMBComponent2D &c) {
	c.logweight_ = c.prevLogWeight_;
	c.DA_bimap_ = c.prevDA_bimap_;
	c.landmarks_numDetections_ = c.prevlandmarks_numDetections_;
	updateGraph(c);
	c.reverted_ = true;
}

inline double VectorGLMBSLAM2D::sampleLMBirth(VectorGLMBComponent2D &c) {
	double expectedWeightChange = 0;
	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif

	for (int i = 0; i < c.landmarks_.size(); i++) {
		if (c.landmarks_numDetections_[i] > 0 || c.landmarks_numFoV_[i] == 0) {
			continue;
		}
		//
		c.landmarksInitProb_[i] = c.landmarksInitProb_[i]
				/ (c.landmarks_numFoV_[i] * config.PD_);
		if (c.landmarksInitProb_[i] > uni_dist(randomGenerators_[threadnum])) {
			// reset all associations to false alarms
			expectedWeightChange += (config.logKappa_ + (1 - config.PD_))
					* c.landmarks_numFoV_[i];
			int numdet = 0;
			for (int k = minpose_; k < maxpose_; k++) {
				for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
					// if measurement is associated, continue
					auto it = c.DA_bimap_[k].left.find(nz);
					if (it != c.DA_bimap_[k].left.end()) {
						continue;
					}
					for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++) {
						if (c.DAProbs_[k][nz].i[a] == c.landmarks_[i]->id()) {
							if (c.DAProbs_[k][nz].l[a]
									> c.DAProbs_[k][nz].l[c.DAProbs_[k][nz].l.size()
											- 1]) {
								c.DA_bimap_[k].insert( { nz,
										c.landmarks_[i]->id() });
								c.landmarks_numDetections_[i]++;
								expectedWeightChange +=
										c.DAProbs_[k][nz].l[a]
												- c.DAProbs_[k][nz].l[c.DAProbs_[k][nz].l.size()
														- 1];
							}

						}
					}

				}

			}
			expectedWeightChange += std::log(config.PE_)
					- std::log(1 - config.PE_);
/*
			std::cout << termcolor::green << "LANDMARK BORN "
					<< termcolor::reset << " initprob: "
					<< c.landmarksInitProb_[i] << " numDet "
					<< c.landmarks_numDetections_[i] << " numfov: "
					<< c.landmarks_numFoV_[i] << "  expectedChange "
					<< expectedWeightChange << "\n";
					*/

			c.landmarks_numDetections_[i] = 0;

		}
	}

//std::cout << "Death Change  " <<expectedWeightChange <<"\n";
	return expectedWeightChange;
}

inline double VectorGLMBSLAM2D::mergeLM(VectorGLMBComponent2D &c) {
	double expectedWeightChange = 0;
	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif

	if (c.tomerge_.size() == 0) {
		std::cout << termcolor::blue << "no jumps so no merge \n"
				<< termcolor::reset;
		return 0;
	}
	boost::uniform_int<> random_pair(0, c.tomerge_.size() - 1);

	int rp = random_pair(rfs::randomGenerators_[threadnum]);

	int todelete = c.tomerge_[rp].first;
	int toAddMeasurements = c.tomerge_[rp].second;

	for (int k = minpose_; k < maxpose_; k++) {
		auto it = c.DA_bimap_[k].right.find(todelete);
		if (it != c.DA_bimap_[k].right.end()) {
			for (int l = 0; l < c.DAProbs_[k][it->second].i.size(); l++) {
				if (c.DAProbs_[k][it->second].i[l] == it->first) {

					expectedWeightChange -= c.DAProbs_[k][it->second].l[l];
					break;

				}
			}
			c.landmarks_numDetections_[todelete - c.landmarks_[0]->id()]--;
			c.landmarks_numDetections_[toAddMeasurements - c.landmarks_[0]->id()]++;

			c.DA_bimap_[k].right.replace_key(it, toAddMeasurements);

		}
	}
	if (c.landmarks_numDetections_[todelete - c.landmarks_[0]->id()] != 0) {
		std::cerr << "landmarks_numDetections_ not zero"
				<< c.landmarks_numDetections_[todelete - c.landmarks_[0]->id()]
				<< "\n";
	}
	c.tomerge_.clear();
	return expectedWeightChange;
}

inline double VectorGLMBSLAM2D::sampleLMDeath(VectorGLMBComponent2D &c) {
	double expectedWeightChange = 0;
	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif

	for (int i = 0; i < c.landmarks_.size(); i++) {
		if (c.landmarks_numDetections_[i] <= 1) {
			continue;
		}

		c.landmarksResetProb_[i] = -(c.landmarks_numDetections_[i])* std::log(config.PD_)-(c.landmarks_numFoV_[i]-c.landmarks_numDetections_[i])
				* std::log(1 - config.PD_)-std::log(config.PE_)+std::log(1-config.PE_);
		double p=std::exp(c.landmarksResetProb_[i]);
		if ( uni_dist(randomGenerators_[threadnum]) < p / (1 + p)) {
		/*
		c.landmarksResetProb_[i] = (1-((double)c.landmarks_numDetections_[i])/c.landmarks_numFoV_[i])*(config.PD_);
		if(uni_dist(randomGenerators_[threadnum]) < c.landmarksResetProb_[i]){*/
			// reset all associations to false alarms
			expectedWeightChange += config.logKappa_
					* c.landmarks_numDetections_[i];
			int numdet = 0;
			for (int k = minpose_; k < maxpose_; k++) {
				auto it = c.DA_bimap_[k].right.find(c.landmarks_[i]->id());
				if (it != c.DA_bimap_[k].right.end()) {
					for (int l = 0; l < c.DAProbs_[k][it->second].i.size();
							l++) {
						if (c.DAProbs_[k][it->second].i[l] == it->first) {
							numdet++;
							expectedWeightChange -=
									c.DAProbs_[k][it->second].l[l];
							break;

						}
					}
					c.DA_bimap_[k].right.erase(it);
				}
			}
			expectedWeightChange += std::log(1 - config.PE_)
					- std::log(config.PE_);
			expectedWeightChange += -std::log(1 - config.PD_)
					* c.landmarks_numFoV_[i];
/*
			std::cout << termcolor::red << "KILL LANDMARK\n" << termcolor::reset
					<< c.landmarksResetProb_[i] << " n "
					<< c.landmarks_numDetections_[i] << " nfov:"
					<< c.landmarks_numFoV_[i] << "  expectedChange "
					<< expectedWeightChange << "\n";
					*/

			c.landmarks_numDetections_[i] = 0;

		}
	}

	//std::cout << "Death Change  " <<expectedWeightChange <<"\n";
	return expectedWeightChange;

}

inline double VectorGLMBSLAM2D::sampleDA(VectorGLMBComponent2D &c) {
	std::vector<double> P ;
	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif

	AssociationProbabilities probs;
	double expectedWeightChange = 0;

	std::fill(c.landmarksResetProb_.begin(), c.landmarksResetProb_.end(),
			std::log(1 - config.PE_) - std::log(config.PE_));
	std::fill(c.landmarksInitProb_.begin(), c.landmarksInitProb_.end(), 0.0);
	for (int k = minpose_; k < maxpose_; k++) {

		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
			probs.i.clear();
			probs.l.clear();
			double maxprob = -std::numeric_limits<double>::infinity();
			int maxprobi = 0;
			auto it = c.DA_bimap_[k].left.find(nz);
			double selectedProb;
			int selectedDA = -2;
			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}
			double maxlikelihood = -std::numeric_limits<double>::infinity();
			int maxlikelihoodi = 0;
			for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++) {
				double likelihood = c.DAProbs_[k][nz].l[a];

				if (maxlikelihood < likelihood) {
					maxlikelihood = likelihood;
					maxlikelihoodi = a;
				}
				if (c.DAProbs_[k][nz].i[a] == -2) {
					probs.i.push_back(c.DAProbs_[k][nz].i[a]);
					probs.l.push_back(c.DAProbs_[k][nz].l[a]);
					if (c.DAProbs_[k][nz].l[a] > maxprob) {
						maxprob = c.DAProbs_[k][nz].l[a];
						maxprobi = a;
					}
				} else if (c.DAProbs_[k][nz].i[a] == selectedDA) {
					probs.i.push_back(c.DAProbs_[k][nz].i[a]);
					if (c.landmarks_numDetections_[c.DAProbs_[k][nz].i[a]
							- c.landmarks_[0]->id()] == 1) {
						likelihood += std::log(config.PE_)
								- std::log(1 - config.PE_)
								+ (c.landmarks_numFoV_[c.DAProbs_[k][nz].i[a]
										- c.landmarks_[0]->id()])
										* std::log(1 - config.PD_);
						//std::cout <<" single detection: increase:  " << std::log(config.PE_)-std::log(1-config.PE_) + (c.landmarks_numFoV_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0]->id()])*std::log(1-config.PD_) <<"\n";
						probs.l.push_back(likelihood);
					} else {
						probs.l.push_back(c.DAProbs_[k][nz].l[a]);
					}
					if (likelihood > maxprob) {
						maxprob = likelihood;
						maxprobi = a;
					}
				} else {
					if (c.DA_bimap_[k].right.count(c.DAProbs_[k][nz].i[a])
							== 0) { // landmark is not already associated to another measurement
						probs.i.push_back(c.DAProbs_[k][nz].i[a]);
						if (c.landmarks_numDetections_[c.DAProbs_[k][nz].i[a]
								- c.landmarks_[0]->id()] == 0) {
							likelihood +=
									std::log(config.PE_)
											- std::log(1 - config.PE_)
											+ (c.landmarks_numFoV_[c.DAProbs_[k][nz].i[a]
													- c.landmarks_[0]->id()])
													* std::log(1 - config.PD_);
							//std::cout <<" 0 detection: increase:  " << std::log(config.PE_)-std::log(1-config.PE_) + (c.landmarks_numFoV_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0]->id()])*std::log(1-config.PD_)<<"\n";
							probs.l.push_back(likelihood);
						} else {
							probs.l.push_back(c.DAProbs_[k][nz].l[a]);
						}
						if (likelihood > maxprob) {
							maxprob = likelihood;
							maxprobi = a;
						}
					}
				}
				if (c.DAProbs_[k][nz].i[a] == selectedDA) {
					expectedWeightChange -= probs.l[probs.l.size() - 1];
				}
			}

			P.resize(probs.l.size());
			double alternativeprob = 0;
			for (int i = 0; i < P.size(); i++) {

				P[i] = exp((probs.l[i] - maxprob));// /temp_);

				//std::cout << p << "   ";
				alternativeprob += P[i];
			}

			size_t sample = GibbsSampler::sample(randomGenerators_[threadnum],
					P);

			//alternativeprob=(alternativeprob -P[sample])/alternativeprob;
			if (alternativeprob < 1) {
				std::cout << P[maxprobi]
						<< " panicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanic  \n";
			}

			expectedWeightChange += probs.l[sample];
			if (probs.i[sample] >= 0) {
				//c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0]->id()] *= (P[ P.size()-1] )/P[sample];
				/*
				 if(probs.i[sample] != c.DAProbs_[k][nz].i[maxprobi]){
				 c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0]->id()] +=  c.DAProbs_[k][nz].l[maxprobi] - probs.l[sample]; //(1 )/alternativeprob;

				 }else{
				 c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0]->id()] += probs.l[probs.l.size()-1] - probs.l[sample] ;
				 }*/

				c.landmarksResetProb_[probs.i[sample] - c.landmarks_[0]->id()] +=
						std::log(P[sample] / alternativeprob);

			} else {

				if (c.DAProbs_[k][nz].i[maxlikelihoodi] >= 0) {
					//std::cout << "increasing init prob of lm " <<c.DAProbs_[k][nz].i[maxlikelihoodi] << "  by " <<maxlikelihood  << "- " << probs.l[sample]<< "\n";
					c.landmarksInitProb_[c.DAProbs_[k][nz].i[maxlikelihoodi]
							- c.landmarks_[0]->id()] += 1;
				}
			}

			if (probs.i[sample] != selectedDA) { // if selected association, change bimap

				if (probs.i[sample] >= 0) {
					c.landmarks_numDetections_[probs.i[sample]
							- c.landmarks_[0]->id()]++;
					if (selectedDA < 0) {
						c.DA_bimap_[k].insert( { nz, probs.i[sample] });
					} else {

						c.landmarks_numDetections_[selectedDA
								- c.landmarks_[0]->id()]--;
						c.DA_bimap_[k].left.replace_data(it, probs.i[sample]);

						// add an log for possible landmark merge
						c.tomerge_.push_back(
								std::make_pair(probs.i[sample], selectedDA));
					}
				} else { // if a change has to be made and new DA is false alarm, we need to remove the association
					c.DA_bimap_[k].left.erase(it);
					c.landmarks_numDetections_[selectedDA
							- c.landmarks_[0]->id()]--;

				}

			}

		}
	}

	return expectedWeightChange;
}

inline void VectorGLMBSLAM2D::updateFoV(VectorGLMBComponent2D &c) {
	c.landmarks_numFoV_.resize(c.landmarks_.size());
	std::fill(c.landmarks_numFoV_.begin(), c.landmarks_numFoV_.end(), 0);
	for (int k = 0; k < maxpose_; k++) {
		c.fov_[k].clear();
		if (c.Z_[k].size() > 0) { // if no measurements we set FoV to empty ,
			for (int lm = 0; lm < c.landmarks_.size(); lm++) {
				if (distance(c.poses_[k], c.landmarks_[lm])
						<= config.maxRange_) {
					double bearing = angle(c.poses_[k], c.landmarks_[lm]);
					if (abs(bearing) < 0.75 *M_PI){
						c.fov_[k].push_back(c.landmarks_[lm]->id());
						c.landmarks_numFoV_[lm]++;
					}
				}
			}
		}
	}
}

inline void VectorGLMBSLAM2D::updateDAProbs(VectorGLMBComponent2D &c, int minpose, int maxpose) {

	g2o::JacobianWorkspace jac_ws;
	MeasurementEdge z;
	jac_ws.updateSize(2, 2 * 3);
	jac_ws.allocate();

	for (int k = minpose; k < maxpose; k++) {
		c.DAProbs_[k].resize(c.Z_[k].size());

		double posHLogDet;
		if (!c.poses_[k]->fixed()) {
			posHLogDet = std::log(c.poses_[k]->hessianDeterminant());
		}
		PoseType::HessianBlockType poseHessian(c.poses_[k]->hessianData());

		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {

			// setting the topology of DAProbs to include all measurements in current FoV
			c.DAProbs_[k][nz].i = c.fov_[k];
			c.DAProbs_[k][nz].i.push_back(-2); // add posibility of false alarm
			c.DAProbs_[k][nz].l.resize(c.DAProbs_[k][nz].i.size());

			auto it = c.DA_bimap_[k].left.find(nz);
			int selectedDA = -2;
			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}
			Eigen::Matrix<double, PoseType::HessianBlockType::RowsAtCompileTime,
					PoseType::HessianBlockType::ColsAtCompileTime> poseHessianCopy =
					poseHessian;
			if (selectedDA >= 0) {
				c.Z_[k][nz]->g2o::BaseBinaryEdge<2, g2o::Vector2,
						g2o::VertexSE2, g2o::VertexPointXY>::linearizeOplus(
						jac_ws);
				MeasurementEdge::JacobianXiOplusType Jpose =
						c.Z_[k][nz]->jacobianOplusXi();
				poseHessianCopy -= Jpose.transpose()
						* c.Z_[k][nz]->information() * Jpose;
			}
			for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++) {
				c.DAProbs_[k][nz].l[a] = 0;
				if (c.DAProbs_[k][nz].i[a] == -2) { // set measurement to false alarm
					c.DAProbs_[k][nz].l[a] = config.logKappa_;
				} else {


					c.DAProbs_[k][nz].l[a] += std::log(config.PD_)
							- std::log(1 - config.PD_);
					c.Z_[k][nz]->setVertex(1,
							dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
									c.DAProbs_[k][nz].i[a])->second));

					c.Z_[k][nz]->g2o::BaseBinaryEdge<2, g2o::Vector2,
							g2o::VertexSE2, g2o::VertexPointXY>::linearizeOplus(
							jac_ws);
					c.Z_[k][nz]->computeError();

					Eigen::Matrix<double, MeasurementEdge::Dimension, 1,
							Eigen::ColMajor> omega_r = -c.Z_[k][nz]->error();

					// if pose is not fixed, calc updated pose and lm
					if (!c.poses_[k]->fixed()) {
						PointType::HessianBlockType::PlainMatrix h;
						PointType::HessianBlockType pointHessian(h.data());

						if (c.landmarks_numDetections_[c.DAProbs_[k][nz].i[a]
								- c.landmarks_[0]->id()] > 0) {
							new (&pointHessian) PointType::HessianBlockType(
									c.landmarks_[c.DAProbs_[k][nz].i[a]
											- c.landmarks_[0]->id()]->hessianData());

							//std::cout << "g2o pointH: " << pointHessian << "\n\n\n";
						} else {
							h = config.anchorInfo_;
							//std::cout << "calc pointH: " << pointHessian << "\n\n\n";

						}
						//std::cout << "numdetections:  " << c.landmarks_numDetections_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0]->id()]  << "\n";

						MeasurementEdge::JacobianXiOplusType Jpose =
								c.Z_[k][nz]->jacobianOplusXi();
						MeasurementEdge::JacobianXjOplusType Jpoint =
								c.Z_[k][nz]->jacobianOplusXj();

						Eigen::Matrix<double,
								PoseType::Dimension + PointType::Dimension,
								PoseType::Dimension + PointType::Dimension> H;
						Eigen::Matrix<double,
								PoseType::Dimension + PointType::Dimension, 1>
								b, sol;
						H.setZero();

						H.block(0, 0, PoseType::Dimension, PoseType::Dimension) =
								poseHessianCopy;
						H.block(PoseType::Dimension, PoseType::Dimension,
								PointType::Dimension, PointType::Dimension) =
								pointHessian;

						H.block(0, 0, PoseType::Dimension, PoseType::Dimension) +=
								Jpose.transpose() * c.Z_[k][nz]->information()
										* Jpose;
						H.block(PoseType::Dimension, PoseType::Dimension,
								PointType::Dimension, PointType::Dimension) +=
								Jpoint.transpose() * c.Z_[k][nz]->information()
										* Jpoint;

						H.block(PoseType::Dimension, 0, PointType::Dimension,
								PoseType::Dimension) = Jpoint.transpose()
								* c.Z_[k][nz]->information() * Jpose;
						H.block(0, PoseType::Dimension, PoseType::Dimension,
								PointType::Dimension) = H.block(
								PoseType::Dimension, 0, PointType::Dimension,
								PoseType::Dimension).transpose();
						b.block(0, 0, PoseType::Dimension, 1) =
								Jpose.transpose() * omega_r;
						b.block(PoseType::Dimension, 0, PointType::Dimension, 1) =
								Jpoint.transpose() * omega_r;

						Eigen::LLT<
								Eigen::Matrix<double,
										PoseType::Dimension
												+ PointType::Dimension,
										PoseType::Dimension
												+ PointType::Dimension>> lltofH(
								H);
						sol = lltofH.solve(b);

						c.DAProbs_[k][nz].l[a] += std::log(
								poseHessianCopy.determinant())
								+ std::log(pointHessian.determinant())
								- std::log(lltofH.matrixL().determinant());
						c.DAProbs_[k][nz].l[a] += std::log(
								c.Z_[k][nz]->information().determinant())
								+ posHLogDet;

						c.DAProbs_[k][nz].l[a] += -0.5
								* (c.Z_[k][nz]->chi2() - sol.dot(b));
						c.DAProbs_[k][nz].l[a] += -0.5
								* c.Z_[k][nz]->dimension() * std::log(2 * M_PI);
					} else { // if pose is fixed only calculate updated landmark
						PointType::HessianBlockType pointHessian(
								c.landmarks_[c.DAProbs_[k][nz].i[a]
										- c.landmarks_[0]->id()]->hessianData());

						MeasurementEdge::JacobianXjOplusType Jpoint =
								c.Z_[k][nz]->jacobianOplusXj();

						Eigen::Matrix<double, PointType::Dimension,
								PointType::Dimension> H;
						H.setZero();
						H = pointHessian
								+ Jpoint.transpose()
										* c.Z_[k][nz]->information() * Jpoint;
						Eigen::Matrix<double, PointType::Dimension, 1> b, sol;
						b = Jpoint.transpose() * omega_r;

						Eigen::LLT<
								Eigen::Matrix<double, PointType::Dimension,
										PointType::Dimension>> lltofH(H);
						sol = lltofH.solve(b);

						c.DAProbs_[k][nz].l[a] += -std::log(
								lltofH.matrixL().determinant());
						c.DAProbs_[k][nz].l[a] += std::log(
								c.Z_[k][nz]->information().determinant());

						//c.DAProbs_[k][nz].l[a] += -0.5 * (c.Z_[k][nz]->chi2() - sol.dot(b));
						c.DAProbs_[k][nz].l[a] += -0.5 * (c.Z_[k][nz]->chi2());
						c.DAProbs_[k][nz].l[a] += -0.5
								* c.Z_[k][nz]->dimension() * std::log(2 * M_PI);

					}
				}
			}
			if (selectedDA >= 0) {
				c.Z_[k][nz]->setVertex(1,
						dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
								selectedDA)->second));
				c.Z_[k][nz]->linearizeOplus();
				c.Z_[k][nz]->computeError();
			} else {
				c.Z_[k][nz]->setVertex(1, NULL);

			}

		}

	}

}

inline void VectorGLMBSLAM2D::constructGraph(VectorGLMBComponent2D &c) {

	c.numPoses_ = 0;
	c.numPoints_ = 0;
	//Copy Vertices from optimizer with data association
	int maxid = 0;
	for (auto pair : gt_graph.optimizer_->vertices()) {
		g2o::HyperGraph::Vertex *v = pair.second;
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

			if (maxid < pose->id()) {
				maxid = pose->id();
			}
		}
		//sort by id

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
	std::sort(c.poses_.begin(), c.poses_.end(),
			[](const auto &lhs, const auto &rhs) {
				return lhs->id() < rhs->id();
			});

	int lmid = maxid + 1;
	for (double x = config.xlim_[0]; x <= config.xlim_[1];
			x += (config.xlim_[1] - config.xlim_[0]) / config.numLandmarks_) {
		for (double y = config.ylim_[0]; y <= config.ylim_[1];
				y += (config.ylim_[1] - config.ylim_[0])
						/ config.numLandmarks_) {
			PointType *lm = new PointType();
			PointAnchorEdge *anchor = new PointAnchorEdge();
			Eigen::Vector2d xy(x, y);
			lm->setEstimateData(xy.data());
			lm->setId(lmid++);
			c.optimizer_->addVertex(lm);
			c.landmarks_.push_back(lm);

			anchor->setVertex(0, lm);
			anchor->setMeasurement(xy);
			anchor->setInformation(config.anchorInfo_);

			if (!c.optimizer_->addEdge(anchor)) {
				std::cerr << "anchor edge insert fail \n";
			}
		}

	}

	//Copy odometry measurements, Copy and save landmark measurements

	c.landmarks_numDetections_.resize(c.landmarks_.size(), 0);
	c.landmarksResetProb_.resize(c.landmarks_.size(), 0.0);
	c.landmarksInitProb_.resize(c.landmarks_.size(), 0.0);
	c.DA_bimap_.resize(c.numPoses_);
	boost::bimap<int, int, boost::container::allocator<int>> empty_bimap ;
	for(int p=0;p<c.numPoses_;p++){
		c.DA_bimap_[p] = empty_bimap;
	}
	c.prevDA_bimap_ = c.DA_bimap_;

	c.Z_.resize(c.numPoses_);
	c.DAProbs_.resize(c.numPoses_);
	c.fov_.resize(c.numPoses_);

	for (g2o::HyperGraph::Edge *e : gt_graph.optimizer_->edges()) {
		OdometryEdge *odo = dynamic_cast<OdometryEdge*>(e);
		if (odo != NULL) {
			OdometryEdge *odocopy = new OdometryEdge();
			int firstvertex = odo->vertex(0)->id();
			odocopy->setVertex(0,
					dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
							firstvertex)->second));
			int secondvertex = odo->vertex(1)->id();
			odocopy->setVertex(1,
					dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
							secondvertex)->second));
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
			zcopy->setVertex(0,
					dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
							firstvertex)->second));
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
	temp_ = config.initTemp_;
	auto linearSolver = g2o::make_unique<SlamLinearSolver>();
	linearSolver->setBlockOrdering(false);
	c.linearSolver_ = linearSolver.get();
	auto blockSolver = g2o::make_unique<SlamBlockSolver>(
			std::move(linearSolver));
	c.blockSolver_ = blockSolver.get();
	c.solverLevenberg_ = new g2o::OptimizationAlgorithmLevenberg(
			std::move(blockSolver));

	c.optimizer_ = new g2o::SparseOptimizer();
	c.optimizer_->setAlgorithm(c.solverLevenberg_);

}

}
#endif
