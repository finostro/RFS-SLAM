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

	std::vector<boost::bimap<int, int>> DA_bimap_, prevDA_bimap_; /**< Bimap containing data association hypothesis at time k  */

	std::vector<std::vector<MeasurementEdge*> > Z_; /**< Measurement edges stored, in order to set data association and add to graph later */
	std::vector<std::vector<AssociationProbabilities> > DAProbs_; /**< DAProbs_ [k][nz] are is the association probabilities of measurement
	 nz at time k, used for switching using gibbs sampling*/
	std::vector<std::vector<int> > fov_; /**< indices of landmarks in field of view at time k */

	std::vector<PoseType*> poses_;
	std::vector<PointType*> landmarks_;
	std::vector<int> landmarks_numDetections_;

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
	typedef g2o::EdgeXYPrior PointAnchorEdge;
	typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> > SlamBlockSolver;
	typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	/**
	 * \brief Configurations for this RFSBatchPSO optimizer
	 */
	struct Config {
	public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
		int numIterations_; /**< number of iterations of main algorithm */
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
	double sampleDA(VectorGLMBComponent2D &c);

    /**
     * Revert the current data association to keep the last one
     * @param c the GLMB component
     */
    double revertDA(VectorGLMBComponent2D &c);

    /**
     * print the data association in component c
     * @param c the GLMB component
     */
    void printDA(VectorGLMBComponent2D &c,std::ostream &s = std::cout);
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

	int nThreads_; /**< Number of threads  */

	VectorGLMBComponent2D gt_graph;

	std::vector<VectorGLMBComponent2D> components_; /**< VGLMB components */

	int iteration_=0;

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
    config.numIterations_ = node["numIterations"].as<int>();
	config.numLevenbergIterations_ = node["numLevenbergIterations"].as<int>();
	config.xlim_.push_back(node["xlim"][0].as<double>()) ;
	config.xlim_.push_back(node["xlim"][1].as<double>()) ;
	config.ylim_.push_back(node["ylim"][0].as<double>()) ;
	config.ylim_.push_back(node["ylim"][1].as<double>()) ;

	config.finalStateFile_ =  node["finalStateFile"].as<std::string>();


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

		// optimize once at the start to calculate the hessian.
		c.poses_[0]->setFixed(true);
		c.optimizer_->initializeOptimization(c.optimizer_->edges());
		//c.optimizer_->computeInitialGuess();
		c.optimizer_->setVerbose(true);
		std::cout <<"niterations  " <<c.optimizer_->optimize(1) << "\n";
	}

}
inline void VectorGLMBSLAM2D::run(int numSteps) {
    for( int i =0; i < numSteps; i++){
        optimize(config.numLevenbergIterations_);
    }
}
inline void VectorGLMBSLAM2D::optimize(int ni) {
	for (auto &c : components_) {
		updateFoV(c);
		updateDAProbs(c);
		c.prevDA_bimap_ = c.DA_bimap_;
		double expectedChange=0;
		for(int i=0; i< config.numGibbs_ ; i++){
			expectedChange += sampleDA(c);
		}
        //printFoV(c);
		std::ofstream dafile;
		std::stringstream filename;
		filename << "DA__" << iteration_ << ".txt";
		dafile.open(filename.str());
		std::cout<<" iteraton " <<iteration_++  << " :::: \n";
        printDA(c,dafile);
        printDAProbs(c);
		updateGraph(c);
		c.poses_[0]->setFixed(true);
		c.optimizer_->initializeOptimization(c.optimizer_->edges());
		//c.optimizer_->computeInitialGuess();
		c.optimizer_->setVerbose(false);
		std::cout <<"niterations  " <<c.optimizer_->optimize(ni) << "\n";
		calculateWeight(c);

		double accept = std::min(1.0 ,  std::exp(c.logweight_-c.prevLogWeight_ - expectedChange));
	    int threadnum = 0;
	#ifdef _OPENMP
	threadnum = omp_get_thread_num();
	#endif
	boost::uniform_real<> uni_dist(0, 1);

	uni_dist(rfs::randomGenerators_[threadnum]);

		std::cout << "accept: " << accept << "\n";
		std::cout << "weight: " << c.logweight_ << " prevWeight: " << c.prevLogWeight_ << " expectedChange " << expectedChange << "   chi2:  " <<c.optimizer_->chi2() << "  determinant: " << c.linearSolver_->_determinant<< "\n";

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
		    int selectedDA = -2;
			auto it = c.DA_bimap_[k].left.find(nz);

			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}
			int previd = c.Z_[k][nz]->vertex(1)? c.Z_[k][nz]->vertex(1)->id():-2; /**< previous data association */
			if(previd == selectedDA){
				continue;
			}
			if(selectedDA>=0){


				// if edge was already in graph, modify it
				 if(previd>=0){
					 c.optimizer_->setEdgeVertex(c.Z_[k][nz] , 1 , dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(selectedDA)->second)); // this removes the edge from the list in both vertices
				 }else{
				     c.Z_[k][nz]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(selectedDA)->second));

				     c.optimizer_->addEdge(c.Z_[k][nz]);
				 }

			}else{

				c.optimizer_->removeEdge(c.Z_[k][nz]);
				c.Z_[k][nz]->setVertex(1,NULL);

			}

		}
	}
}
template< class MapType >
void print_map(const MapType & m, std::ostream &s = std::cout)
{
    typedef typename MapType::const_iterator const_iterator;
    for( const_iterator iter = m.begin(), iend = m.end(); iter != iend; ++iter )
    {
        s << iter->first << "-->" << iter->second << std::endl;
    }
}
inline void VectorGLMBSLAM2D::printFoV(VectorGLMBComponent2D &c) {
    std::cout << "FoV:\n";
    for(int k=0; k< c.fov_.size() ; k++){
        std::cout <<  k << "  FoV at:   ";
        for(int lmid:c.fov_[k]){
            std::cout <<"  ,  "<< lmid ;
        }
        std::cout << "\n";
    }
}
inline void VectorGLMBSLAM2D::printDAProbs(VectorGLMBComponent2D &c) {
    for(int k=0; k< c.DAProbs_.size() ; k++){
    	if (k==2) break;
        std::cout << k << "da probs:\n";
        for(int nz=0; nz < c.DAProbs_[k].size(); nz++){
            std::cout <<"z =  "<< nz << "  ;";
            for(double l:c.DAProbs_[k][nz].l){
                std::cout<< std::max(l,-100.0) << " , ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
inline void VectorGLMBSLAM2D::printDA(VectorGLMBComponent2D &c, std::ostream &s) {
    for(int k=0; k< c.DA_bimap_.size() ; k++){
        s << k << ":\n";
        print_map(c.DA_bimap_[k].left,s);
    }
}


inline double VectorGLMBSLAM2D::revertDA(VectorGLMBComponent2D &c) {
    c.logweight_ = c.prevLogWeight_;
    c.DA_bimap_  = c.prevDA_bimap_;
    updateGraph(c);
}

inline double VectorGLMBSLAM2D::sampleDA(VectorGLMBComponent2D &c) {
	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif

	AssociationProbabilities probs;
	double expectedWeightChange = 0;
	for (int k = 0; k < c.DAProbs_.size(); k++) {

		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
			probs.i.clear();
			probs.l.clear();
			double maxprob = -std::numeric_limits<double>::infinity();
			auto it = c.DA_bimap_[k].left.find(nz);
			double selectedProb;
			int selectedDA = -2;
			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}
			for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++) {
				double likelihood =c.DAProbs_[k][nz].l[a];
				if (c.DAProbs_[k][nz].i[a] == -2) {
					probs.i.push_back(c.DAProbs_[k][nz].i[a]);
					probs.l.push_back(c.DAProbs_[k][nz].l[a]);
					if (c.DAProbs_[k][nz].l[a] > maxprob)
						maxprob = c.DAProbs_[k][nz].l[a];
				}else if (c.DAProbs_[k][nz].i[a] == selectedDA ) {
					probs.i.push_back(c.DAProbs_[k][nz].i[a]);
					if(c.landmarks_numDetections_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0]->id()] == 1){
						likelihood += std::log(config.PE_)-std::log(1-config.PE_);
						probs.l.push_back(likelihood);
					}else{
						probs.l.push_back(c.DAProbs_[k][nz].l[a]);
					}
					if (c.DAProbs_[k][nz].l[a] > maxprob)
						maxprob = c.DAProbs_[k][nz].l[a];
				}else  {
					if (c.DA_bimap_[k].right.count(c.DAProbs_[k][nz].i[a]) == 0) {  // landmark is not already associated to another measurement
						probs.i.push_back(c.DAProbs_[k][nz].i[a]);
						if(c.landmarks_numDetections_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0]->id()] == 0){
							likelihood += std::log(config.PE_)-std::log(1-config.PE_);
							probs.l.push_back(likelihood);
						}else{
							probs.l.push_back(c.DAProbs_[k][nz].l[a]);
						}
						if (c.DAProbs_[k][nz].l[a] > maxprob)
							maxprob = c.DAProbs_[k][nz].l[a];
					}
				}
				if(c.DAProbs_[k][nz].i[a] == selectedDA ){
				    expectedWeightChange -= probs.l[probs.l.size()-1];
				}
				}

			auto P= probs.l;
			for (auto &p : P) {
				p = std::exp(p - maxprob);
			}
			size_t sample = GibbsSampler::sample(randomGenerators_[threadnum], P);
			expectedWeightChange+= probs.l[sample];
			if (probs.i[sample] != selectedDA) { // if selected association, change bimap

				if (probs.i[sample] >= 0) {
					c.landmarks_numDetections_[probs.i[sample]-c.landmarks_[0]->id()]++;
					if (selectedDA < 0) {
						c.DA_bimap_[k].insert( { nz, probs.i[sample] });
					} else {
						c.landmarks_numDetections_[selectedDA-c.landmarks_[0]->id()]--;
						c.DA_bimap_[k].left.replace_data(it, probs.i[sample]);
					}
				} else { // if a change has to be made and new DA is false alarm, we need to remove the association
					c.DA_bimap_[k].left.erase(it);
					c.landmarks_numDetections_[selectedDA-c.landmarks_[0]->id()]--;

				}

			}

		}
	}

	return expectedWeightChange;
}

inline void VectorGLMBSLAM2D::updateFoV(VectorGLMBComponent2D &c) {
    for (int k = 0; k < c.fov_.size(); k++) {
        c.fov_[k].clear();
        if (c.Z_[k].size() > 0) { // if no measurements we set FoV to empty ,
            for (auto lm : c.landmarks_) {
                if (distance(c.poses_[k], lm) <= config.maxRange_) {
                    c.fov_[k].push_back(lm->id());
                }
            }
        }
    }
}

inline void VectorGLMBSLAM2D::updateDAProbs(VectorGLMBComponent2D &c) {

    g2o::JacobianWorkspace  jac_ws;
    MeasurementEdge z;
    jac_ws.updateSize(2,2*3);
    jac_ws.allocate();

	for (int k = 0; k < c.DAProbs_.size(); k++) {
	    c.DAProbs_[k].resize(c.Z_[k].size());

		double posHLogDet ;
		if(!c.poses_[k]->fixed()){
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
			Eigen::Matrix<double ,PoseType::HessianBlockType::RowsAtCompileTime ,PoseType::HessianBlockType::ColsAtCompileTime > poseHessianCopy = poseHessian;
			if (selectedDA>=0){
                c.Z_[k][nz]->g2o::BaseBinaryEdge<2, g2o::Vector2, g2o::VertexSE2, g2o::VertexPointXY>::linearizeOplus(jac_ws);
                MeasurementEdge::JacobianXiOplusType Jpose = c.Z_[k][nz]->jacobianOplusXi();
				poseHessianCopy -=  Jpose.transpose() * c.Z_[k][nz]->information() * Jpose;
			}
            for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++) {
                c.DAProbs_[k][nz].l[a] =0;
                if (c.DAProbs_[k][nz].i[a] == -2) { // set measurement to false alarm
                    c.DAProbs_[k][nz].l[a] = config.logKappa_ ;
                } else {


                    c.DAProbs_[k][nz].l[a] += std::log(config.PD_) - std::log(1 - config.PD_);
                    c.Z_[k][nz]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(c.DAProbs_[k][nz].i[a])->second));

                    c.Z_[k][nz]->g2o::BaseBinaryEdge<2, g2o::Vector2, g2o::VertexSE2, g2o::VertexPointXY>::linearizeOplus(jac_ws);
                    c.Z_[k][nz]->computeError();

                    // if pose is not fixed, calc updated pose and lm
                    if (!c.poses_[k]->fixed()) {
                        PointType::HessianBlockType pointHessian(c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0]->id()]->hessianData());

                        MeasurementEdge::JacobianXiOplusType Jpose = c.Z_[k][nz]->jacobianOplusXi();
                        MeasurementEdge::JacobianXjOplusType Jpoint = c.Z_[k][nz]->jacobianOplusXj();

                        Eigen::Matrix<double, PoseType::Dimension + PointType::Dimension, PoseType::Dimension + PointType::Dimension> H;
                        H.setZero();

                        H.block(0, 0, PoseType::Dimension, PoseType::Dimension) = poseHessianCopy + Jpose.transpose() * c.Z_[k][nz]->information() * Jpose;
                        H.block(PoseType::Dimension, PoseType::Dimension, PointType::Dimension, PointType::Dimension) = pointHessian + Jpoint.transpose() * c.Z_[k][nz]->information() * Jpoint;

                        H.block(PoseType::Dimension, 0, PointType::Dimension, PoseType::Dimension) = Jpoint.transpose() * c.Z_[k][nz]->information() * Jpose;
                        H.block(0, PoseType::Dimension, PoseType::Dimension, PointType::Dimension) = H.block(PoseType::Dimension, 0, PointType::Dimension, PoseType::Dimension).transpose() ;
                        Eigen::Matrix<double, PoseType::Dimension + PointType::Dimension, 1> b, sol;
                        b.block(0, 0, PoseType::Dimension, 1) = Jpose.transpose() * c.Z_[k][nz]->error();
                        b.block(PoseType::Dimension, 0, PointType::Dimension, 1) = Jpoint.transpose() * c.Z_[k][nz]->error();

                        Eigen::LLT<Eigen::Matrix<double, PoseType::Dimension + PointType::Dimension, PoseType::Dimension + PointType::Dimension>> lltofH(H);
                        sol = lltofH.solve(b);

                        c.DAProbs_[k][nz].l[a] += -std::log(lltofH.matrixL().determinant());
                        c.DAProbs_[k][nz].l[a] += std::log(c.Z_[k][nz]->information().determinant()) + posHLogDet;

                        c.DAProbs_[k][nz].l[a] += -0.5 * (c.Z_[k][nz]->chi2() - sol.dot(b));
                        c.DAProbs_[k][nz].l[a] += -0.5 * c.Z_[k][nz]->dimension() * std::log(2 * M_PI);
                    } else { // if pose is fixed only calculate updated landmark
                        PointType::HessianBlockType pointHessian(c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0]->id()]->hessianData());

                        MeasurementEdge::JacobianXjOplusType Jpoint = c.Z_[k][nz]->jacobianOplusXj();

                        Eigen::Matrix<double, PointType::Dimension, PointType::Dimension> H;
                        H.setZero();
                        H = pointHessian + Jpoint.transpose() * c.Z_[k][nz]->information() * Jpoint;
                        Eigen::Matrix<double, PointType::Dimension, 1> b, sol;
                        b = Jpoint.transpose() * c.Z_[k][nz]->error();

                        Eigen::LLT<Eigen::Matrix<double,  PointType::Dimension,  PointType::Dimension>> lltofH(H);
                        sol = lltofH.solve(b);

                        c.DAProbs_[k][nz].l[a] += -std::log(lltofH.matrixL().determinant());
                        c.DAProbs_[k][nz].l[a] += std::log(c.Z_[k][nz]->information().determinant());

                        c.DAProbs_[k][nz].l[a] += -0.5 * (c.Z_[k][nz]->chi2() - sol.dot(b));
                        c.DAProbs_[k][nz].l[a] += -0.5 * c.Z_[k][nz]->dimension() * std::log(2 * M_PI);

                    }
                }
            }
			if(selectedDA>=0){
				c.Z_[k][nz]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(selectedDA)->second));
				c.Z_[k][nz]->linearizeOplus();
				c.Z_[k][nz]->computeError();
			}else{
				c.Z_[k][nz]->setVertex(1, NULL);

			}

		}

	}

}

inline void VectorGLMBSLAM2D::constructGraph(VectorGLMBComponent2D &c) {

	c.numPoses_ = 0;
	c.numPoints_ = 0;
	//Copy Vertices from optimizer with data association
	int maxid=0;
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

			if (maxid <pose->id()){
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
    std::sort(c.poses_.begin(),c.poses_.end(),  [] (const auto& lhs, const auto& rhs) {
        return lhs->id() < rhs->id();
    } );

	int lmid = maxid+1;
	for (double x = config.xlim_[0]; x <= config.xlim_[1]; x += (config.xlim_[1] - config.xlim_[0]) / config.numLandmarks_) {
		for (double y = config.ylim_[0]; y <= config.ylim_[1]; y += (config.ylim_[1] - config.ylim_[0]) / config.numLandmarks_) {
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

			if(!c.optimizer_->addEdge(anchor)){
				std::cerr << "anchor edge insert fail \n";
			}
		}

	}

	//Copy odometry measurements, Copy and save landmark measurements

	c.landmarks_numDetections_.resize(c.landmarks_.size(),0);
	c.DA_bimap_.resize(c.numPoses_);

	c.Z_.resize(c.numPoses_);
	c.DAProbs_.resize(c.numPoses_);
	c.fov_.resize(c.numPoses_);

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
	auto blockSolver =  g2o::make_unique<SlamBlockSolver>(std::move(linearSolver));
	c.blockSolver_ = blockSolver.get();
	c.solverLevenberg_ = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

	c.optimizer_ =  new g2o::SparseOptimizer();
	c.optimizer_->setAlgorithm(c.solverLevenberg_);

}

}
#endif
