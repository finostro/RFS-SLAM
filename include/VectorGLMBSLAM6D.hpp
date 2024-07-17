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
#ifndef VECTORGLMBSLAM6D_HPP
#define VECTORGLMBSLAM6D_HPP

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
#include "OrbslamMapPoint.hpp"
#include "OrbslamPose.hpp"
#include "VectorGLMBComponent6D.hpp"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/core/robust_kernel_impl.h"

#include "g2o/types/sba/types_six_dof_expmap.h" // se3 poses

#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>

#include <boost/bimap.hpp>
#include <boost/container/allocator.hpp>
#include <yaml-cpp/yaml.h>

#include "misc/EigenYamlSerialization.hpp"
#include <misc/termcolor.hpp>

#include <external/ORBextractor.h>
#include <external/Converter.h>

#include<opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "sophus/se3.hpp"

#include <Visualizer6D.hpp>

#ifdef _PERFTOOLS_CPU
#include <gperftools/profiler.h>
#endif
#ifdef _PERFTOOLS_HEAP
#include <gperftools/heap-profiler.h>
#endif

namespace rfs {

static const int orb_th_low = 50;
static const int orb_th_high = 100;

struct bimap_less {
	bool operator()(
			const boost::bimap<int, int, boost::container::allocator<int>> x,
			const boost::bimap<int, int, boost::container::allocator<int>> y) const {

		return x.left < y.left;
	}
};

// Computes the Hamming distance between two ORB descriptors
static int descriptorDistance(const cv::Mat &a, const cv::Mat &b) {
	const int *pa = a.ptr<int32_t>();
	const int *pb = b.ptr<int32_t>();

	int dist = 0;

	for (int i = 0; i < 8; i++, pa++, pb++) {
		unsigned int v = *pa ^ *pb;
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}

	return dist;
}

/**
 *  \class VectorGLMBSLAM6D
 *  \brief Random Finite Set  optimization using ceres solver for  feature based SLAM
 *
 *
 *  \author  Felipe Inostroza
 */
class VectorGLMBSLAM6D {
public:
	typedef g2o::VertexSBAPointXYZ PointType;
	typedef g2o::VertexSE3Expmap PoseType;
	typedef g2o::EdgeProjectXYZ2UV MonocularMeasurementEdge;
	typedef g2o::EdgeStereoSE3ProjectXYZ StereoMeasurementEdge;
	typedef g2o::EdgeSE3Expmap OdometryEdge;

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
		Eigen::Matrix3d anchorInfo_; /** information for anchor edges, should be low*/
		Eigen::Matrix3d stereoInfo_; /** information for stereo uvu edges */
		Eigen::Matrix<double , 6 , 6 > odomInfo_; /** information for odometry edges */

		std::string finalStateFile_;

		g2o::CameraParameters *g2o_cam_params;
		double viewingCosLimit_;

		std::string eurocFolder_, eurocTimestampsFilename_;

		bool 	use_gui_;

		// camera distortion params

		struct CameraParams {
			double fx;
			double fy;
			double cx;
			double cy;
			double k1;
			double k2;
			double p1;
			double p2;

			cv::Size originalImSize, newImSize;
			cv::Mat opencv_distort_coeffs, opencv_calibration;
			cv::Mat M1, M2;

			cv::Mat cv_c0_to_camera;

			Eigen::MatrixXd cv_c0_to_camera_eigen;
		};
		std::vector<CameraParams> camera_parameters_;

		double stereo_baseline;
		double stereo_baseline_f;
		double stereo_init_max_depth;

		Sophus::SE3<float> pose_left_to_right, pose_right_to_left;
		Eigen::Matrix<float, 3, 3> rotation_matrix_left_to_right;
		Eigen::Vector3f mtlr;
		/*
		 * Rectification stuff
		 */
		cv::Mat M1l_, M2l_;
		cv::Mat M1r_, M2r_;

		// ORB params
		struct ORBExtractor {
			int nFeatures;
			double scaleFactor;
			int nLevels;
			int iniThFAST;
			int minThFAST;

		} orb_extractor;

	} config;

	/**
	 * Constructor
	 */
	VectorGLMBSLAM6D();

	/** Destructor */
	~VectorGLMBSLAM6D();

	/**
	 *  Load a yaml style config file
	 * @param filename filename of the yaml config file
	 */
	void
	loadConfig(std::string filename);

	/**
	 *  Load imaged from an euroc dataset , folder set in config file
	 * 
	 */
	void
	loadEuroc();

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
	void selectNN(VectorGLMBComponent6D &c);
	/**
	 * Sample n data associations from the already visited group, in order to perform gibbs sampler on each.
	 * @param ni number of iterations of the optimizer
	 */
	void sampleComponents();
	/**
	 * Computes the stereo matches in the orb measurements stored in c
	 * @param pose an orbslam pose with keypoints and descriptors
	 */
	void computeStereoMatches(OrbslamPose &pose);

	/**
	 * Use the data association stored in DA_ to create the graph.
	 * @param c the GLMB component
	 */
	void constructGraph(VectorGLMBComponent6D &c);
	/**
	 *
	 * Initialize a VGLMB component , setting the data association to all false alarms
	 * @param c the GLMB component
	 */
	void init(VectorGLMBComponent6D &c);

	/**
	 * Calculate the probability of each measurement being associated with a specific landmark
	 * @param c the GLMB component
	 */
	void updateDAProbs(VectorGLMBComponent6D &c, int minpose, int maxpose);

	/**
	 * Calculate the FoV at each time
	 * @param c the GLMB component
	 */
	void updateFoV(VectorGLMBComponent6D &c);

	/**
	 * Use the probabilities calculated using updateDAProbs to sample a new data association through gibbs sampling
	 * @param c the GLMB component
	 */
	double sampleDA(VectorGLMBComponent6D &c);

	/**
	 * Merge two data associations into a third one, by selecting a random merge time.
	 */
	std::vector<boost::bimap<int, int, boost::container::allocator<int>> > sexyTime(
			VectorGLMBComponent6D &c1, VectorGLMBComponent6D &c2);

	/**
	 * Use the probabilities calculated in sampleDA to reset all the detections of a single landmark to all false alarms.
	 * @param c the GLMB component
	 */
	double sampleLMDeath(VectorGLMBComponent6D &c);

	/**
	 * Randomly merge landmarks in order to improve the sampling algorithm
	 * @param c the GLMB component
	 */
	double mergeLM(VectorGLMBComponent6D &c);

	/**
	 * Use the probabilities calculated in sampleDA to initialize landmarks from  false alarms.
	 * @param c the GLMB component
	 */
	double sampleLMBirth(VectorGLMBComponent6D &c);


	/**
	 * Change the data association in component c to the one stored on da.
	 */
	void changeDA(VectorGLMBComponent6D &c,
			const std::vector<
					boost::bimap<int, int, boost::container::allocator<int>> > &da);

	/**
	 * print the data association in component c
	 * @param c the GLMB component
	 */
	void printDA(VectorGLMBComponent6D &c, std::ostream &s = std::cout);
	/**
	 * print the data association in component c
	 * @param c the GLMB component
	 */
	void printDAProbs(VectorGLMBComponent6D &c);
	/**
	 * print the data association in component c
	 * @param c the GLMB component
	 */
	void printFoV(VectorGLMBComponent6D &c);
	/**
	 * Use the data association hipothesis and the optimized state to calculate the component weight.
	 * @param c the GLMB component
	 */
	void calculateWeight(VectorGLMBComponent6D &c);
	/**
	 * Use the new sampled data association to update the g2o graph
	 * @param c the GLMB component
	 */
	void updateGraph(VectorGLMBComponent6D &c);

	/**
	 * @brief waits for the gui to exit 
	 * 
	 */
	void waitForGuiClose();

	/**
	 * Calculate the range between a pose and a landmark, to calculate the probability of detection.
	 * @param pose A 6D pose
	 * @param lm A 3D landmark
	 * @return the distance between pose and landmark
	 */
	static double distance(PoseType *pose, PointType *lm);

/**
 *  convert stereo measurement into xyz
 * @param measurement
 * @param trans_xyz
 */
    bool cam_unproject(const StereoMeasurementEdge &measurement,  Eigen::Vector3d &trans_xyz);

    /**
     * initialize a map point using a pose and stereo match
     * @param pose
     * @param numMatch
     * @param lm
     * @param newId
     * @return
     */
    bool initMapPoint(OrbslamPose &pose, int numMatch,  OrbslamMapPoint &lm, int newId);

	int nThreads_; /**< Number of threads  */

	VectorGLMBComponent6D initial_component_;

	std::vector<VectorGLMBComponent6D> components_; /**< VGLMB components */
	double bestWeight_ = -std::numeric_limits<double>::infinity();
	std::vector<boost::bimap<int, int, boost::container::allocator<int>> > best_DA_;
	int best_DA_max_detection_time_ = 0; /**< last association time */

	std::map<
			std::vector<boost::bimap<int, int, boost::container::allocator<int>> >,
			double> visited_;
	double temp_;
	int minpose_ = 0; /**< sample data association from this pose  onwards*/
	int maxpose_ = 0; /**< optimize only up to this pose */
	int maxpose_prev_ = 0;
	int iteration_ = 0;
	int iterationBest_ = 0;
	double insertionP_ = 0.5;

	//euroc dataset

	//filenames
	std::vector<std::string> vstrImageLeft;
	std::vector<std::string> vstrImageRight;
	//timestamps

	std::vector<double> vTimestampsCam;
	int nImages;

	ORB_SLAM3::ORBextractor *mpORBextractorLeft;
	ORB_SLAM3::ORBextractor *mpORBextractorRight;
	std::vector<float> orbExtractorInvScaleFactors;
	std::vector<float> orbExtractorScaleFactors;

	std::shared_ptr<Visualizer6D> visualizer;

};

//////////////////////////////// Implementation ////////////////////////





VectorGLMBSLAM6D::VectorGLMBSLAM6D() {
	nThreads_ = 1;

#ifdef _OPENMP
      nThreads_ = omp_get_max_threads();
#endif

}

VectorGLMBSLAM6D::~VectorGLMBSLAM6D() {

}

void VectorGLMBSLAM6D::changeDA(VectorGLMBComponent6D &c,
		const std::vector<
				boost::bimap<int, int, boost::container::allocator<int>> > &da) {
	// update bimaps!!!
	c.DA_bimap_ = da;

	// update the numdetections
	for (auto &mp : c.landmarks_) {
		mp.numDetections_ = 0;
	}
	for (auto &bimap : da) {
		for (auto it = bimap.begin(), it_end = bimap.end(); it != it_end;
				it++) {
			c.landmarks_[it->right - c.landmarks_[0].pPoint->id()].numDetections_++;
		}
	}
	updateGraph(c);
	c.poses_[0].pPose->setFixed(true);
	c.optimizer_->initializeOptimization();
	//c.optimizer_->computeInitialGuess();
	c.optimizer_->setVerbose(false);
	c.optimizer_->optimize(config.numLevenbergIterations_);

}
void VectorGLMBSLAM6D::sampleComponents() {

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

void VectorGLMBSLAM6D::loadEuroc() {
	std::string pathCam0 = config.eurocFolder_ + "/mav0/cam0/data";
	std::string pathCam1 = config.eurocFolder_ + "/mav0/cam1/data";

//Loading image filenames and timestamps
	std::ifstream fTimes;
	fTimes.open(config.eurocTimestampsFilename_.c_str());
	vTimestampsCam.reserve(5000);
	vstrImageLeft.reserve(5000);
	vstrImageRight.reserve(5000);
	while (!fTimes.eof()) {

		std::string s;
		std::getline(fTimes, s);
		if (!s.empty()) {
			std::stringstream ss;
			ss << s;
			vstrImageLeft.push_back(pathCam0 + "/" + ss.str() + ".png");
			vstrImageRight.push_back(pathCam1 + "/" + ss.str() + ".png");
			double t;
			ss >> t;
			vTimestampsCam.push_back(t / 1e9);

		}
	}
	nImages = vstrImageLeft.size();

	cv::Mat imLeft, imRight;
	cv::Mat imLeft_rect, imRight_rect;

	// Seq loop
	double t_resize = 0;
	double t_rect = 0;
	double t_track = 0;
	int num_rect = 0;

	initial_component_.poses_.resize(nImages);
	initial_component_.numPoses_= nImages;
	initial_component_.numPoints_ = 0;


	for (int ni = 0; ni < nImages; ni++) {
		// Read left and right images from file
		imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
		imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

		if (imLeft.empty()) {
			std::cerr << std::endl << "Failed to load image at: "
					<< std::string(vstrImageLeft[ni]) << std::endl;
			exit(1);
		}

		if (imRight.empty()) {
			std::cerr << std::endl << "Failed to load image at: "
					<< std::string(vstrImageRight[ni]) << std::endl;
			exit(1);
		}

		double tframe = vTimestampsCam[ni];

		cv::remap(imLeft, imLeft_rect, config.camera_parameters_[0].M1,
				config.camera_parameters_[0].M2, cv::INTER_LINEAR);
		cv::remap(imRight, imRight_rect, config.camera_parameters_[1].M1,
				config.camera_parameters_[1].M2, cv::INTER_LINEAR);

		initial_component_.poses_[ni];
		std::vector<int> vLapping_left = { 0, 0 };
		std::vector<int> vLapping_right = { 0, 0 };
		cv::Mat mask_left, mask_right;

//		std::thread threadLeft(&ORB_SLAM3::ORBextractor::extract,
//				mpORBextractorLeft, &imLeft_rect, &mask_left,
//				keypoints_left,
//				descriptors_left,
//				&vLapping_left);
//		std::thread threadRight(&ORB_SLAM3::ORBextractor::extract,
//				mpORBextractorRight, &imRight_rect, &mask_right,
//				keypoints_right,
//				descriptors_right,
//				&vLapping_right);
//		threadLeft.join();
//		threadRight.join();

		cv::Mat desc_left , desc_right;
		ORB_SLAM3::ORBextractor::extract(mpORBextractorLeft,
						 &imLeft_rect, &mask_left,
						keypoints_left,
						desc_left,
						&vLapping_left);
		ORB_SLAM3::ORBextractor::extract(mpORBextractorRight,
						 &imRight_rect, &mask_right,
						keypoints_right,
						desc_right,
						&vLapping_right);
		computeStereoMatches(initial_component_.poses_[ni]);

		initial_component_.numPoints_ += initial_component_.poses_[ni].matches_left_to_right.size();

		// plot stereo matches
		cv::Mat imLeftKeys, imRightKeys, imMatches;
		cv::Scalar kpColor = cv::Scalar(255, 0, 0);

		cv::drawMatches(imLeft_rect,
				initial_component_.poses_[ni].keypoints_left, imRight_rect,
				initial_component_.poses_[ni].keypoints_right,
				initial_component_.poses_[ni].matches_left_to_right, imMatches);

		cv::drawKeypoints(imRight_rect,
				initial_component_.poses_[ni].keypoints_right, imRightKeys,
				kpColor);
		cv::drawKeypoints(imLeft_rect,
				initial_component_.poses_[ni].keypoints_left, imLeftKeys,
				kpColor);
		//cv::imshow("matches", imMatches);

		//cv::waitKey(1); // Wait for a keystroke in the window
		std::cout << ni+1 << "/" << nImages << "                                   \r";

	}
	std::cout <<"\n";

}

void VectorGLMBSLAM6D::computeStereoMatches(OrbslamPose &pose) {


    pose.uRight = std::vector<float>(pose.keypoints_left.size() ,-1.0f);
    pose.depth = std::vector<float>(pose.keypoints_left.size() ,-1.0f);

    pose.mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    pose.mnScaleLevels = mpORBextractorLeft->GetLevels();


	static constexpr int thOrbDist = (orb_th_high + orb_th_low) / 2;

	const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

	std::vector<std::vector<std::size_t> > vRowIndices(nRows,
			std::vector<std::size_t>());

	for (int i = 0; i < nRows; i++)
		vRowIndices[i].reserve(200);

	const int Nl = pose.keypoints_left.size();
	const int Nr = pose.keypoints_right.size();

	for (int iR = 0; iR < Nr; iR++) {
		const cv::KeyPoint &kp = pose.keypoints_right[iR];
		const float &kpY = kp.pt.y;
		const float r =
				2.0f
						* mpORBextractorLeft->GetScaleFactors()[pose.keypoints_right[iR].octave];
		const int maxr = ceil(kpY + r);
		const int minr = floor(kpY - r);

		for (int yi = minr; yi <= maxr; yi++)
			vRowIndices[yi].push_back(iR);
	}

	// Set limits for search
	const float minZ = config.stereo_baseline;
	const float minD = 0;
	const float maxD = config.stereo_baseline_f / minZ;

	// For each left keypoint search a match in the right image
	std::vector<std::pair<int, int> > vDistIdx;
	vDistIdx.reserve(Nl);

	for (int iL = 0; iL < Nl; iL++) {
		const cv::KeyPoint &kpL = pose.keypoints_left[iL];
		const int &levelL = kpL.octave;
		const float &vL = kpL.pt.y;
		const float &uL = kpL.pt.x;

		const std::vector<std::size_t> &vCandidates = vRowIndices[vL];

		if (vCandidates.empty())
			continue;

		const float minU = uL - maxD;
		const float maxU = uL - minD;

		if (maxU < 0)
			continue;

		int bestDist = orb_th_high;
		size_t bestIdxR = 0;

		const cv::Mat &dL = pose.descriptors_left.row(iL);

		// Compare descriptor to right keypoints
		for (size_t iC = 0; iC < vCandidates.size(); iC++) {
			const size_t iR = vCandidates[iC];
			const cv::KeyPoint &kpR = pose.keypoints_right[iR];

			if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
				continue;

			const float &uR = kpR.pt.x;

			if (uR >= minU && uR <= maxU) {
				const cv::Mat &dR = pose.descriptors_right.row(iR);
				const int dist = descriptorDistance(dL, dR);

				if (dist < bestDist) {
					bestDist = dist;
					bestIdxR = iR;
				}
			}
		}

		// Subpixel match by correlation
		if (bestDist < thOrbDist) {
			// coordinates in image pyramid at keypoint scale
			const float uR0 = pose.keypoints_right[bestIdxR].pt.x;
			const float scaleFactor = orbExtractorInvScaleFactors[kpL.octave];
			const float scaleduL = round(kpL.pt.x * scaleFactor);
			const float scaledvL = round(kpL.pt.y * scaleFactor);
			const float scaleduR0 = round(uR0 * scaleFactor);

			// sliding window search
			const int w = 5;
			cv::Mat IL =
					mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(
							scaledvL - w, scaledvL + w + 1).colRange(
							scaleduL - w, scaleduL + w + 1);

			int bestDist = INT_MAX;
			int bestincR = 0;
			const int L = 5;
			std::vector<float> vDists;
			vDists.resize(2 * L + 1);

			const float iniu = scaleduR0 + L - w;
			const float endu = scaleduR0 + L + w + 1;
			if (iniu < 0
					|| endu
							>= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
				continue;

			for (int incR = -L; incR <= +L; incR++) {
				cv::Mat IR =
						mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(
								scaledvL - w, scaledvL + w + 1).colRange(
								scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

				float dist = cv::norm(IL, IR, cv::NORM_L1);
				if (dist < bestDist) {
					bestDist = dist;
					bestincR = incR;
				}

				vDists[L + incR] = dist;
			}

			if (bestincR == -L || bestincR == L)
				continue;

			// Sub-pixel match (Parabola fitting)
			const float dist1 = vDists[L + bestincR - 1];
			const float dist2 = vDists[L + bestincR];
			const float dist3 = vDists[L + bestincR + 1];

			const float deltaR = (dist1 - dist3)
					/ (2.0f * (dist1 + dist3 - 2.0f * dist2));

			if (deltaR < -1 || deltaR > 1)
				continue;

			// Re-scaled coordinate
			float bestuR = orbExtractorScaleFactors[kpL.octave]
					* ((float) scaleduR0 + (float) bestincR + deltaR);

			float disparity = (uL - bestuR);

			if (disparity >= minD && disparity < maxD) {
				if (disparity <= 0) {
					disparity = 0.01;
					bestuR = uL - 0.01;
				}
				pose.depth[iL] = config.stereo_baseline_f / disparity;
				pose.uRight[iL] = bestuR;

				pose.matches_left_to_right.push_back(
						cv::DMatch(iL, bestIdxR, bestDist));

				vDistIdx.push_back(std::pair<int, int>(bestDist, iL));
			}
		}
	}

	sort(vDistIdx.begin(), vDistIdx.end());
	const float median = vDistIdx[vDistIdx.size() / 2].first;
	const float thDist = 1.5f * 1.4f * median;

	for (int i = vDistIdx.size() - 1; i >= 0; i--) {
		if (vDistIdx[i].first < thDist)
			break;
		else {
			pose.uRight[vDistIdx[i].second] = -1;
			pose.depth[vDistIdx[i].second] = -1;
		}
	}

}

void VectorGLMBSLAM6D::loadConfig(std::string filename) {

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

	if (!YAML::convert<Eigen::Matrix3d>::decode(node["anchorInfo"],
			config.anchorInfo_)) {
		std::cerr << "could not load anchor info matrix \n";
		exit(1);
	}
	if (!YAML::convert<Eigen::Matrix<double , 6 , 6>>::decode(node["odomInfo"],
			config.odomInfo_)) {
		std::cerr << "could not load odom info matrix \n";
		exit(1);
	}

	if (!YAML::convert<Eigen::Matrix3d>::decode(node["stereoInfo"],
			config.stereoInfo_)) {
		std::cerr << "could not load stereo info matrix \n";
		exit(1);
	}

	double focal_length = node["camera.focal_length"].as<double>();

	config.viewingCosLimit_ = node["viewingCosLimit"].as<double>();

	config.eurocFolder_ = node["eurocFolder"].as<std::string>();
	config.use_gui_ = node["use_gui"].as<bool>();

	config.eurocTimestampsFilename_ = node["eurocTimestampsFilename"].as<
			std::string>();

	for (auto camera : node["camera_params"]) {
		Config::CameraParams params;
		params.fx = camera["fx"].as<double>();
		params.fy = camera["fy"].as<double>();
		params.cx = camera["cx"].as<double>();
		params.cy = camera["cy"].as<double>();
		params.k1 = camera["k1"].as<double>();
		params.k2 = camera["k2"].as<double>();
		params.p1 = camera["p1"].as<double>();
		params.p2 = camera["p2"].as<double>();
		params.originalImSize.width = camera["width"].as<int>();
		params.originalImSize.height = camera["height"].as<int>();

		params.newImSize = params.originalImSize;

		if (!YAML::convert<Eigen::MatrixXd>::decode(camera["cv_c0_to_camera"],
				params.cv_c0_to_camera_eigen)) {
			std::cerr << "could not load principal_point \n";
			exit(1);
		}
		cv::eigen2cv(params.cv_c0_to_camera_eigen, params.cv_c0_to_camera);

		cv::Mat dist_coeffs(4, 1, CV_64F);
		dist_coeffs.at<float>(0, 0) = params.k1;
		dist_coeffs.at<float>(1, 0) = params.k2;
		dist_coeffs.at<float>(2, 0) = params.p1;
		dist_coeffs.at<float>(3, 0) = params.p2;
		params.opencv_distort_coeffs =
				(cv::Mat_<double>(4, 1) << params.k1, params.k2, params.p1, params.p2);

		params.opencv_calibration =
				(cv::Mat_<double>(3, 3) << (float) params.fx, 0.f, (float) params.cx, 0.f, (float) params.fy, (float) params.cy, 0.f, 0.f, 1.f);

		config.camera_parameters_.push_back(params);
	}

	cv::Mat cvTlr = config.camera_parameters_[1].cv_c0_to_camera;
	Sophus::SE3d Tlr = ORB_SLAM3::Converter::toSophusd(cvTlr);

	cv::Mat R12 = cvTlr.rowRange(0, 3).colRange(0, 3);
	R12.convertTo(R12, CV_64F);
	cv::Mat t12 = cvTlr.rowRange(0, 3).col(3);
	t12.convertTo(t12, CV_64F);

	config.stereo_baseline = Tlr.translation().norm();
	config.stereo_baseline_f = config.stereo_baseline
			* config.camera_parameters_[0].fx;

	Eigen::Vector2d principal_point = { config.camera_parameters_[0].cx,
			config.camera_parameters_[0].cy };

	g2o::CameraParameters *cam_params = new g2o::CameraParameters(
			config.camera_parameters_[0].fx, principal_point,
			config.stereo_baseline);
	cam_params->setId(0);

	cv::Mat R_r1_u1, R_r2_u2;
	cv::Mat P1, P2, Q;

	cv::stereoRectify(config.camera_parameters_[0].opencv_calibration,
			config.camera_parameters_[0].opencv_distort_coeffs,
			config.camera_parameters_[1].opencv_calibration,
			config.camera_parameters_[1].opencv_distort_coeffs,
			config.camera_parameters_[0].newImSize, R12, t12, R_r1_u1, R_r2_u2,
			P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1,
			config.camera_parameters_[0].newImSize);

	cv::initUndistortRectifyMap(config.camera_parameters_[0].opencv_calibration,
			config.camera_parameters_[0].opencv_distort_coeffs, R_r1_u1,
			P1.rowRange(0, 3).colRange(0, 3),
			config.camera_parameters_[0].newImSize, CV_32F,
			config.camera_parameters_[0].M1, config.camera_parameters_[0].M2);
	cv::initUndistortRectifyMap(config.camera_parameters_[1].opencv_calibration,
			config.camera_parameters_[1].opencv_distort_coeffs, R_r2_u2,
			P2.rowRange(0, 3).colRange(0, 3),
			config.camera_parameters_[1].newImSize, CV_32F,
			config.camera_parameters_[1].M1, config.camera_parameters_[1].M2);

	std::cout << "cam0 opencv_calibration: "
			<< config.camera_parameters_[0].opencv_calibration << "\n";
	std::cout << "cam1 opencv_calibration: "
			<< config.camera_parameters_[1].opencv_calibration << "\n";
	std::cout << "cam0 opencv_distort_coeffs: "
			<< config.camera_parameters_[0].opencv_distort_coeffs << "\n";
	std::cout << "cam1 opencv_distort_coeffs: "
			<< config.camera_parameters_[1].opencv_distort_coeffs << "\n";

//    std::cout << "cam0 M1: " << config.camera_parameters_[0].M1 << "\n";
//    std::cout << "cam0 M2: " << config.camera_parameters_[0].M2 << "\n";
//    std::cout << "cam1 M1: " << config.camera_parameters_[1].M1 << "\n";
//    std::cout << "cam1 M2: " << config.camera_parameters_[1].M2 << "\n";

	config.orb_extractor.nFeatures = node["ORBextractor.nFeatures"].as<int>();
	config.orb_extractor.scaleFactor = node["ORBextractor.scaleFactor"].as<
			double>();
	config.orb_extractor.nLevels = node["ORBextractor.nLevels"].as<int>();
	config.orb_extractor.iniThFAST = node["ORBextractor.iniThFAST"].as<int>();
	config.orb_extractor.minThFAST = node["ORBextractor.minThFAST"].as<int>();
	config.stereo_init_max_depth = node["stereo_init_max_depth"].as<double>();

	mpORBextractorLeft = new ORB_SLAM3::ORBextractor(
			config.orb_extractor.nFeatures, config.orb_extractor.scaleFactor,
			config.orb_extractor.nLevels, config.orb_extractor.iniThFAST,
			config.orb_extractor.minThFAST);
	mpORBextractorRight = new ORB_SLAM3::ORBextractor(
			config.orb_extractor.nFeatures, config.orb_extractor.scaleFactor,
			config.orb_extractor.nLevels, config.orb_extractor.iniThFAST,
			config.orb_extractor.minThFAST);

	orbExtractorInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	orbExtractorScaleFactors = mpORBextractorLeft->GetScaleFactors();
}

double VectorGLMBSLAM6D::distance(PoseType *pose, PointType *lm) {

	Eigen::Vector3d point_in_camera_frame = pose->estimate().map(
			lm->estimate());
	return point_in_camera_frame.norm();

}

inline void VectorGLMBSLAM6D::initComponents() {
	components_.resize(config.numComponents_);
int i=0;
	for (auto &c : components_) {
		init(c);
		constructGraph(c);

		// optimize once at the start to calculate the hessian.
		c.poses_[0].pPose->setFixed(true);
		c.optimizer_->initializeOptimization();
		c.optimizer_->computeInitialGuess();
		c.optimizer_->setVerbose(true);
		std::string filename = std::string("init") + std::to_string(i++) +".g2o";
		c.optimizer_->save(filename.c_str(), 0);
		//std::cout << "niterations  " << c.optimizer_->optimize(1) << "\n";
		
	}

	// Visualization
	if(config.use_gui_){
		visualizer = std::make_shared<Visualizer6D>();
		visualizer->setup();
		visualizer->start();
	}

}
inline void VectorGLMBSLAM6D::run(int numSteps)
{

	for (int i = 0; i < numSteps; i++)
	{
		maxpose_prev_ = maxpose_;
		maxpose_ = components_[0].poses_.size() * i / (numSteps * 0.95);

		if (best_DA_max_detection_time_ + config.numPosesToOptimize_ < maxpose_)
		{
			maxpose_ = best_DA_max_detection_time_ + config.numPosesToOptimize_;
		}
		if (maxpose_ > components_[0].poses_.size())
			maxpose_ = components_[0].poses_.size();
		if (best_DA_max_detection_time_ == components_[0].poses_.size() - 1)
		{
			minpose_ = 0;
		}
		else
		{
			minpose_ = std::max(0,
								std::min(maxpose_ - 2 * config.numPosesToOptimize_,
										 best_DA_max_detection_time_ - config.numPosesToOptimize_));
		}

		// minpose_ = 0;
		std::cout << "maxpose: " << maxpose_ << " max det:  "
				  << best_DA_max_detection_time_ << "  " << minpose_ << "\n";
		std::cout << "iteration: " << iteration_ << " / " << numSteps << "\n";
		optimize(config.numLevenbergIterations_);


	}

}

void VectorGLMBSLAM6D::selectNN(VectorGLMBComponent6D &c) {
	std::vector<boost::bimap<int, int, boost::container::allocator<int>>> out;
	out.resize(c.DA_bimap_.size());
	for (int i = 0; i < maxpose_; i++) {
		out[i] = c.DA_bimap_[i];
	}
	int max_detection_time = maxpose_ - 1;
	while (max_detection_time > 0 && c.DA_bimap_[max_detection_time].size() == 0) {
		max_detection_time--;
	}

	for (int k = max_detection_time + 1; k < maxpose_; k++) {

		updateGraph(c);
		c.poses_[0].pPose->setFixed(true);
		c.optimizer_->initializeOptimization();
		c.optimizer_->computeInitialGuess();
		c.optimizer_->setVerbose(false);
		c.optimizer_->optimize(30);
		calculateWeight(c);
		updateDAProbs(c, k, k + 1);

		AssociationProbabilities probs;
		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
			// std::cout << "selectNN\n";
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
					if (c.landmarks_[c.DAProbs_[k][nz].i[a]
							- c.landmarks_[0].pPoint->id()].numDetections_
							== 1) {
						likelihood += std::log(config.PE_)
								- std::log(1 - config.PE_)
								+ (c.landmarks_[c.DAProbs_[k][nz].i[a]
										- c.landmarks_[0].pPoint->id()].numFoV_)
										* std::log(1 - config.PD_);
						// std::cout <<" single detection: increase:  " << std::log(config.PE_)-std::log(1-config.PE_) + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].pPoint->id()].numFoV_)*std::log(1-config.PD_) <<"\n";
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
						if (c.landmarks_[c.DAProbs_[k][nz].i[a]
								- c.landmarks_[0].pPoint->id()].numDetections_
								== 0) {
							likelihood +=
									std::log(config.PE_)
											- std::log(1 - config.PE_)
											+ (c.landmarks_[c.DAProbs_[k][nz].i[a]
													- c.landmarks_[0].pPoint->id()].numFoV_)
													* std::log(1 - config.PD_);
							// std::cout <<" 0 detection: increase:  " << std::log(config.PE_)-std::log(1-config.PE_) + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].pPoint->id()].numFoV_)*std::log(1-config.PD_)<<"\n";
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
			}
			int newdai = c.DAProbs_[k][nz].i[maxprobi];

			if (newdai == -2) {
				//std::cout << "false alarm\n";
			}
			// std::cout << "ass\n";
			if (newdai != selectedDA) { // if selected association, change bimap

				if (newdai >= 0) {
					c.landmarks_[newdai - c.landmarks_[0].pPoint->id()].numDetections_++;
					if (selectedDA < 0) {
						c.DA_bimap_[k].insert( { nz, newdai });
					} else {

						c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_--;
						c.DA_bimap_[k].left.replace_data(it, newdai);
					}
				} else { // if a change has to be made and new DA is false alarm, we need to remove the association
					c.DA_bimap_[k].left.erase(it);
					c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_--;
				}
			}
		}
	}
}

std::vector<boost::bimap<int, int, boost::container::allocator<int>> > VectorGLMBSLAM6D::sexyTime(
		VectorGLMBComponent6D &c1, VectorGLMBComponent6D &c2) {

	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif
	if (maxpose_ == 0) {
		std::vector<boost::bimap<int, int, boost::container::allocator<int>> > out(
				c1.DA_bimap_);
		return out;
	}
	boost::uniform_int<> random_merge_point(-maxpose_, maxpose_);
	std::vector<boost::bimap<int, int, boost::container::allocator<int>> > out;
	out.resize(c1.DA_bimap_.size());
	int merge_point = random_merge_point(rfs::randomGenerators_[threadnum]);

	if (merge_point >= 0) {
		// first half from da1 second from da2
		for (int i = merge_point; i < c2.DA_bimap_.size(); i++) {
			out[i] = c2.DA_bimap_[i];

		}
		for (int i = 0; i < merge_point; i++) {
			out[i] = c1.DA_bimap_[i];
		}
	} else {
		// first half from da2 second from da1
		for (int i = 0; i < -merge_point; i++) {
			out[i] = c2.DA_bimap_[i];
		}
		for (int i = -merge_point; i < c2.DA_bimap_.size(); i++) {
			out[i] = c1.DA_bimap_[i];

		}

	}
	return out;
}
inline void VectorGLMBSLAM6D::optimize(int ni) {

	std::cout << "visited  " << visited_.size() << "\n";
	if (visited_.size() > 0) {
		sampleComponents();
		std::cout << "sampling compos \n";
	}

	if (iteration_ % config.crossoverNumIter_ == 0 and false) {

		std::cout << termcolor::magenta
				<< " =========== SexyTime!============\n" << termcolor::reset;
		for (int i = 0; i < components_.size(); i++) {
			auto &c = components_[i];
			c.maxpose_ = maxpose_;

			boost::uniform_int<> random_component(0, components_.size() - 1);
			int secondComp;
			do {
				secondComp = random_component(rfs::randomGenerators_[0]);
			} while (secondComp == i);
			auto da = sexyTime(c, components_[secondComp]);
			changeDA(c, da);
		}
#pragma omp parallel for
		for (int i = 0; i < components_.size(); i++) {

			if (maxpose_prev_ != maxpose_) {
				auto &c = components_[i];
				//selectNN( c);

				updateFoV(c);
				//std::cout << "fov update: \n";

				updateDAProbs(c, minpose_, maxpose_);
				//std::cout << "da update: \n";
				c.prevDA_bimap_ = c.DA_bimap_;
				double expectedChange = 0;
				bool inserted;
				selectNN(c);
				//	std::cout << "nn update: \n";
				updateGraph(c);
				c.poses_[0].pPose->setFixed(true);
				c.optimizer_->initializeOptimization();
				c.optimizer_->computeInitialGuess();
				c.optimizer_->setVerbose(false);
				c.optimizer_->optimize(ni);
				calculateWeight(c);

				std::map<
						std::vector<
								boost::bimap<int, int,
										boost::container::allocator<int>> >,
						double>::iterator it;
#pragma omp critical(bestweight)
				{
					if (c.logweight_ > bestWeight_) {
						bestWeight_ = c.logweight_;
						best_DA_ = c.DA_bimap_;
						best_DA_max_detection_time_ = maxpose_ - 1;
						while (best_DA_max_detection_time_ > 0
								&& best_DA_[best_DA_max_detection_time_].size()
										== 0) {
							best_DA_max_detection_time_--;
						}
						std::stringstream filename;

						filename << "video/beststate_" << std::setfill('0')
								<< std::setw(5) << iterationBest_++ << ".g2o";
						c.optimizer_->save(filename.str().c_str(), 0);
						std::cout << termcolor::yellow << "========== newbest:"
								<< bestWeight_ << " ============\n"
								<< termcolor::reset;
						if(config.use_gui_){
							visualizer->update(c);
						}
					}
				}
				auto pair = std::make_pair(c.DA_bimap_, c.logweight_);
#pragma omp critical(insert)
				{
					std::tie(it, inserted) = visited_.insert(pair);
					insertionP_ = insertionP_ * 0.99;
					if (inserted)
						insertionP_ += 0.01;
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
		if (!c.reverted_)
			updateDAProbs(c, minpose_, maxpose_);
		for (int p = 0; p < maxpose_; p++) {
			c.prevDA_bimap_[p] = c.DA_bimap_[p];
		}
		double expectedChange = 0;
		bool inserted;
		std::map<
				std::vector<
						boost::bimap<int, int, boost::container::allocator<int>> >,
				double>::iterator it;

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
		std::pair<
				std::vector<
						boost::bimap<int, int, boost::container::allocator<int>> >,
				double> pair(c.DA_bimap_, c.logweight_);

#pragma omp critical(insert)
		{
			std::tie(it, inserted) = visited_.insert(pair);
			insertionP_ = insertionP_ * 0.99;
			if (inserted)
				insertionP_ += 0.01;
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
		c.poses_[0].pPose->setFixed(true);
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

				best_DA_max_detection_time_ = std::max(maxpose_ - 1, 0);
				while (best_DA_max_detection_time_ > 0
						&& best_DA_[best_DA_max_detection_time_].size() == 0) {
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

	if (insertionP_ > 0.95 && temp_ > 5e-1) {
		temp_ *= 0.5;
		std::cout << "reducing: \n";
	}
	if (insertionP_ < 0.05 && temp_ < 1e5) {
		temp_ *= 2;
		std::cout << "augmenting: ";
	}

}
inline void VectorGLMBSLAM6D::calculateWeight(VectorGLMBComponent6D &c) {
	double logw = 0;

	for (int k = 0; k < c.poses_.size(); k++) {
		for (int nz = 0; nz < c.poses_[k].Z_.size(); nz++) {
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
								* (c.poses_[k].Z_[nz]->dimension()
										* std::log(2 * M_PI)
										- std::log(
												c.poses_[k].Z_[nz]->information().determinant()));
			}
		}

		for (int lm = 0; lm < c.poses_[k].fov_.size(); lm++) {

			if (c.DA_bimap_[k].right.count(c.poses_[k].fov_[lm]) > 0) {
				logw += std::log(config.PD_);
			} else {
				bool exists = c.landmarks_[c.poses_[k].fov_[lm]
						- c.landmarks_[0].pPoint->id()].numDetections_ > 0;
				if (exists) {
					logw += std::log(1 - config.PD_);
				}
			}
		}
	}

	for (int lm = 0; lm < c.landmarks_.size(); lm++) {
		bool exists = c.landmarks_[lm].numDetections_ > 0;
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

inline void VectorGLMBSLAM6D::updateGraph(VectorGLMBComponent6D &c) {
	for (int k = 0; k < maxpose_; k++) {
		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
			int selectedDA = -2;
			auto it = c.DA_bimap_[k].left.find(nz);

			if (it != c.DA_bimap_[k].left.end()) {
				selectedDA = it->second;
			}
			int previd =
					c.poses_[k].Z_[nz]->vertex(1) ?
							c.poses_[k].Z_[nz]->vertex(1)->id() : -2; /**< previous data association */
			if (previd == selectedDA) {
				continue;
			}
			if (selectedDA >= 0) {

				// if edge was already in graph, modify it
				if (previd >= 0) {
					c.optimizer_->setEdgeVertex(c.poses_[k].Z_[nz], 0,
							dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
									selectedDA)->second)); // this removes the edge from the list in both vertices
				} else {
					c.poses_[k].Z_[nz]->setVertex(0,
							dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
									selectedDA)->second));

					c.optimizer_->addEdge(c.poses_[k].Z_[nz]);
				}

			} else {

				c.optimizer_->removeEdge(c.poses_[k].Z_[nz]);
				c.poses_[k].Z_[nz]->setVertex(1, NULL);

			}

		}
	}
	for (auto lm : c.landmarks_) {
		// if landmark has only 1 edge then it is not detected we deactivate it
		if (lm.pPoint->edges().size() == 1) {
			for (auto edge : lm.pPoint->edges()) {
				dynamic_cast<g2o::OptimizableGraph::Edge*>(edge)->setLevel(1);
			}
		} else {
			for (auto edge : lm.pPoint->edges()) {
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
inline void VectorGLMBSLAM6D::printFoV(VectorGLMBComponent6D &c) {
	std::cout << "FoV:\n";
	for (int k = 0; k < c.poses_.size(); k++) {
		std::cout << k << "  FoV at:   ";
		for (int lmid : c.poses_[k].fov_) {
			std::cout << "  ,  " << lmid;
		}
		std::cout << "\n";
	}
}
inline void VectorGLMBSLAM6D::printDAProbs(VectorGLMBComponent6D &c) {
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
inline void VectorGLMBSLAM6D::printDA(VectorGLMBComponent6D &c,
		std::ostream &s) {
	for (int k = 0; k < c.DA_bimap_.size(); k++) {
		s << k << ":\n";
		print_map(c.DA_bimap_[k].left, s);
	}
}

inline double VectorGLMBSLAM6D::sampleLMBirth(VectorGLMBComponent6D &c) {
	double expectedWeightChange = 0;
	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif

	for (int i = 0; i < c.landmarks_.size(); i++) {
		if (c.landmarks_[i].numDetections_ > 0
				|| c.landmarks_[i].numFoV_ == 0) {
			continue;
		}
		//
		c.landmarksInitProb_[i] = c.landmarksInitProb_[i]
				/ (c.landmarks_[i].numFoV_ * config.PD_);
		if (c.landmarksInitProb_[i] > uni_dist(randomGenerators_[threadnum])) {
			// reset all associations to false alarms
			expectedWeightChange += (config.logKappa_ + (1 - config.PD_))
					* c.landmarks_[i].numFoV_;
			int numdet = 0;
			for (int k = minpose_; k < maxpose_; k++) {
				for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {
					// if measurement is associated, continue
					auto it = c.DA_bimap_[k].left.find(nz);
					if (it != c.DA_bimap_[k].left.end()) {
						continue;
					}
					for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++) {
						if (c.DAProbs_[k][nz].i[a]
								== c.landmarks_[i].pPoint->id()) {
							if (c.DAProbs_[k][nz].l[a]
									> c.DAProbs_[k][nz].l[c.DAProbs_[k][nz].l.size()
											- 1]) {
								c.DA_bimap_[k].insert(
										{ nz, c.landmarks_[i].pPoint->id() });
								c.landmarks_[i].numDetections_++;
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
			 << c.landmarks_[i].numDetections_ << " numfov: "
			 << c.landmarks_[i].numFoV_ << "  expectedChange "
			 << expectedWeightChange << "\n";
			 */

			c.landmarks_[i].numDetections_ = 0;

		}
	}

//std::cout << "Death Change  " <<expectedWeightChange <<"\n";
	return expectedWeightChange;
}

inline double VectorGLMBSLAM6D::mergeLM(VectorGLMBComponent6D &c) {
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
			c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_--;
			c.landmarks_[toAddMeasurements - c.landmarks_[0].pPoint->id()].numDetections_++;

			c.DA_bimap_[k].right.replace_key(it, toAddMeasurements);

		}
	}
	if (c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_
			!= 0) {
		std::cerr << "landmark  numDetections_ not zero"
				<< c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_
				<< "\n";
	}
	c.tomerge_.clear();
	return expectedWeightChange;
}

inline double VectorGLMBSLAM6D::sampleLMDeath(VectorGLMBComponent6D &c) {
	double expectedWeightChange = 0;
	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
threadnum = omp_get_thread_num();
#endif

	for (int i = 0; i < c.landmarks_.size(); i++) {
		if (c.landmarks_[i].numDetections_ <= 1) {
			continue;
		}

		c.landmarksResetProb_[i] = -(c.landmarks_[i].numDetections_)
				* std::log(config.PD_)
				- (c.landmarks_[i].numFoV_ - c.landmarks_[i].numDetections_)
						* std::log(1 - config.PD_) - std::log(config.PE_)
				+ std::log(1 - config.PE_);
		double p = std::exp(c.landmarksResetProb_[i]);
		if (uni_dist(randomGenerators_[threadnum]) < p / (1 + p)) {
			/*
			 c.landmarksResetProb_[i] = (1-((double)c.landmarks_[i]numDetections_)/c.landmarks_[i].numFoV_)*(config.PD_);
			 if(uni_dist(randomGenerators_[threadnum]) < c.landmarksResetProb_[i]){*/
			// reset all associations to false alarms
			expectedWeightChange += config.logKappa_
					* c.landmarks_[i].numDetections_;
			int numdet = 0;
			for (int k = minpose_; k < maxpose_; k++) {
				auto it = c.DA_bimap_[k].right.find(
						c.landmarks_[i].pPoint->id());
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
					* c.landmarks_[i].numFoV_;
			/*
			 std::cout << termcolor::red << "KILL LANDMARK\n" << termcolor::reset
			 << c.landmarksResetProb_[i] << " n "
			 << c.landmarks_[i].numDetections_ << " nfov:"
			 << c.landmarks_[i].numFoV_ << "  expectedChange "
			 << expectedWeightChange << "\n";
			 */

			c.landmarks_[i].numDetections_ = 0;

		}
	}

	//std::cout << "Death Change  " <<expectedWeightChange <<"\n";
	return expectedWeightChange;

}

inline double VectorGLMBSLAM6D::sampleDA(VectorGLMBComponent6D &c) {
	std::vector<double> P;
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
					if (c.landmarks_[c.DAProbs_[k][nz].i[a]
							- c.landmarks_[0].pPoint->id()].numDetections_
							== 1) {
						likelihood += std::log(config.PE_)
								- std::log(1 - config.PE_)
								+ (c.landmarks_[c.DAProbs_[k][nz].i[a]
										- c.landmarks_[0].pPoint->id()].numFoV_)
										* std::log(1 - config.PD_);
						//std::cout <<" single detection: increase:  " << std::log(config.PE_)-std::log(1-config.PE_) + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].pPoint->id()].numFoV_)*std::log(1-config.PD_) <<"\n";
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
						if (c.landmarks_[c.DAProbs_[k][nz].i[a]
								- c.landmarks_[0].pPoint->id()].numDetections_
								== 0) {
							likelihood +=
									std::log(config.PE_)
											- std::log(1 - config.PE_)
											+ (c.landmarks_[c.DAProbs_[k][nz].i[a]
													- c.landmarks_[0].pPoint->id()].numFoV_)
													* std::log(1 - config.PD_);
							//std::cout <<" 0 detection: increase:  " << std::log(config.PE_)-std::log(1-config.PE_) + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].pPoint->id()].numFoV_)*std::log(1-config.PD_)<<"\n";
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

				P[i] = exp((probs.l[i] - maxprob)); // /temp_);

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
				//c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0].pPoint->id()] *= (P[ P.size()-1] )/P[sample];
				/*
				 if(probs.i[sample] != c.DAProbs_[k][nz].i[maxprobi]){
				 c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0].pPoint->id()] +=  c.DAProbs_[k][nz].l[maxprobi] - probs.l[sample]; //(1 )/alternativeprob;

				 }else{
				 c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0].pPoint->id()] += probs.l[probs.l.size()-1] - probs.l[sample] ;
				 }*/

				c.landmarksResetProb_[probs.i[sample]
						- c.landmarks_[0].pPoint->id()] += std::log(
						P[sample] / alternativeprob);

			} else {

				if (c.DAProbs_[k][nz].i[maxlikelihoodi] >= 0) {
					//std::cout << "increasing init prob of lm " <<c.DAProbs_[k][nz].i[maxlikelihoodi] << "  by " <<maxlikelihood  << "- " << probs.l[sample]<< "\n";
					c.landmarksInitProb_[c.DAProbs_[k][nz].i[maxlikelihoodi]
							- c.landmarks_[0].pPoint->id()] += 1;
				}
			}

			if (probs.i[sample] != selectedDA) { // if selected association, change bimap

				if (probs.i[sample] >= 0) {
					c.landmarks_[probs.i[sample] - c.landmarks_[0].pPoint->id()].numDetections_++;
					if (selectedDA < 0) {
						c.DA_bimap_[k].insert( { nz, probs.i[sample] });
					} else {

						c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_--;
						c.DA_bimap_[k].left.replace_data(it, probs.i[sample]);

						// add an log for possible landmark merge
						c.tomerge_.push_back(
								std::make_pair(probs.i[sample], selectedDA));
					}
				} else { // if a change has to be made and new DA is false alarm, we need to remove the association
					c.DA_bimap_[k].left.erase(it);
					c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_--;

				}

			}

		}
	}

	return expectedWeightChange;
}

inline void VectorGLMBSLAM6D::updateFoV(VectorGLMBComponent6D &c) {
	for (auto &map_point : c.landmarks_) {
		map_point.numFoV_ = 0;
		map_point.numFoV_ = 0;
		map_point.is_in_fov_.resize(0);
	}

	for (int k = 0; k < maxpose_; k++) {

		c.poses_[k].fov_.clear();

		if (c.poses_[k].Z_.size() > 0) { // if no measurements we set FoV to empty ,

			for (int lm = 0; lm < c.landmarks_.size(); lm++) {

				if (c.poses_[k].isInFrustum(&c.landmarks_[lm],
						config.viewingCosLimit_, config.g2o_cam_params)) {
					c.poses_[k].fov_.push_back(c.landmarks_[lm].pPoint->id());
					c.landmarks_[lm].numFoV_++;
					c.landmarks_[lm].is_in_fov_.push_back(k);
				}
			}
		}
	}
}

inline void VectorGLMBSLAM6D::updateDAProbs(VectorGLMBComponent6D &c,
		int minpose, int maxpose) {

	g2o::JacobianWorkspace jac_ws;
	StereoMeasurementEdge z;
	jac_ws.updateSize(2, 3 * 6);
	jac_ws.allocate();

	for (int k = minpose; k < maxpose; k++) {
		c.DAProbs_[k].resize(c.poses_[k].Z_.size());

		double posHLogDet;
		if (!c.poses_[k].pPose->fixed()) {
			posHLogDet = std::log(c.poses_[k].pPose->hessianDeterminant());
		}
		PoseType::HessianBlockType poseHessian(
				c.poses_[k].pPose->hessianData());

		for (int nz = 0; nz < c.DAProbs_[k].size(); nz++) {

			// setting the topology of DAProbs to include all measurements in current FoV
			c.DAProbs_[k][nz].i = c.poses_[k].fov_;
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
				c.poses_[k].Z_[nz]->g2o::BaseBinaryEdge<3, g2o::Vector3,
						PointType, PoseType>::linearizeOplus(jac_ws);
				StereoMeasurementEdge::JacobianXjOplusType Jpose =
						c.poses_[k].Z_[nz]->jacobianOplusXj();
				poseHessianCopy -= Jpose.transpose()
						* c.poses_[k].Z_[nz]->information() * Jpose;
			}
			for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++) {
				c.DAProbs_[k][nz].l[a] = 0;
				if (c.DAProbs_[k][nz].i[a] == -2) { // set measurement to false alarm
					c.DAProbs_[k][nz].l[a] = config.logKappa_;
				} else {

					c.DAProbs_[k][nz].l[a] += std::log(config.PD_)
							- std::log(1 - config.PD_);
					c.poses_[k].Z_[nz]->setVertex(1,
							dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
									c.DAProbs_[k][nz].i[a])->second));

					c.poses_[k].Z_[nz]->g2o::BaseBinaryEdge<3, g2o::Vector3,
							PointType, PoseType>::linearizeOplus(jac_ws);
					c.poses_[k].Z_[nz]->computeError();

					Eigen::Matrix<double, StereoMeasurementEdge::Dimension, 1,
							Eigen::ColMajor> omega_r =
							-c.poses_[k].Z_[nz]->error();

					// if pose is not fixed, calc updated pose and lm
					if (!c.poses_[k].pPose->fixed()) {
						PointType::HessianBlockType::PlainMatrix h;
						PointType::HessianBlockType pointHessian(h.data());

						if (c.landmarks_[c.DAProbs_[k][nz].i[a]
								- c.landmarks_[0].pPoint->id()].numDetections_
								> 0) {
							new (&pointHessian) PointType::HessianBlockType(
									c.landmarks_[c.DAProbs_[k][nz].i[a]
											- c.landmarks_[0].pPoint->id()].pPoint->hessianData());

							//std::cout << "g2o pointH: " << pointHessian << "\n\n\n";
						} else {
							h = config.anchorInfo_;
							//std::cout << "calc pointH: " << pointHessian << "\n\n\n";

						}
						//std::cout << "numdetections:  " << c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].numDetections_  << "\n";

						StereoMeasurementEdge::JacobianXjOplusType Jpose =
								c.poses_[k].Z_[nz]->jacobianOplusXj();
						StereoMeasurementEdge::JacobianXiOplusType Jpoint =
								c.poses_[k].Z_[nz]->jacobianOplusXi();

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
								Jpose.transpose()
										* c.poses_[k].Z_[nz]->information()
										* Jpose;
						H.block(PoseType::Dimension, PoseType::Dimension,
								PointType::Dimension, PointType::Dimension) +=
								Jpoint.transpose()
										* c.poses_[k].Z_[nz]->information()
										* Jpoint;

						H.block(PoseType::Dimension, 0, PointType::Dimension,
								PoseType::Dimension) = Jpoint.transpose()
								* c.poses_[k].Z_[nz]->information() * Jpose;
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
								c.poses_[k].Z_[nz]->information().determinant())
								+ posHLogDet;

						c.DAProbs_[k][nz].l[a] += -0.5
								* (c.poses_[k].Z_[nz]->chi2() - sol.dot(b));
						c.DAProbs_[k][nz].l[a] += -0.5
								* c.poses_[k].Z_[nz]->dimension()
								* std::log(2 * M_PI);
					} else { // if pose is fixed only calculate updated landmark
						PointType::HessianBlockType pointHessian(
								c.landmarks_[c.DAProbs_[k][nz].i[a]
										- c.landmarks_[0].pPoint->id()].pPoint->hessianData());

						StereoMeasurementEdge::JacobianXiOplusType Jpoint =
								c.poses_[k].Z_[nz]->jacobianOplusXi();

						Eigen::Matrix<double, PointType::Dimension,
								PointType::Dimension> H;
						H.setZero();
						H = pointHessian
								+ Jpoint.transpose()
										* c.poses_[k].Z_[nz]->information()
										* Jpoint;
						Eigen::Matrix<double, PointType::Dimension, 1> b, sol;
						b = Jpoint.transpose() * omega_r;

						Eigen::LLT<
								Eigen::Matrix<double, PointType::Dimension,
										PointType::Dimension>> lltofH(H);
						sol = lltofH.solve(b);

						c.DAProbs_[k][nz].l[a] += -std::log(
								lltofH.matrixL().determinant());
						c.DAProbs_[k][nz].l[a] +=
								std::log(
										c.poses_[k].Z_[nz]->information().determinant());

						//c.DAProbs_[k][nz].l[a] += -0.5 * (.poses_[k].Z_[nz]->chi2() - sol.dot(b));
						c.DAProbs_[k][nz].l[a] += -0.5
								* (c.poses_[k].Z_[nz]->chi2());
						c.DAProbs_[k][nz].l[a] += -0.5
								* c.poses_[k].Z_[nz]->dimension()
								* std::log(2 * M_PI);

					}
				}
			}
			if (selectedDA >= 0) {
				c.poses_[k].Z_[nz]->setVertex(1,
						dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.optimizer_->vertices().find(
								selectedDA)->second));
				c.poses_[k].Z_[nz]->linearizeOplus();
				c.poses_[k].Z_[nz]->computeError();
			} else {
				c.poses_[k].Z_[nz]->setVertex(1, NULL);

			}

		}

	}

}


bool VectorGLMBSLAM6D::initMapPoint(OrbslamPose &pose, int numMatch,  OrbslamMapPoint &lm, int newId){


    Eigen::Vector3d uvu;
    int nl = pose.matches_left_to_right[numMatch].queryIdx;
    uvu[0] = pose.keypoints_left[nl].pt.x;
    uvu[1] = pose.keypoints_left[nl].pt.y;
    uvu[2] = pose.uRight[nl];





	lm.pPoint = new PointType();
	lm.pPoint->setId(newId);

	StereoMeasurementEdge *stereo_edge= new StereoMeasurementEdge();
	stereo_edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(pose.pPose));
	stereo_edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(lm.pPoint));
	stereo_edge->fx = config.camera_parameters_[0].fx;
	stereo_edge->fy = config.camera_parameters_[0].fy;
	stereo_edge->cx = config.camera_parameters_[0].cx;
	stereo_edge->cy = config.camera_parameters_[0].cy;
	stereo_edge->bf = config.stereo_baseline_f;
	stereo_edge->setInformation(config.stereoInfo_);
	stereo_edge->setMeasurement(uvu);
	pose.Z_[numMatch]=stereo_edge;



	Eigen::Vector3d  point_world_frame ;

	lm.numDetections_=0;
	lm.numFoV_=0;
	int level = pose.keypoints_left[nl].octave;
	const int nLevels = pose.mnScaleLevels;
	lm.mfMaxDistance = pose.depth[nl]*pose.mvScaleFactors[level];
	lm.mfMinDistance = lm.mfMaxDistance/pose.mvScaleFactors[nLevels-1];

	if (!cam_unproject(*stereo_edge, pose.point_camera_frame[numMatch] )){
		//std::cerr << "stereofail\n";
		return false;
	}


	lm.normalVector = pose.point_camera_frame[numMatch].normalized();
	// with only 2 descriptors (left,right) we can pick either one to represent the point
	//allocating new
	pose.descriptors_left.row(nl).copyTo(lm.descriptor);

	point_world_frame = pose.pPose->estimate().inverse().map(pose.point_camera_frame[numMatch]);
	lm.pPoint->setEstimate(point_world_frame);
	stereo_edge->computeError();






	return true;
}


inline void VectorGLMBSLAM6D::constructGraph(VectorGLMBComponent6D &c) {
	c.numPoses_ = initial_component_.numPoses_;

	c.poses_ = initial_component_.poses_;

	int edgeid=c.numPoses_;

	for(int k =0 ; k<c.poses_.size() ; k++){

		//create graph pose
		c.poses_[k].pPose = new PoseType();
		g2o::SE3Quat pose_estimate;
		c.poses_[k].pPose->setEstimate(pose_estimate);

		c.poses_[k].pPose->setId(k);
		c.optimizer_->addVertex(c.poses_[k].pPose);
		if (k>0) {
			OdometryEdge * odo = new OdometryEdge;
			odo->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.poses_[k-1].pPose));
			odo->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(c.poses_[k].pPose));
			odo->setInformation(config.odomInfo_);
			if (!c.optimizer_->addEdge(odo) ) {
				std::cerr << "odo edge insert fail \n";
			}
		}

		c.poses_[k].Z_.resize(c.poses_[k].matches_left_to_right.size());
		c.poses_[k].point_camera_frame.resize(c.poses_[k].matches_left_to_right.size());

		for(int nz =0 ; nz<c.poses_[k].matches_left_to_right.size() ; nz++){
			OrbslamMapPoint lm;

			if (initMapPoint(c.poses_[k], nz ,  lm,  edgeid++) ){

				c.optimizer_->addVertex(lm.pPoint);

				// if (!c.optimizer_->addEdge(c.poses_[k].Z_[nz])) {
				// 	std::cerr << "initial measurement edge insert fail \n";
				// }

			}


		}

	}


}

bool VectorGLMBSLAM6D::cam_unproject(const StereoMeasurementEdge &measurement,
		 Eigen::Vector3d &trans_xyz) {

	Eigen::Vector3d meas = measurement.measurement();
	trans_xyz[2] = config.stereo_baseline_f/(meas[0]-meas[2]);
	trans_xyz[0] = (meas[0]-config.camera_parameters_[0].cx) *trans_xyz[2] /config.camera_parameters_[0].fx;
	trans_xyz[1] = (meas[1]-config.camera_parameters_[0].cy) *trans_xyz[2] /config.camera_parameters_[0].fy;
	if (trans_xyz[2]>config.stereo_init_max_depth || trans_xyz[2]<0 || trans_xyz[2]!=trans_xyz[2]){
		return false;
	}


	return true;
}



void VectorGLMBSLAM6D::waitForGuiClose(){
	if (config.use_gui_){
		visualizer->display_thread_->join();
	}
}

inline void VectorGLMBSLAM6D::init(VectorGLMBComponent6D &c) {
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
