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
#ifndef RFSHMCSLAM_HPP
#define RFSHMCSLAM_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Timer.hpp"
#include <Eigen/Core>
#include "MurtyAlgorithm.hpp"
#include "PermutationLexicographic.hpp"
#include "RFSHMCParticle.hpp"
#include <math.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include "OSPA.hpp"
#include "RandomVecMathTools.hpp"
#include "RFSCeresSLAM.hpp"
#include "GaussianGenerators.hpp"
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>

namespace rfs {

/**
 *  \class RFSHMCSLAM
 *  \brief Random Finite Set Hamiltonian Monte Carlo SLAM
 *
 *
 *
 *  \tparam RobotProcessModel A robot process model derived from ProcessModel
 *  \tparam MeasurementModel A sensor model derived from MeasurementModel
 *  \author  Felipe Inostroza
 */
template<class RobotProcessModel, class MeasurementModel>
class RFSHMCSLAM {

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef typename RobotProcessModel::TState TPose;
	typedef typename RobotProcessModel::TInput TInput;
	typedef typename MeasurementModel::TLandmark TLandmark;
	typedef typename MeasurementModel::TMeasurement TMeasurement;
	typedef RFSHMCParticle<RobotProcessModel, MeasurementModel> TParticle;
	typedef std::vector<TParticle> TParticleSet;
	static const int PoseDim = TPose::Vec::RowsAtCompileTime;
	static const int LandmarkDim = TLandmark::Vec::RowsAtCompileTime;
	/**
	 * \brief Configurations for this RFSBatchPSO optimizer
	 */
	struct Config {

		int nParticles_; /**< number of particles for the PSO algorithm */



		/** The threshold used to determine if a possible meaurement-landmark
		 *  pairing is significant to worth considering
		 */
		double MeasurementLikelihoodThreshold_;

		double mapFromMeasurementProb_; /**< probability that each measurement will initialize a landmark on map initialization*/

		double m; /**<  Constant mass, could be replaced by a (possibly diagonal) matrix*/



		int K; /**< Number of timesteps to integrate with leapfrog */

		double temp; /**< temperature  used to tune how greedy the algorithm is*/

		double Pb; /**< birth move probability */
		double Pd; /**< Death move probability  note the HMC move probability is 1-Pd-Pb*/


	} config;

	/**
	 * Constructor
	 */
	RFSHMCSLAM();

	/** Destructor */
	~RFSHMCSLAM();

	/**
	 * Add a single measurement
	 * @param z The measurement to add, make sure the timestamp is valid.
	 */
	void
	addMeasurement(TMeasurement z);

	/**
	 * Add a set of measurements
	 * @param Z The measurements to add, make sure the timestamp is valid.
	 */
	void
	addMeasurement(std::vector<TMeasurement> Z);

	/**
	 * Add a single odometry input
	 * @param u the odometry input
	 */
	void
	addInput(TInput u);

	/**
	 * set all the odometry inputs
	 * @param U vector containing all odometry inputs
	 */
	void
	setInputs(std::vector<TInput> U);

	/**
	 * Generate random trajectories based on the motion model of the robot
	 */
	void
	initTrajectories(std::vector<TParticle> &particles);

	/**
	 * Generate random maps based on the already initialized trajectories
	 */
	void
	initMaps(std::vector<TParticle> &particles);

	/**
	 * initialize the particles
	 */
	void
	init(std::vector<TParticle> &particles);

	/**
	 * Get the best  PSO particle
	 * @return pointer to the particle
	 */
	TParticle*
	getBestParticle(std::vector<TParticle> &particles);

	/**
	 * Sets the likelihood and gradients of a particle to zero.
	 * @param particle the particle
	 */
	void clear(TParticle &particle);

	/**
	 * Calculates the measurement likelihood of particle at time k, gradients are added to the current gradients in the particle object.
	 * @param particleIdx The particle, trajectory and map
	 * @param k the time for which to calculate the likelihood
	 * @return The landmarks that have nonzero detection probability
	 */

	std::vector<unsigned int>
	rfsMeasurementLogLikelihood(TParticle &particle, const int k);

	/**
	 * Calculates the measurement likelihood of particle  including all available times stores the likelihood value and gradients on the particle object.
	 * @param particle[in,out] The particle, trajectory and map,
	 * @return The landmarks that have nonzero detection probability
	 */

	std::vector<unsigned int>
	rfsMeasurementLogLikelihood(TParticle &particle);

	/**
	 * Evaluate the current likelihood of all the particles
	 */
	void
	evaluateLikelihoods(std::vector<TParticle> &particles);

	/**
	 * Perform the reversible jump HMC step,where either the birth death or HMC move is selected randomly and executed.
	 */
	void
	reversibleJumpHMC(std::vector<TParticle> &particles);

	/**
	 * Perform the basic hamiltonian step, where the leapfrog algorithm is used to propose move and it is accepted with probability equal to the Metropolis Hasting acceptance rate.
	 * @param particle the input particle
	 * @return the new sample, possibly equal to particle
	 */
	TParticle basicHamiltonianMCMC(TParticle &particle);

	/**
	 * Randomly select between the birth death and HMC method, completing the reverisble jump HMC algorithm.
	 * @param particle the input particle
	 * @return the new sample, possibly equal to particle
	 */
	TParticle reversibleJumpHMC(TParticle &particle);

	/**
	 * Run the leapFrog algorithm n times on a particle
	 * @param[in,out] particle  input particle to start Hamiltonian Simulation
	 * @param[in] n number of leapfrog iterations to run
	 * @return false if the simulation has become unstable are states are not finite.
	 *
	 */
	bool leapFrog(TParticle &particle, int n);

	/**
	 * Perform a half step on the momentum state, ie predict the momentum at t + e/2
	 * @param particle[in,out] the state of the particle, the momentum is predicted in place.
	 */
	void momentumHalfStep(TParticle &particle);

	/**
	 * Perform a  step on the state (trajectory and landmarks),ie. predict the state at t+e using the momentum at t + e/2
	 * @param particle[in,out] the state of the particle, the state is predicted in place.
	 */
	void stateFullStep(TParticle &particle);
	/***
	 * Resample random velocities with  Gaussian distribution.
	 * @param particle[in,out] The particle, whose momentum is changed
	 */
	void
	resampleMomentum(TParticle &particle);

	/***
	 * Reset initial pose to zero
	 * @param particle[in,out] The particle, poses and map are changed
	 */
	void
	renormalize(TParticle &particle);


	/***
	 * Apply the birth MCMC transition kernel
	 * @param particle[in,out] TThe particle, map is changed
	 */
	void birthMove(TParticle &particle);
	/***
	 * Apply the death MCMC transition kernel
	 * @param particle[in,out] TThe particle, map is changed
	 */
	void deathMove(TParticle &particle);

	void
	birthDeathStep(TParticle &particle);

	/***
	 * Calculate the Hamiltonian
	 * @param particle[in] The particle
	 */
	double hamiltonian(const TParticle &particle);
	/***
	 * Check the particle state
	 * @return true if all states are finite
	 */
	bool isFinite(TParticle &particle);

	MeasurementModel *mModelPtr_;
	RobotProcessModel *robotProcessModelPtr_;
private:

	int nThreads_; /**< Number of threads  */
	int iteration_;
	std::vector<TInput> inputs_; /**< vector containing all odometry inputs */
	std::vector<std::vector<TMeasurement> > Z_; /**< vector containing all feature measurements */
	std::vector<TimeStamp> time_;

	static double constexpr BIG_NEG_NUM = -1000; // to represent log(0)

};

//////////////////////////////// Implementation ////////////////////////

template<class RobotProcessModel, class MeasurementModel>
typename RFSHMCSLAM<RobotProcessModel, MeasurementModel>::TParticle*
RFSHMCSLAM<RobotProcessModel, MeasurementModel>::getBestParticle(std::vector<TParticle> &particles) {

	double maxlikelihood = -std::numeric_limits<double>::infinity();
	double maxi = -1;
	for (int i = 0; i < particles.size(); i++) {
		if (maxlikelihood < particles[i].bestLikelihood) {
			maxi = i;
			maxlikelihood = particles[i].bestLikelihood;
		}

	}

	return &(particles[maxi]);
}

template<class RobotProcessModel, class MeasurementModel>
RFSHMCSLAM<RobotProcessModel, MeasurementModel>::RFSHMCSLAM() {
	nThreads_ = 1;
	iteration_ = 0;

#ifdef _OPENMP
	nThreads_ = omp_get_max_threads();
#endif
	mModelPtr_ = new MeasurementModel();
	robotProcessModelPtr_ = new RobotProcessModel();

}

template<class RobotProcessModel, class MeasurementModel>
RFSHMCSLAM<RobotProcessModel, MeasurementModel>::~RFSHMCSLAM() {

	delete mModelPtr_;
	delete robotProcessModelPtr_;
}

template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::addInput(TInput u) {

	inputs_.push_back(u);
	time_.push_back(u.getTime());
	std::sort(inputs_.begin(), inputs_.begin());
	std::sort(time_.begin(), time_.begin());
	Z_.resize(inputs_.size() + 1);

}

template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::setInputs(std::vector<TInput> U) {

	inputs_ = U;
	std::sort(inputs_.begin(), inputs_.begin());

	time_.resize(inputs_.size() + 1);
	time_[0] = 0;
	for (int i = 0; i < inputs_.size(); i++) {
		time_[i + 1] = inputs_[i].getTime();
	}
	Z_.resize(inputs_.size() + 1);
}

template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::addMeasurement(TMeasurement z) {

	TimeStamp zTime = z.getTime();

	auto it = std::lower_bound(time_.begin(), time_.end(), zTime);
	if (*it != zTime) {
		std::cerr << "Measurement time does not match with any of the odometries\n zTime: " << zTime.getTimeAsDouble() << "\n";
		std::exit(1);
	}
	int k = it - time_.begin();
	Z_[k].push_back(z);

}

template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::addMeasurement(std::vector<TMeasurement> Z) {

	for (int i = 0; i < Z.size(); i++) {
		this->addMeasurement(Z[i]);
	}
}

template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::init(std::vector<TParticle> &particles) {
	particles.resize(config.nParticles_);

	initTrajectories(particles);
	initMaps(particles);
}

template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::initTrajectories(std::vector<TParticle> &particles) {

	for (int i = 0; i < config.nParticles_; i++) {

		particles[i].trajectory.resize(inputs_.size() + 1);
		particles[i].trajectory_momentum.resize(inputs_.size() + 1);
		particles[i].trajectory_gradient.resize(inputs_.size() + 1);
		particles[i].inputs.resize(inputs_.size());
		particles[i].inputs_momentum.resize(inputs_.size());
		particles[i].trajectory[0].setZero();
		for (int k = 0; k < inputs_.size(); k++) {
			TimeStamp dT = time_[k+1] -time_[k];
			TPose prePose(particles[i].trajectory[k],time_[k]), postPose;
			TInput in;
			robotProcessModelPtr_->sample(postPose, prePose, inputs_[k], dT, false, true, &in);
			particles[i].trajectory[k + 1] = postPose.get();
			particles[i].inputs[k]= in.get();
		}
		particles[i].bestTrajectory = particles[i].trajectory;
		particles[i].bestTrajectory_momentum = particles[i].trajectory_momentum;
		particles[i].bestInputs = particles[i].inputs;

		particles[i].bestInputs_momentum.resize(inputs_.size());

	}
}
template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::initMaps(std::vector<TParticle> &particles) {

	for (int i = 0; i < config.nParticles_; i++) {

		for (int k = 0; k < particles[i].trajectory.size(); k++) {
			for (int nz = 0; nz < Z_[k].size(); nz++) {

				if (drand48() < config.mapFromMeasurementProb_) {

					TLandmark lm;
					this->mModelPtr_->inverseMeasure(particles[i].trajectory[k], Z_[k][nz], lm);
					lm.sample(lm);
					particles[i].landmarks.push_back(lm.get());
				}
			}
		}

		particles[i].landmarks_momentum.resize(particles[i].landmarks.size());
		particles[i].landmarks_gradient.resize(particles[i].landmarks.size());
		particles[i].bestLandmarks_momentum.resize(particles[i].landmarks.size());
		particles[i].bestLandmarks = particles[i].landmarks;

	}

}

template<class RobotProcessModel, class MeasurementModel>
inline void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::clear(TParticle& particle) {
	particle.currentLikelihood = 0;
	for (auto &grad : particle.trajectory_gradient) {
		grad.setZero();
	}
	for (auto &grad : particle.landmarks_gradient) {
		grad.setZero();
	}
}

template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::evaluateLikelihoods(std::vector<TParticle> &particles) {

#pragma omp parallel for
	for (int i = 0; i < particles.size(); i++) {
		rfsMeasurementLogLikelihood(particles[i]);

	}

}
template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::reversibleJumpHMC(std::vector<TParticle> &particles) {

#pragma omp parallel for
	for (int i = 0; i < particles.size(); i++) {
		particles[i] = reversibleJumpHMC(particles[i]);

	}

}


template<class RobotProcessModel, class MeasurementModel>
std::vector<unsigned int> RFSHMCSLAM<RobotProcessModel, MeasurementModel>::rfsMeasurementLogLikelihood(TParticle &particle) {
	clear(particle);

	std::vector<unsigned int> lmInFovIdx;
	std::vector<unsigned int> removeLM;
	removeLM.resize(particle.landmarks.size(),1);


	lmInFovIdx = rfsMeasurementLogLikelihood(particle, 0);
	for (auto &i:lmInFovIdx){
		removeLM[i]=0;
	}
	TimeStamp dT;
	for (int k = 1; k < particle.trajectory.size(); k++) {

		lmInFovIdx = rfsMeasurementLogLikelihood(particle, k);

		for (auto &i:lmInFovIdx){
			removeLM[i]=0;
		}

		dT = time_[k] - time_[k - 1];
		typename TPose::Vec pose_process_gradient;
		double processlikelihood = robotProcessModelPtr_->logLikelihood(particle.trajectory[k], particle.trajectory[k - 1], inputs_[k - 1], dT, &pose_process_gradient);

		particle.currentLikelihood += processlikelihood;
		particle.trajectory_gradient[k] -= pose_process_gradient;
		particle.trajectory_gradient[k - 1] += pose_process_gradient;

	}
	if (particle.currentLikelihood > particle.bestLikelihood){
		particle.bestTrajectory = particle.trajectory;
		particle.bestLikelihood = particle.currentLikelihood;
		particle.bestLandmarks = particle.landmarks;
	}

	/*
	std::cout << "traj grad:  ";
	for(auto grad:particle.trajectory_gradient){
		std::cout << grad << "   ";
	}
	std::cout << "\n";

	std::cout << "lm grad:  ";
	for(auto grad:particle.landmarks_gradient){
		std::cout << grad << "   ";
	}
	std::cout << "\n";
	//std::cout << "likelihood   " << l << "\n";
	*/
	return removeLM;
}

template<class RobotProcessModel, class MeasurementModel>
std::vector<unsigned int>  RFSHMCSLAM<RobotProcessModel, MeasurementModel>::rfsMeasurementLogLikelihood(TParticle &particle, const int k) {

	//std::cout << "LIKELY -----------------------------------------------------\n\n\n";

	assert(std::isfinite(particle.trajectory[k][0] ));
	const TPose &pose = particle.trajectory[k];
	const int nZ = this->Z_[k].size();
	const unsigned int mapSize = particle.landmarks.size();

	assert(pose.getPos()[0] == pose.getPos()[0]);

	// Find map points within field of view and their probability of detection
	std::vector<unsigned int> lmInFovIdx;
	lmInFovIdx.reserve(mapSize);
	std::vector<double> lmInFovPd;
	std::vector<typename TLandmark::Vec> lmInFovGrad;
	lmInFovPd.reserve(mapSize);
	std::vector<int> landmarkCloseToSensingLimit;
	landmarkCloseToSensingLimit.reserve(mapSize);

	for (unsigned int m = 0; m < mapSize; m++) {

		bool isCloseToSensingLimit = false;


		TLandmark lm(particle.landmarks[m],MeasurementModel::TLandmark::Mat::Zero());
		typename TLandmark::Vec zerograd;
		zerograd.setZero();
		bool isClose;
		double Pd = this->mModelPtr_->probabilityOfDetection(pose, lm, isCloseToSensingLimit);

		if (Pd > 0) {
			lmInFovIdx.push_back(m);
			lmInFovPd.push_back(Pd);
			lmInFovGrad.push_back(zerograd);
			landmarkCloseToSensingLimit.push_back(isCloseToSensingLimit);
		}

	}

	const unsigned int nM = lmInFovIdx.size();

	// If map is empty everything must be a false alarm
	double clutter[nZ];
	for (int n = 0; n < nZ; n++) {

			double lclut = log(this->mModelPtr_->clutterIntensity(this->Z_[k][n], nZ));
			clutter[n] = lclut < BIG_NEG_NUM ? BIG_NEG_NUM : lclut;


	}

	if (nM == 0) {
		double l = 0;
		for (int n = 0; n < nZ; n++) {
			l += clutter[n];
		}
		return lmInFovIdx;
	}

	TLandmark* evalPt;
	TLandmark evalPt_copy;
	TMeasurement expected_z;

	double md2; // Mahalanobis distance squared

	// Create and fill in likelihood table (nM x nZ)
	double** L;
	CostMatrixGeneral likelihoodMatrix(L, nM, nZ);
	likelihoodMatrix.MIN_LIKELIHOOD = BIG_NEG_NUM;

	Eigen::Matrix<double, MeasurementModel::TMeasurement::Vec::RowsAtCompileTime, MeasurementModel::TPose::Vec::RowsAtCompileTime> jacobian_wrt_pose;
	Eigen::Matrix<double, MeasurementModel::TMeasurement::Vec::RowsAtCompileTime, MeasurementModel::TLandmark::Vec::RowsAtCompileTime> jacobian_wrt_lmk;

	std::unordered_map<int, typename MeasurementModel::TLandmark::Vec> landmark_gradients;
	std::unordered_map<int, typename MeasurementModel::TPose::Vec> pose_gradients;

	//std::cout << "L: \n";
	for (int m = 0; m < nM; m++) {


		evalPt_copy.set(particle.landmarks[lmInFovIdx[m]],MeasurementModel::TLandmark::Mat::Zero()); // so that we don't change the actual data //


		this->mModelPtr_->measure(pose, evalPt_copy, expected_z, &(jacobian_wrt_lmk), &(jacobian_wrt_pose)); // get expected measurement for m
		double Pd = lmInFovPd[m]; // get the prob of detection of m

		for (int n = 0; n < nZ; n++) {

			typename MeasurementModel::TMeasurement::Vec n_error;
			// calculate measurement likelihood with detection statistics

			L[m][n] = expected_z.evalGaussianLogLikelihood(this->Z_[k][n], n_error, &md2) + log(Pd) - log(1 - Pd) - clutter[n]; // new line
			if (L[m][n] < config.MeasurementLikelihoodThreshold_) {
				L[m][n] = BIG_NEG_NUM;
				continue;
			}

			typename MeasurementModel::TLandmark::Vec lm_grad;
			typename MeasurementModel::TPose::Vec pose_grad;
			lm_grad = jacobian_wrt_lmk.transpose() * n_error;
			pose_grad = jacobian_wrt_pose.transpose() * n_error;
			//std::cout << "pose: " << pose[0] << "  lm:  " <<  evalPt_copy[0] << "  Z:  " << this->Z_[k][n][0] << "  posegrad:  " << pose_grad[0]  << " lmgrad: " << lm_grad << "\n";

			landmark_gradients.insert(std::make_pair(m * nZ + n, lm_grad));
			pose_gradients.insert(std::make_pair(m * nZ + n, pose_grad));

			//std::cout << L[m][n] << "  ";

		}
		//std::cout << "\n";
	}

	// Partition the Likelihood Table and turn into a log-likelihood table
	int nP = likelihoodMatrix.partition();
	double l = 0;

	// Go through each partition and determine the likelihood
	for (int p = 0; p < nP; p++) {

		double partition_log_likelihood = 0;

		unsigned int nCols, nRows;
		double** Cp;
		unsigned int* rowIdx;
		unsigned int* colIdx;
		Eigen::Matrix<double, PoseDim, 1> partition_pose_gradient;
		partition_pose_gradient.setZero();

		bool isZeroPartition = !likelihoodMatrix.getPartitionSize(p, nRows, nCols);
		bool useMurtyAlgorithm = true;

		isZeroPartition = !likelihoodMatrix.getPartition(p, Cp, nRows, nCols, rowIdx, colIdx, useMurtyAlgorithm);

		for (int r = 0; r < nRows; r++) {
			partition_log_likelihood += log(lmInFovPd[rowIdx[r]]) < BIG_NEG_NUM ? BIG_NEG_NUM : log(lmInFovPd[rowIdx[r]]);
		}

		for (int c = 0; c < nCols; c++) {
			partition_log_likelihood += clutter[colIdx[c]] < BIG_NEG_NUM ? BIG_NEG_NUM : clutter[colIdx[c]];
		}
		//std::cout << "partition_log_likelihood" <<partition_log_likelihood << "\n";

		if (isZeroPartition) { // all landmarks in this partition are mis-detected. All measurements are outliers

			// This likelihood is already there
		} else {

			//  fill in the extended part of the partition

			for (int r = 0; r < nRows; r++) {
				for (int c = 0; c < nCols; c++) {

					if (Cp[r][c] < BIG_NEG_NUM)
						Cp[r][c] = BIG_NEG_NUM;

				}
			}

			if (useMurtyAlgorithm) { // use Murty's algorithm

				// mis-detections
				for (int r = 0; r < nRows; r++) {
					for (int c = nCols; c < nRows + nCols; c++) {
						if (r == c - nCols)
							Cp[r][c] = 0;
						else
							Cp[r][c] = BIG_NEG_NUM;
					}
				}

				// clutter
				for (int r = nRows; r < nRows + nCols; r++) {
					for (int c = 0; c < nCols; c++) {
						if (r - nRows == c)
							Cp[r][c] = 0;
						else
							Cp[r][c] = BIG_NEG_NUM;
						if (Cp[r][c] < BIG_NEG_NUM)
							Cp[r][c] = BIG_NEG_NUM;
					}
				}

				// the lower right corner
				for (int r = nRows; r < nRows + nCols; r++) {
					for (int c = nCols; c < nRows + nCols; c++) {
						Cp[r][c] = 0;
					}
				}

				Murty murtyAlgo(Cp, nRows + nCols);
				Murty::Assignment a, first_a;

				double permutation_log_likelihood = 0;
				double first_permutation_log_likelihood = 0;
				double permutation_likelihood = 0;
				murtyAlgo.setRealAssignmentBlock(nRows, nCols);
				//find the best assignment
				int rank = murtyAlgo.findNextBest(a, first_permutation_log_likelihood);
				if (rank == -1 || first_permutation_log_likelihood < BIG_NEG_NUM) {
					std::cerr << "First association is zero!!\n";
				}
				first_a = a;



				//std::cout << "first perm loglike: " <<first_permutation_log_likelihood<<"\n";
				partition_log_likelihood += first_permutation_log_likelihood;

				double partition_correction_component = 1; //< likelihood due to other data associations
				for (int k = 0; k < 200; k++) {
					int rank = murtyAlgo.findNextBest(a, permutation_log_likelihood);
					//std::cout << "permutation loglike: " << permurandomGenerators_[threadnum]tation_log_likelihood <<"\n";
					if (rank == -1 || permutation_log_likelihood < BIG_NEG_NUM)
						break;

					double permutation_correction_component = exp(permutation_log_likelihood - first_permutation_log_likelihood);
					partition_correction_component += permutation_correction_component;

					// find the gradients

					for (int r = 0; r < nRows; r++) {
						if (a[r] != first_a[r]) {

							typename MeasurementModel::TLandmark::Vec lm_grad;
							typename MeasurementModel::TPose::Vec pose_grad;
							lm_grad.setZero();
							pose_grad.setZero();
							if (a[r] < nCols) {
								lm_grad += landmark_gradients.at(rowIdx[r] * nZ + colIdx[a[r]]);
								pose_grad += pose_gradients.at(rowIdx[r] * nZ + colIdx[a[r]]);

							}
							if (first_a[r] < nCols) {
								pose_grad -= pose_gradients.at(rowIdx[r] * nZ + colIdx[first_a[r]]);
								lm_grad -= landmark_gradients.at(rowIdx[r] * nZ + colIdx[first_a[r]]);
							}
							lmInFovGrad[rowIdx[r]] += lm_grad * permutation_correction_component;
							partition_pose_gradient += pose_grad * permutation_correction_component;
						}
					}

				}
				// normalize the gradient corrections
				for (int r = 0; r < nRows; r++) {
					lmInFovGrad[rowIdx[r]] /= partition_correction_component;

				}
				partition_pose_gradient/= partition_correction_component;

				// find the gradients
				if (first_permutation_log_likelihood > BIG_NEG_NUM) {
					for (int r = 0; r < nRows; r++) {
						if (first_a[r] < nCols) {

							typename MeasurementModel::TLandmark::Vec lm_grad = landmark_gradients.at(rowIdx[r] * nZ + colIdx[first_a[r]]);
							typename MeasurementModel::TPose::Vec pose_grad = pose_gradients.at(rowIdx[r] * nZ + colIdx[first_a[r]]);
							lmInFovGrad[rowIdx[r]] += lm_grad;
							partition_pose_gradient += pose_grad;
						}
					}
				}

				//std::cout << "partition correction : " << partition_correction_component<< "\n";

				particle.trajectory_gradient[k] -= partition_pose_gradient;
				partition_log_likelihood += log(partition_correction_component);
				//std::cout << "partition partition_log_likelihood : " << partition_log_likelihood<< "\n";

			} else { // use lexicographic ordering

				std::cerr << "CANNOT USE LEXICOGRAPHICAL ORDER \n\n\n";
				// CANNOT USE LEXICOGRAPHICAL ORDER
			} // End lexicographic ordering

		} // End non zero partition

		// normalize landmark gradients

		for (int lmk = 0; lmk < lmInFovGrad.size(); lmk++) {
			assert(lmInFovGrad[lmk] == lmInFovGrad[lmk]);
			particle.landmarks_gradient[lmInFovIdx[lmk]] -= lmInFovGrad[lmk] ;
		}

		l += partition_log_likelihood;
		//std::cout << " partition loglike: " << partition_log_likelihood << "\n";
		//std::cout << " partial: " << l << "\n";

	} // End partitions

	particle.currentLikelihood += l - this->mModelPtr_->clutterIntensityIntegral(nZ);
	return lmInFovIdx;
}

template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::momentumHalfStep(TParticle &particle){

	for(int i=0; i<particle.landmarks.size(); i++){
		particle.landmarks_momentum[i] -= 0.5*particle.epsilon*particle.landmarks_gradient[i];

	}
	for(int i=0; i< particle.trajectory.size() ;  i++){
		particle.trajectory_momentum[i] -= 0.5*particle.epsilon * particle.trajectory_gradient[i];
	}

}
template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::stateFullStep(TParticle &particle){

	for(int i=0; i<particle.landmarks.size(); i++){
		particle.landmarks[i] = particle.landmarks[i]+ particle.epsilon*particle.landmarks_momentum[i]/config.m;
	}
	for(int i=0; i< particle.trajectory.size() ;  i++){
		particle.trajectory[i] = particle.trajectory[i]+ particle.epsilon * particle.trajectory_momentum[i]/config.m;
	}
}

template<class RobotProcessModel, class MeasurementModel>
double RFSHMCSLAM<RobotProcessModel, MeasurementModel>::hamiltonian(const TParticle& particle) {

	double hamiltonian=-particle.currentLikelihood/config.temp;

	for(auto &p:particle.trajectory_momentum){
		hamiltonian+=0.5*p.squaredNorm()/config.m;
	}
	for(auto &p:particle.landmarks_momentum){
			hamiltonian+=0.5*p.squaredNorm()/config.m;
		}
	return hamiltonian;
}
template<class RobotProcessModel, class MeasurementModel>
typename RFSHMCSLAM<RobotProcessModel, MeasurementModel>::TParticle RFSHMCSLAM<RobotProcessModel, MeasurementModel>::reversibleJumpHMC(TParticle& particle){
	int threadnum=0;
#ifdef _OPENMP
      threadnum = omp_get_thread_num();
#endif

    	boost::uniform_real<> uni_dist(0,1);
    	double r=uni_dist(randomGenerators_[threadnum]);
    	if (r<config.Pb){
    		TParticle particle_out =particle;
    		birthMove(particle_out);
    		return particle_out;
    	}

    	if (particle.landmarks.size()>0 && r<config.Pb+config.Pd){
    		TParticle particle_out =particle;
    		deathMove(particle_out);
    		return particle_out;
    	}
    	return basicHamiltonianMCMC(particle);

}

template<class RobotProcessModel, class MeasurementModel>
typename RFSHMCSLAM<RobotProcessModel, MeasurementModel>::TParticle RFSHMCSLAM<RobotProcessModel, MeasurementModel>::basicHamiltonianMCMC(TParticle& particle){


	int threadnum=0;
#ifdef _OPENMP
      threadnum = omp_get_thread_num();
#endif

	resampleMomentum(particle);
	TParticle particle_out =particle;


	boost::uniform_int<> uni_int(1, config.K);
	if(leapFrog(particle_out, uni_int(randomGenerators_[threadnum]))){
	boost::uniform_real<> uni_dist(0,1);




	double p = std::exp(hamiltonian(particle)-hamiltonian(particle_out));
	//std::cout << "p:   " <<p  << " h1 :" << hamiltonian(particle) << "  h2:  " <<hamiltonian(particle_out) <<"\n";
	if (uni_dist(randomGenerators_[threadnum])<p){
		renormalize(particle_out);
		particle_out.n_accept++;
		if (particle_out.n_accept + particle_out.n_reject >= 10) {
			if (particle_out.n_accept <= 2) {
				particle_out.epsilon *= 0.8;
			}
			if (particle_out.n_reject <= 2) {
				particle_out.epsilon *= 1.2;
			}
			particle_out.n_accept=0;
			particle_out.n_reject=0;
			//std::cout <<"eps: " << particle_out.epsilon << "\n";
		}

		std::vector<unsigned int> removeLM = rfsMeasurementLogLikelihood(particle_out);
/*
		if (removeLM.size()>0){
		for(int i=0; i< removeLM.size() ; i++){
			std::cout << particle_out.landmarks[i] << "  " << removeLM[i] << "  ";
		}
		std::cout << "\n";
		}*/
		int i=0,j=removeLM.size()-1 ;
		while ( i <= j ){
			while (i <= j  && removeLM[j]==1) j--;
			while (i <= j && removeLM[i]==0) i++;
			if (i<j){

				particle_out.landmarks[i] = particle_out.landmarks[j];
				particle_out.landmarks_gradient[i] = particle_out.landmarks_gradient[j];
				particle_out.landmarks_momentum[i] = particle_out.landmarks_momentum[j];
				removeLM[i]=0;
				removeLM[j]=1;


			}
		}
		if(j!=removeLM.size()-1) std::cout << "removed !\n";
		if (j<-1) j=-1;
		particle_out.landmarks.resize(j+1);
		particle_out.landmarks_gradient.resize(j+1);
		particle_out.landmarks_momentum.resize(j+1);


		//std::cout <<"eps: " << particle_out.epsilon << "\n";
		return particle_out;

	}
	}
	particle.n_reject++;
	if (particle.n_accept + particle.n_reject >= 10) {
		if (particle.n_accept <= 2) {
			particle.epsilon *= 0.8;
		}
		if (particle.n_reject <= 2) {
			particle.epsilon *= 1.2;
		}
		particle.n_accept=0;
		particle.n_reject=0;
		//std::cout <<"eps: " << particle.epsilon << "\n";
	}
	//std::cout <<"eps: " << particle.epsilon << "\n";
	return particle;
}
template<class RobotProcessModel, class MeasurementModel>
bool RFSHMCSLAM<RobotProcessModel, MeasurementModel>::leapFrog(TParticle& particle, int n) {
	rfsMeasurementLogLikelihood(particle);
	for (int i = 0; i < n; i++) {

		momentumHalfStep(particle);
		stateFullStep(particle);


		if(!isFinite(particle)){
			std::cout << "inf\n";
			return false;
		}
		rfsMeasurementLogLikelihood(particle);
		momentumHalfStep(particle);
	}

return true;

}
template<class RobotProcessModel, class MeasurementModel>
inline void rfs::RFSHMCSLAM<RobotProcessModel, MeasurementModel>::renormalize(TParticle& particle) {


	auto initPose=particle.trajectory[0];
	for(int i=0; i<particle.landmarks.size(); i++){

		particle.landmarks[i] -= initPose;

	}

	for(int i=0; i< particle.trajectory.size() ;  i++){

			particle.trajectory[i] -= initPose;

	}
}
template<class RobotProcessModel, class MeasurementModel>
inline void rfs::RFSHMCSLAM<RobotProcessModel, MeasurementModel>::deathMove(TParticle& particle) {

	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
	threadnum = omp_get_thread_num();
#endif
	//remove a random landmark
	double accept=0;
	double prevloglike=particle.currentLikelihood;
	if (particle.landmarks.size()>0 ) {

		boost::uniform_int<> uni_int_m(0, particle.landmarks.size()-1);
		int m = uni_int_m(randomGenerators_[threadnum]);
		double qbirth=0;
		int numZ=0;
		for(int k=0; k < Z_.size(); k++){
			for(int nz=0; nz < Z_[k].size(); nz++){
				TLandmark lm;
						this->mModelPtr_->inverseMeasure(particle.trajectory[k], Z_[k][nz], lm);
						qbirth+=lm.evalGaussianLikelihood(particle.landmarks[m]);
						numZ++;
			}
		}
		qbirth/=numZ;
		double qdeath=1.0/particle.landmarks.size();



		particle.landmarks[m] = particle.landmarks.back();
		typename TLandmark::Vec lm=particle.landmarks.back();
		particle.landmarks.pop_back();
		particle.landmarks_momentum.resize(particle.landmarks.size());
		particle.landmarks_gradient.resize(particle.landmarks.size());
		rfsMeasurementLogLikelihood(particle);
		accept= exp(particle.currentLikelihood-prevloglike)*config.Pb*qbirth/(qdeath*config.Pd);
		if(uni_dist(randomGenerators_[threadnum]) > accept){
			particle.landmarks.push_back(lm);
		}

	}



	particle.landmarks_momentum.resize(particle.landmarks.size());
	particle.landmarks_gradient.resize(particle.landmarks.size());

}

template<class RobotProcessModel, class MeasurementModel>
inline void rfs::RFSHMCSLAM<RobotProcessModel, MeasurementModel>::birthMove(TParticle& particle) {


	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
	threadnum = omp_get_thread_num();
#endif

	double accept=0;
	double prevloglike=particle.currentLikelihood;

	// create a new random landmark

		boost::uniform_int<> uni_int_k(0, Z_.size() - 1);
		int k = uni_int_k(randomGenerators_[threadnum]);
		boost::uniform_int<> uni_int_nz(0, Z_[k].size() - 1);
		int nz = uni_int_nz(randomGenerators_[threadnum]);

		TLandmark lm;
		this->mModelPtr_->inverseMeasure(particle.trajectory[k], Z_[k][nz], lm);
		lm.sample(lm);
		particle.landmarks.push_back(lm.get());

		double qbirth = 0;
		int numZ = 0;
		for (int k = 0; k < Z_.size(); k++) {
			for (int nz = 0; nz < Z_[k].size(); nz++) {
				TLandmark lm_z;
				this->mModelPtr_->inverseMeasure(particle.trajectory[k], Z_[k][nz], lm_z);
				qbirth += lm_z.evalGaussianLikelihood(particle.landmarks.back());
				numZ++;
			}
		}
		qbirth /= numZ;
		double qdeath = 1.0 / particle.landmarks.size();
		particle.landmarks_momentum.resize(particle.landmarks.size());
		particle.landmarks_gradient.resize(particle.landmarks.size());

		rfsMeasurementLogLikelihood(particle);
		accept = exp((particle.currentLikelihood - prevloglike)/config.temp) * qdeath *config.Pd/ (qbirth*config.Pb);

		if (uni_dist(randomGenerators_[threadnum]) > accept) {

			particle.landmarks.pop_back();
		}else{

		}


	particle.landmarks_momentum.resize(particle.landmarks.size());
	particle.landmarks_gradient.resize(particle.landmarks.size());

}

template<class RobotProcessModel, class MeasurementModel>
inline void rfs::RFSHMCSLAM<RobotProcessModel, MeasurementModel>::birthDeathStep(TParticle& particle) {

	boost::uniform_real<> uni_dist(0, 1);
	int threadnum = 0;
#ifdef _OPENMP
	threadnum = omp_get_thread_num();
#endif
	//remove a random landmark
	double accept=0;
	double prevloglike=particle.currentLikelihood;
	if (particle.landmarks.size()>0 && uni_dist(randomGenerators_[threadnum]) < config.mapFromMeasurementProb_) {

		boost::uniform_int<> uni_int_m(0, particle.landmarks.size()-1);
		int m = uni_int_m(randomGenerators_[threadnum]);
		double qbirth=0;
		int numZ=0;
		for(int k=0; k < Z_.size(); k++){
			for(int nz=0; nz < Z_[k].size(); nz++){
				TLandmark lm;
						this->mModelPtr_->inverseMeasure(particle.trajectory[k], Z_[k][nz], lm);
						qbirth+=lm.evalGaussianLikelihood(particle.landmarks[m]);
						numZ++;
			}
		}
		qbirth/=numZ;
		double qdeath=1.0/particle.landmarks.size();



		particle.landmarks[m] = particle.landmarks.back();
		typename TLandmark::Vec lm=particle.landmarks.back();
		particle.landmarks.pop_back();
		rfsMeasurementLogLikelihood(particle);
		accept= exp((particle.currentLikelihood-prevloglike)/config.temp)*qbirth/qdeath;
		if(uni_dist(randomGenerators_[threadnum]) > accept){
			particle.landmarks.push_back(lm);
		}

	}

	// create a new random landmark
	if (uni_dist(randomGenerators_[threadnum]) < config.mapFromMeasurementProb_) {

		boost::uniform_int<> uni_int_k(0, Z_.size() - 1);
		int k = uni_int_k(randomGenerators_[threadnum]);
		boost::uniform_int<> uni_int_nz(0, Z_[k].size() - 1);
		int nz = uni_int_nz(randomGenerators_[threadnum]);

		TLandmark lm;
		this->mModelPtr_->inverseMeasure(particle.trajectory[k], Z_[k][nz], lm);
		lm.sample(lm);
		particle.landmarks.push_back(lm.get());

		double qbirth = 0;
		int numZ = 0;
		for (int k = 0; k < Z_.size(); k++) {
			for (int nz = 0; nz < Z_[k].size(); nz++) {
				TLandmark lm_z;
				this->mModelPtr_->inverseMeasure(particle.trajectory[k], Z_[k][nz], lm_z);
				qbirth += lm_z.evalGaussianLikelihood(particle.landmarks.back());
				numZ++;
			}
		}
		qbirth /= numZ;
		double qdeath = 1.0 / particle.landmarks.size();
		rfsMeasurementLogLikelihood(particle);
		accept = exp(particle.currentLikelihood - prevloglike) * qdeath / qbirth;
		if (uni_dist(randomGenerators_[threadnum]) > accept) {
			particle.landmarks.pop_back();
		}

	}

	particle.landmarks_momentum.resize(particle.landmarks.size());
	particle.landmarks_gradient.resize(particle.landmarks.size());
	particle.bestLandmarks_momentum.resize(particle.landmarks.size());
	particle.bestLandmarks = particle.landmarks;

}

}
template<class RobotProcessModel, class MeasurementModel>
inline void rfs::RFSHMCSLAM<RobotProcessModel, MeasurementModel>::resampleMomentum(TParticle& particle) {

	int threadnum=0;
#ifdef _OPENMP
      threadnum = omp_get_thread_num();
#endif

	for(int i=0; i<particle.landmarks.size(); i++){
		for(int j = 0; j< LandmarkDim ; j++){
			particle.landmarks_momentum[i][j] = gaussianGenerators_[threadnum](randomGenerators_[threadnum]);
		}
	}
	for(int i=0; i< particle.trajectory.size() ;  i++){
		for(int j = 0; j< PoseDim ; j++){
			particle.trajectory_momentum[i][j] = gaussianGenerators_[threadnum](randomGenerators_[threadnum]);
		}
	}
}

template<class RobotProcessModel, class MeasurementModel>
inline bool rfs::RFSHMCSLAM<RobotProcessModel, MeasurementModel>::isFinite(
		TParticle& particle) {
	for(auto &pose:particle.trajectory){
		if(! pose.allFinite()){
			return false;
		}
	}

	for(auto &landmark:particle.landmarks){
		if (! landmark.allFinite()){
			return false;
		}
	}
	for (auto &g:particle.landmarks_gradient) {
		if (! g.allFinite()) return false;
	}
	for (auto &g:particle.trajectory_gradient) {
		if (! g.allFinite()) return false;
	}
	for (auto &m:particle.landmarks_momentum) {
		if (! m.allFinite()) return false;
	}
	for (auto &m:particle.trajectory_momentum) {
		if (! m.allFinite()) return false;
	}
	return true;
}

#endif
