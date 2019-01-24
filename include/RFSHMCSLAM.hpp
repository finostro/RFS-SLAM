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

		double ospa_c_; /**< ospa-like c value for calculating set differences and speed*/

		/** The threshold used to determine if a possible meaurement-landmark
		 *  pairing is significant to worth considering
		 */
		double MeasurementLikelihoodThreshold_;

		double mapFromMeasurementProb_; /**< probability that each measurement will initialize a landmark on map initialization*/

		double m; /**<  Constant mass, could be replaced by a (possibly diagonal) matrix*/

		double phi_p; /**<  particle minimum influence   */

		double card_phi_p; /**<  particle minimum influence on cardinality  */

		double phi_g; /**< global minimum influence  */

		double card_phi_g; /**< global minimum influence on cardinality */

		int K; /**< Number of timesteps to integrate with leapfrog */

		double epsilon; /**< integration time for leapfrog simulation */

		bool use_global; /**< use global  topology*/

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
	initTrajectories();

	/**
	 * Generate random maps based on the already initialized trajectories
	 */
	void
	initMaps();

	/**
	 * initialize the particles
	 */
	void
	init();

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
	 * @return the measurement likelihood
	 */

	double
	rfsMeasurementLogLikelihood(TParticle &particle, const int k);

	/**
	 * Calculates the measurement likelihood of particle  including all available times stores the likelihood value and gradients on the particle object.
	 * @param particle[in,out] The particle, trajectory and map,
	 * @return the measurement likelihood
	 */

	double
	rfsMeasurementLogLikelihood(TParticle &particle);

	/**
	 * Evaluate the current likelihood of all the particles
	 */
	void
	evaluateLikelihoods(std::vector<TParticle> &particles);
	/**
	 * Run the leapFrog algorithm n times on a particle
	 * @param[in] particle  input particle to start Hamiltonian Simulation
	 * @param[in] n number of leapfrog iterations to run
	 * @return
	 */
	TParticle leapFrog(const TParticle &particle, int n);

	/***
	 * Resample random velocities with  Gaussian distribution.
	 * @param particle[in,out] The particle, whose velocity is changed
	 */
	void
	resampleLandmarkVelocity(TParticle &particle);

	MeasurementModel *mModelPtr_;
	RobotProcessModel *robotProcessModelPtr_;
private:

	int nThreads_; /**< Number of threads  */
	int iteration_;
	std::vector<TInput> inputs_; /**< vector containing all odometry inputs */
	std::vector<std::vector<TMeasurement> > Z_; /**< vector containing all feature measurements */
	std::vector<std::vector<int> > DA_; /**< vector containing available data associations, -1 meaning unknown data association, -2 known to be false alarm */
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
		if (maxlikelihood > particles[i].currentLikelihood) {
			maxi = i;
			maxlikelihood = particles[i].currentLikelihood;
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
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::init() {
	particles_.resize(config.nParticles_);

	initTrajectories();
	initMaps();
}

template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::initTrajectories() {

	for (int i = 0; i < config.nParticles_; i++) {

		particles_[i].trajectory.resize(inputs_.size() + 1);
		particles_[i].trajectory_velocity.resize(inputs_.size() + 1);
		particles_[i].inputs.resize(inputs_.size());
		particles_[i].inputs_velocity.resize(inputs_.size());
		for (int k = 0; k < inputs_.size(); k++) {
			TimeStamp dT = inputs_[k].getTime() - particles_[i].trajectory[k].getTime();
			robotProcessModelPtr_->sample(particles_[i].trajectory[k + 1], particles_[i].trajectory[k], inputs_[k], dT, false, true, &particles_[i].inputs[k]);
		}
		particles_[i].bestTrajectory = particles_[i].trajectory;
		particles_[i].bestTrajectory_velocity = particles_[i].trajectory_velocity;
		particles_[i].bestInputs = particles_[i].inputs;

		particles_[i].bestInputs_velocity.resize(inputs_.size());

	}
}
template<class RobotProcessModel, class MeasurementModel>
void RFSHMCSLAM<RobotProcessModel, MeasurementModel>::initMaps() {

	for (int i = 0; i < config.nParticles_; i++) {

		for (int k = 0; k < particles_[i].trajectory.size(); k++) {
			for (int nz = 0; nz < Z_[k].size(); nz++) {

				if (drand48() < config.mapFromMeasurementProb_) {

					TLandmark lm;
					this->mModelPtr_->inverseMeasure(particles_[i].trajectory[k], Z_[k][nz], lm);
					particles_[i].landmarks.push_back(lm);
				}
			}
		}

		particles_[i].landmarks_velocity.resize(particles_[i].landmarks.size());
		particles_[i].bestLandmarks_velocity.resize(particles_[i].landmarks.size());
		particles_[i].bestLandmarks = particles_[i].landmarks;

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
double RFSHMCSLAM<RobotProcessModel, MeasurementModel>::rfsMeasurementLogLikelihood(TParticle &particle) {
	clear(particle);
	double l = rfsMeasurementLogLikelihood(particle, 0);
	TimeStamp dT;
	for (int k = 1; k < particle.trajectory.size(); k++) {

		l += rfsMeasurementLogLikelihood(particle, k);
		dT = time_[k] - time_[k - 1];
		typename TPose::Vec pose_process_gradient;
		l += robotProcessModelPtr_->logLikelihood(particle.trajectory[k], particle.trajectory[k - 1], inputs_[k - 1], dT, &pose_process_gradient);
		particle.trajectory_gradient[k] += pose_process_gradient;
		particle.trajectory_gradient[k - 1] -= pose_process_gradient;
	}
	//std::cout << "likelihood   " << l << "\n";
	return l;
}

template<class RobotProcessModel, class MeasurementModel>
double RFSHMCSLAM<RobotProcessModel, MeasurementModel>::rfsMeasurementLogLikelihood(TParticle &particle, const int k) {

	//std::cout << "LIKELY -----------------------------------------------------\n\n\n";

	assert(particle.trajectory_[k].getPos()[0] == particle.trajectory_[k].getPos()[0]);
	const TPose &pose = particle.trajectory_[k];
	const int nZ = this->Z_[k].size();
	const unsigned int mapSize = particle.landmarks_.size();

	assert(pose.getPos()[0] == pose.getPos()[0]);

	// Find map points within field of view and their probability of detection
	std::vector<unsigned int> lmInFovIdx;
	lmInFovIdx.reserve(mapSize);
	std::vector<double> lmInFovPd;
	std::vector<TLandmark::Vec> lmInFovGrad;
	lmInFovPd.reserve(mapSize);
	std::vector<int> landmarkCloseToSensingLimit;
	landmarkCloseToSensingLimit.reserve(mapSize);

	for (unsigned int m = 0; m < mapSize; m++) {

		bool isCloseToSensingLimit = false;

		const TLandmark &lm = particle.landmarks[m];
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
		if (DA_[k][n] == -1) {
			double lclut = log(this->mModelPtr_->clutterIntensity(this->Z_[k][n], nZ));
			clutter[n] = lclut < BIG_NEG_NUM ? BIG_NEG_NUM : lclut;
		} else if (DA_[k][n] == -2)
			clutter[n] = 0;
		else
			clutter[n] = BIG_NEG_NUM;

	}

	if (nM == 0) {
		double l = 0;
		for (int n = 0; n < nZ; n++) {
			l += clutter[n];
		}
		return l - this->mModelPtr_->clutterIntensityIntegral(nZ);
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

		evalPt = &particle.landmarks[lmInFovIdx[m]]; // get location of m
		evalPt_copy = *evalPt; // so that we don't change the actual data //
		evalPt_copy.setCov(MeasurementModel::TLandmark::Mat::Zero()); //

		this->mModelPtr_->measure(pose, evalPt_copy, expected_z, &(jacobian_wrt_lmk), &(jacobian_wrt_pose)); // get expected measurement for m
		double Pd = lmInFovPd[m]; // get the prob of detection of m

		for (int n = 0; n < nZ; n++) {
			if (DA_[k][n] != m && DA_[k][n] != -1) { // data association exists and this is not it
				L[m][n] = BIG_NEG_NUM;
				continue;
			}
			typename MeasurementModel::TMeasurement::Vec n_error;
			// calculate measurement likelihood with detection statistics
			if (DA_[k][n] >= 0)
				L[m][n] = expected_z.evalGaussianLogLikelihood(this->Z_[k][n], n_error, &md2) + log(Pd) - log(1 - Pd) - clutter[n]; // new line
			else
				L[m][n] = expected_z.evalGaussianLogLikelihood(this->Z_[k][n], n_error, &md2) + log(Pd) - log(1 - Pd) - clutter[n]; // new line
			if (L[m][n] < config.MeasurementLikelihoodThreshold_) {
				L[m][n] = BIG_NEG_NUM;
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
					//std::cout << "permutation loglike: " << permutation_log_likelihood <<"\n";
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

				particle.trajectory_gradient[k] += partition_pose_gradient;
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
			particle.landmarks_gradient[lmInFovIdx[lmk]] += lmInFovGrad[lmk] ;
		}

		l += partition_log_likelihood;
		//std::cout << " partition loglike: " << partition_log_likelihood << "\n";
		//std::cout << " partial: " << l << "\n";

	} // End partitions

	return l - this->mModelPtr_->clutterIntensityIntegral(nZ);
}

}

#endif
