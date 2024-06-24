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

#include <filesystem>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "ProcessModel_Odometry1D.hpp"
#include "RFSHMCSLAM.hpp"
#include "measurement_models/MeasurementModel_Rng1D.hpp"
#include "external/argparse.hpp"
#include <stdio.h>
#include <string>
#include <sys/ioctl.h>

#ifdef _PERFTOOLS_CPU
#include <gperftools/profiler.h>
#endif
#ifdef _PERFTOOLS_HEAP
#include <gperftools/heap-profiler.h>
#endif

using namespace rfs;

/**
 * \class Simulator_RFSPSOSLAM_1d
 * \brief A 1d SLAM Simulator using PSO
 * \author Felipe Inostroza
 */
class Simulator_RFSPSOSLAM_1d {

public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	typedef RFSHMCSLAM<MotionModel_Odometry1d, MeasurementModel_Rng1D>::TParticle TParticle;

	Simulator_RFSPSOSLAM_1d() {
		hmcslam_ = new RFSHMCSLAM<MotionModel_Odometry1d, MeasurementModel_Rng1D>();
	}

	~Simulator_RFSPSOSLAM_1d() {

		if (hmcslam_ != NULL) {
			delete hmcslam_;
		}

	}

	/** Read the simulator configuration file */
	bool readConfigFile(const char* fileName) {

		cfgFileName_ = fileName;

		boost::property_tree::ptree pt;
		boost::property_tree::xml_parser::read_xml(fileName, pt);

		logToFile_ = false;
		if (pt.get("config.logging.logToFile", 0) == 1)
			logToFile_ = true;
		logDirPrefix_ = pt.get<std::string>("config.logging.logDirPrefix", "./");

		kMax_ = pt.get<int>("config.timesteps");
		dT_ = pt.get<double>("config.sec_per_timestep");
		dTimeStamp_ = TimeStamp(dT_);

		nSegments_ = pt.get<int>("config.trajectory.nSegments");
		max_dx_ = pt.get<double>("config.trajectory.max_dx_per_sec");
		min_dx_ = pt.get<double>("config.trajectory.min_dx_per_sec");
		vardx_ = pt.get<double>("config.trajectory.vardx");

		nLandmarks_ = pt.get<int>("config.landmarks.nLandmarks");

		rangeLimitMax_ = pt.get<double>("config.measurements.rangeLimitMax");
		rangeLimitMin_ = pt.get<double>("config.measurements.rangeLimitMin");
		rangeLimitBuffer_ = pt.get<double>("config.measurements.rangeLimitBuffer");
		Pd_ = pt.get<double>("config.measurements.probDetection");
		c_ = pt.get<double>("config.measurements.clutterIntensity");
		varzr_ = pt.get<double>("config.measurements.varzr");

		nParticles_ = pt.get("config.optimizer.nParticles", 200);
		maxiter_ = pt.get("config.optimizer.iterations", 200);
		ospa_c_ = pt.get("config.optimizer.ospaC", 1.0);
		Pbirth_ = pt.get<double>("config.optimizer.Pbirth");
		Pdeath_ = pt.get<double>("config.optimizer.Pdeath");
		initMapProb_ = pt.get<double>("config.optimizer.initMapProb");


		hmcslam_->config.temp = pt.get<double>("config.optimizer.initTemp");
		tempReduceFactor_ = pt.get<double>("config.optimizer.reduceFactor");
		hmcslam_->config.K = pt.get<int>("config.optimizer.K");


		hmcslam_->config.m = pt.get<double>("config.optimizer.m");
		optimizeEveryIterations_ = pt.get<int>("config.optimizer.optimizeIter");

		pNoiseInflation_ = pt.get("config.optimizer.predict.processNoiseInflationFactor", 1.0);

		zNoiseInflation_ = pt.get("config.optimizer.update.measurementNoiseInflationFactor", 1.0);

		MeasurementLikelihoodThreshold_ = pt.get<double>("config.optimizer.MLThreshold");

		return true;
	}

	/** Generate a trajectory in 2d space
	 *  \param[in] randSeed random seed for generating trajectory
	 */
	void generateTrajectory(int randSeed = 0) {

		srand48(randSeed);
		initializeGaussianGenerators();

		TimeStamp t;
		int seg = 0;
		MotionModel_Odometry1d::TState::Mat Q;
		Q << vardx_;
		MotionModel_Odometry1d motionModel(Q);
		MotionModel_Odometry1d::TInput input_k(t);
		MotionModel_Odometry1d::TState pose_k(t);
		MotionModel_Odometry1d::TState pose_km(t);
		groundtruth_displacement_.reserve(kMax_);
		groundtruth_pose_.reserve(kMax_);
		groundtruth_displacement_.push_back(input_k);
		groundtruth_pose_.push_back(pose_k);

		for (int k = 1; k < kMax_; k++) {

			t += dTimeStamp_;

			if (k <= 0) {
				double dx = 0;
				MotionModel_Odometry1d::TInput::Vec d;
				MotionModel_Odometry1d::TInput::Vec dCovDiag;
				d << dx;
				dCovDiag << 0;
				input_k = MotionModel_Odometry1d::TInput(d, dCovDiag.asDiagonal(), k);
			} else if (k >= kMax_ / nSegments_ * seg) {
				seg++;
				double dx = (drand48() * (max_dx_ - min_dx_) + min_dx_) * dT_;

				MotionModel_Odometry1d::TInput::Vec d;
				MotionModel_Odometry1d::TInput::Vec dCovDiag;
				d << dx;
				dCovDiag << Q(0, 0);
				input_k = MotionModel_Odometry1d::TInput(d, dCovDiag.asDiagonal(), k);
			}

			groundtruth_displacement_.push_back(input_k);
			groundtruth_displacement_.back().setTime(t);

			MotionModel_Odometry1d::TState x_k;
			motionModel.step(x_k, groundtruth_pose_[k - 1], input_k, dTimeStamp_);
			groundtruth_pose_.push_back(x_k);
			groundtruth_pose_.back().setTime(t);

		}

	}

	/** Generate odometry measurements */
	void generateOdometry() {

		odometry_.reserve(kMax_);
		MotionModel_Odometry1d::TInput zero;
		MotionModel_Odometry1d::TInput::Vec u0;
		u0.setZero();
		zero.set(u0, 0);

		MotionModel_Odometry1d::TState::Mat Q;
		Q << vardx_;
		MotionModel_Odometry1d motionModel(Q);
		deadReckoning_pose_.reserve(kMax_);
		deadReckoning_pose_.push_back(groundtruth_pose_[0]);

		TimeStamp t;

		for (int k = 1; k < kMax_; k++) {

			t += dTimeStamp_;
			double dt = dTimeStamp_.getTimeAsDouble();

			MotionModel_Odometry1d::TInput in = groundtruth_displacement_[k];
			MotionModel_Odometry1d::TState::Mat Qk = Q * dt * dt;
			in.setCov(Qk);
			MotionModel_Odometry1d::TInput out;
			in.sample(out);

			odometry_.push_back(out);

			MotionModel_Odometry1d::TState p;
			motionModel.step(p, deadReckoning_pose_[k - 1], odometry_[k - 1], dTimeStamp_);
			p.setTime(t);

			deadReckoning_pose_.push_back(p);
		}

	}

	/** Generate landmarks */
	void generateLandmarks() {

		MeasurementModel_Rng1D measurementModel(varzr_);
		MeasurementModel_Rng1D::TPose pose;

		groundtruth_landmark_.reserve(nLandmarks_);

		int nLandmarksCreated = 0;
		for (int k = 0; k < kMax_; k++) {

			if (k >= (double) kMax_ / nLandmarks_ * nLandmarksCreated) {

				MeasurementModel_Rng1D::TPose pose;
				MeasurementModel_Rng1D::TMeasurement measurementToCreateLandmark;
				MeasurementModel_Rng1D::TMeasurement::Vec z;
				double r = drand48() * rangeLimitMax_;
				z << r;
				measurementToCreateLandmark.set(z);
				MeasurementModel_Rng1D::TLandmark lm;

				measurementModel.inverseMeasure(groundtruth_pose_[k], measurementToCreateLandmark, lm);

				groundtruth_landmark_.push_back(lm);

				nLandmarksCreated++;

			}

		}

	}

	/** Generate landmark measurements */
	void generateMeasurements() {

		MeasurementModel_Rng1D measurementModel(varzr_);
		MeasurementModel_Rng1D::TMeasurement::Mat R;
		measurementModel.getNoise(R);
		measurementModel.config.rangeLimMax_ = rangeLimitMax_;
		measurementModel.config.rangeLimMin_ = rangeLimitMin_;
		measurementModel.config.probabilityOfDetection_ = Pd_;
		measurementModel.config.uniformClutterIntensity_ = c_;
		double meanClutter = measurementModel.clutterIntensityIntegral();

		double expNegMeanClutter = exp(-meanClutter);
		double poissonPmf[100];
		double poissonCmf[100];
		double mean_pow_i = 1;
		double i_factorial = 1;
		poissonPmf[0] = expNegMeanClutter;
		poissonCmf[0] = poissonPmf[0];
		for (int i = 1; i < 100; i++) {
			mean_pow_i *= meanClutter;
			i_factorial *= i;
			poissonPmf[i] = mean_pow_i / i_factorial * expNegMeanClutter;
			poissonCmf[i] = poissonCmf[i - 1] + poissonPmf[i];
		}

		lmkFirstObsTime_.resize(groundtruth_landmark_.size());
		for (int m = 0; m < lmkFirstObsTime_.size(); m++) {
			lmkFirstObsTime_[m] = -1;
		}

		TimeStamp t;

		for (int k = 0; k < kMax_; k++) {

			groundtruth_pose_[k];

			// Real detections
			for (int m = 0; m < groundtruth_landmark_.size(); m++) {

				bool success;
				MeasurementModel_Rng1D::TMeasurement z_m_k;
				success = measurementModel.sample(groundtruth_pose_[k], groundtruth_landmark_[m], z_m_k);
				if (success) {

					if (fabs(z_m_k.get(0)) <= rangeLimitMax_ && fabs(z_m_k.get(0)) >= rangeLimitMin_ && drand48() <= Pd_) {
						z_m_k.setTime(t);
						// z_m_k.setCov(R);
						measurements_.push_back(z_m_k);
						groundtruthDataAssociation_.push_back(m);
						if (lmkFirstObsTime_[m] == -1) {
							lmkFirstObsTime_[m] = t.getTimeAsDouble();
						}
					}

				}

			}

			// False alarms
			double randomNum = drand48();
			int nClutterToGen = 0;
			while (randomNum > poissonCmf[nClutterToGen]) {
				nClutterToGen++;
			}
			for (int i = 0; i < nClutterToGen; i++) {

				double r = drand48() * rangeLimitMax_;
				while (r < rangeLimitMin_)
					r = drand48() * rangeLimitMax_;
				MeasurementModel_Rng1D::TMeasurement z_clutter;
				MeasurementModel_Rng1D::TMeasurement::Vec z;
				z << r;
				z_clutter.set(z, t);
				measurements_.push_back(z_clutter);
				groundtruthDataAssociation_.push_back(-1);
			}
			t += dTimeStamp_;
		}

	}

	/** Data Logging */
	void exportSimData() {

		if (!logToFile_)
			return;

		std::filesystem::path dir(logDirPrefix_);
		std::filesystem::create_directories(dir);

		std::filesystem::path cfgFilePathSrc(cfgFileName_);
		std::string cfgFileDst(logDirPrefix_);
		cfgFileDst += "simSettings.cfg";
		std::filesystem::path cfgFilePathDst(cfgFileDst.data());
		std::filesystem::copy_file(cfgFilePathSrc, cfgFilePathDst, std::filesystem::copy_options::overwrite_existing);

		TimeStamp t;

		FILE* pGTPoseFile;
		std::string filenameGTPose(logDirPrefix_);
		filenameGTPose += "gtPose.dat";
		pGTPoseFile = fopen(filenameGTPose.data(), "w");
		MotionModel_Odometry1d::TState::Vec x;
		for (int i = 0; i < groundtruth_pose_.size(); i++) {
			groundtruth_pose_[i].get(x, t);
			fprintf(pGTPoseFile, "%f   %f\n", t.getTimeAsDouble(), x(0));
		}
		fclose(pGTPoseFile);

		FILE* pGTLandmarkFile;
		std::string filenameGTLandmark(logDirPrefix_);
		filenameGTLandmark += "gtLandmark.dat";
		pGTLandmarkFile = fopen(filenameGTLandmark.data(), "w");
		MeasurementModel_Rng1D::TLandmark::Vec m;
		for (int i = 0; i < groundtruth_landmark_.size(); i++) {
			groundtruth_landmark_[i].get(m);
			fprintf(pGTLandmarkFile, "%f   %f\n", m(0), lmkFirstObsTime_[i]);
		}
		fclose(pGTLandmarkFile);

		FILE* pOdomFile;
		std::string filenameOdom(logDirPrefix_);
		filenameOdom += "odometry.dat";
		pOdomFile = fopen(filenameOdom.data(), "w");
		MotionModel_Odometry1d::TInput::Vec u;
		for (int i = 0; i < odometry_.size(); i++) {
			odometry_[i].get(u, t);
			fprintf(pOdomFile, "%f   %f\n", t.getTimeAsDouble(), u(0));
		}
		fclose(pOdomFile);

		FILE* pMeasurementFile;
		std::string filenameMeasurement(logDirPrefix_);
		filenameMeasurement += "measurement.dat";
		pMeasurementFile = fopen(filenameMeasurement.data(), "w");
		MeasurementModel_Rng1D::TMeasurement::Vec z;
		for (int i = 0; i < measurements_.size(); i++) {
			measurements_[i].get(z, t);
			fprintf(pMeasurementFile, "%f   %f\n", t.getTimeAsDouble(), z(0));
		}
		fclose(pMeasurementFile);

		FILE* pDeadReckoningFile;
		std::string filenameDeadReckoning(logDirPrefix_);
		filenameDeadReckoning += "deadReckoning.dat";
		pDeadReckoningFile = fopen(filenameDeadReckoning.data(), "w");
		MotionModel_Odometry1d::TState::Vec odo;
		for (int i = 0; i < deadReckoning_pose_.size(); i++) {
			deadReckoning_pose_[i].get(odo, t);
			fprintf(pDeadReckoningFile, "%f   %f\n", t.getTimeAsDouble(), odo(0));
		}
		fclose(pDeadReckoningFile);

	}

	/** PSO optimizer Setup */
	void setup() {

		double dt = dTimeStamp_.getTimeAsDouble();

		// configure robot motion model (only need to set once since timesteps are constant)
		MotionModel_Odometry1d::TState::Mat Q;
		Q << vardx_;
		Q *= (pNoiseInflation_ * dt * dt);
		hmcslam_->robotProcessModelPtr_->setNoise(Q);

		// configure landmark process model (only need to set once since timesteps are constant)

		// configure measurement model
		MeasurementModel_Rng1D::TMeasurement::Mat R;
		R << varzr_;
		R *= zNoiseInflation_;
		hmcslam_->mModelPtr_->setNoise(R);
		hmcslam_->mModelPtr_->config.probabilityOfDetection_ = Pd_;
		hmcslam_->mModelPtr_->config.uniformClutterIntensity_ = c_;
		hmcslam_->mModelPtr_->config.rangeLimMax_ = rangeLimitMax_;
		hmcslam_->mModelPtr_->config.rangeLimMin_ = rangeLimitMin_;
		hmcslam_->mModelPtr_->config.rangeLimBuffer_ = rangeLimitBuffer_;

		// configure the filter
		hmcslam_->config.nParticles_ = nParticles_;
		hmcslam_->config.mapFromMeasurementProb_ = initMapProb_;
		hmcslam_->config.Pb = Pbirth_;
		hmcslam_->config.Pd = Pdeath_;
		hmcslam_->config.MeasurementLikelihoodThreshold_ = MeasurementLikelihoodThreshold_;

		hmcslam_->setInputs(odometry_);
		hmcslam_->addMeasurement(measurements_);

	}

	/** Run the simulator */
	void run() {

		printf("Running simulation\n\n");

#ifdef _PERFTOOLS_CPU
		std::string perfCPU_file = logDirPrefix_ + "rbcbmemberslam2dSim_cpu.prof";
		ProfilerStart(perfCPU_file.data());
#endif
#ifdef _PERFTOOLS_HEAP
		std::string perfHEAP_file = logDirPrefix_ + "rbcbmemberslam2dSim_heap.prof";
		HeapProfilerStart(perfHEAP_file.data());
#endif
		//////// Initialization by sampling motion model //////////

		FILE* pParticlePoseFile;
		if (logToFile_) {
			std::string filenameParticlePoseFile(logDirPrefix_);
			filenameParticlePoseFile += "particlePose.dat";
			pParticlePoseFile = fopen(filenameParticlePoseFile.data(), "w");
		}
		FILE* pBestParticlePoseFile;
		if (logToFile_) {
			std::string filenameBestParticlePoseFile(logDirPrefix_);
			filenameBestParticlePoseFile += "bestParticlePose.dat";
			pBestParticlePoseFile = fopen(filenameBestParticlePoseFile.data(), "w");
		}
		FILE* pLandmarkEstFile;
		if (logToFile_) {
			std::string filenameLandmarkEstFile(logDirPrefix_);
			filenameLandmarkEstFile += "landmarkEst.dat";
			pLandmarkEstFile = fopen(filenameLandmarkEstFile.data(), "w");
		}
		FILE* pBestLandmarkEstFile;
		if (logToFile_) {
			std::string filenameBestLandmarkEstFile(logDirPrefix_);
			filenameBestLandmarkEstFile += "landmarkEst.dat";
			pBestLandmarkEstFile = fopen(filenameBestLandmarkEstFile.data(), "w");
		}
		MotionModel_Odometry1d::TState x_i;
		int zIdx = 0;

		std::vector<TParticle> particles;
		hmcslam_->init(particles);
		int iteration = 0;

		if (logToFile_) {
			/////////// log particle trajectories  //////////////
			for (int i = 0; i < hmcslam_->config.nParticles_; i++) {
				fprintf(pParticlePoseFile, "%d   ", iteration);
				fprintf(pParticlePoseFile, "%f   ", particles.at(i).currentLikelihood);
				for (int k = 0; k < particles.at(i).trajectory.size(); k++) {
					x_i = particles.at(i).trajectory[k];
					fprintf(pParticlePoseFile, "%f   ", x_i.get(0));
				}
				fprintf(pParticlePoseFile, "\n");
			}

			///////////// log landmarks ///////////////////

			for (int i = 0; i < hmcslam_->config.nParticles_; i++) {
				fprintf(pLandmarkEstFile, "%d    ", iteration);
				for (int l = 0; l < particles.at(i).landmarks.size(); l++) {

					fprintf(pLandmarkEstFile, "%f   ", particles.at(i).landmarks[l][0]);
				}
				fprintf(pLandmarkEstFile, "\n");

			}

		}

		// run the optimization process
		for (iteration++; iteration < maxiter_; iteration++) {

			if (iteration % 100 == 0) {
				hmcslam_->config.temp *=tempReduceFactor_;
			}


			hmcslam_->evaluateLikelihoods(particles);
			hmcslam_->reversibleJumpHMC(particles);

			if (iteration % 10 == 0 || iteration == maxiter_ - 1) {
				float progressPercent = float(iteration + 1) / float(maxiter_);
				int progressBarW = 50;
				struct winsize ws;
				if (ioctl(1, TIOCGWINSZ, &ws) >= 0)
					progressBarW = ws.ws_col - 30;
				int progressPos = progressPercent * progressBarW;
				if (progressBarW >= 50) {
					std::cout << "[";
					for (int i = 0; i < progressBarW; i++) {
						if (i < progressPos)
							std::cout << "=";
						else if (i == progressPos)
							std::cout << ">";
						else
							std::cout << " ";
					}
					std::cout << "] ";
				}
				std::cout << "iteration = " << iteration << " (" << int(progressPercent * 100.0) << " %)\r";
				std::cout.flush();
			}
			if (iteration == maxiter_ - 1)
				std::cout << std::endl << std::endl;

			if (logToFile_) {
				/////////// log particle trajectories  //////////////
				for (int i = 0; i < hmcslam_->config.nParticles_; i++) {
					fprintf(pParticlePoseFile, "%d   ", iteration);
					fprintf(pParticlePoseFile, "%f   ", particles.at(i).currentLikelihood);
					for (int k = 0; k < particles.at(i).trajectory.size(); k++) {
						x_i = particles.at(i).trajectory[k];
						fprintf(pParticlePoseFile, "%f   ", x_i.get(0));
					}
					fprintf(pParticlePoseFile, "\n");
				}

				///////////// log landmarks ///////////////////

				for (int i = 0; i < hmcslam_->config.nParticles_; i++) {
					fprintf(pLandmarkEstFile, "%d    ", iteration);
					for (int l = 0; l < particles.at(i).landmarks.size(); l++) {


						fprintf(pLandmarkEstFile, "%f   ", particles.at(i).landmarks[l][0]);
					}
					fprintf(pLandmarkEstFile, "\n");

				}

			}
			std::cout << "iteration:"  << iteration << "  likelihood: "<< hmcslam_->getBestParticle(particles)->bestLikelihood << "\n";
		}

		// measure ground truth likelihood!
		particles.at(0).landmarks.resize(groundtruth_landmark_.size());
		particles.at(0).landmarks_gradient.resize(groundtruth_landmark_.size());
		for(int i=0; i< groundtruth_landmark_.size() ; i++){
			particles.at(0).landmarks[i] = groundtruth_landmark_[i].get();
		}
		for(int i=0 ; i< groundtruth_pose_.size() ; i++){
		particles.at(0).trajectory[i] = groundtruth_pose_[i].get();
		}

		hmcslam_->evaluateLikelihoods(particles);
		std::cout << "GT likelihood:  " << particles.at(0).currentLikelihood << "\n";

		if (logToFile_) {
			fclose(pParticlePoseFile);
			fclose(pLandmarkEstFile);
		}
	}

private:

	const char* cfgFileName_;

	int kMax_; /**< number of timesteps */
	double dT_; /**< duration of timestep in seconds */
	TimeStamp dTimeStamp_; /**< duration of timestep in timestamp */

	// Trajectory
	int nSegments_;
	double max_dx_;
	double min_dx_;
	double vardx_;
	std::vector<MotionModel_Odometry1d::TInput> groundtruth_displacement_;
	std::vector<MotionModel_Odometry1d::TState> groundtruth_pose_;
	std::vector<MotionModel_Odometry1d::TInput> odometry_;
	std::vector<MotionModel_Odometry1d::TState> deadReckoning_pose_;

	// Landmarks
	int nLandmarks_;
	std::vector<MeasurementModel_Rng1D::TLandmark> groundtruth_landmark_;

	std::vector<double> lmkFirstObsTime_;

	// Range-Bearing Measurements
	double rangeLimitMax_;
	double rangeLimitMin_;
	double rangeLimitBuffer_;
	double Pd_;
	double c_;
	double varzr_;
	std::vector<MeasurementModel_Rng1D::TMeasurement> measurements_;
	std::vector<int> groundtruthDataAssociation_;

	// Filters
	RFSHMCSLAM<MotionModel_Odometry1d, MeasurementModel_Rng1D> *hmcslam_;
	int nParticles_;
	double ospa_c_;
	double initMapProb_,Pbirth_,Pdeath_;
	int maxiter_;
	double tempReduceFactor_;

	double pNoiseInflation_;
	double zNoiseInflation_;

	int optimizeEveryIterations_;

	double MeasurementLikelihoodThreshold_;

	bool logToFile_;

public:
	std::string logDirPrefix_;
};

int main(int argc, char* argv[]) {

	Simulator_RFSPSOSLAM_1d sim;
	int seed = time(NULL);
	srand(seed);
	int trajNum = rand();
	std::string cfgFileName;
  bool printHelp = false;
  argparse::ArgumentParser parser("This is a test program for argparse");
  parser.add_argument("-h", "--help").help("produce this help message").store_into(printHelp);
  parser.add_argument("-c", "--cfg").help("configuration xml file").default_value("cfg/rbphdslam2dSim.xml").store_into(cfgFileName);
  parser.add_argument("-t", "--trajectory").help("trajectory number (default: a random integer)").store_into(trajNum);
  parser.add_argument("-s", "--seed").help("random seed for running the simulation (default: based on current system time)").store_into(seed);
  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error& e) {
    std::cout << e.what() << std::endl;
    std::cout << parser;
    return 1;
  }

  if(printHelp){
    std::cout << parser;
    return 1;
  }

  std::cout << "Configuration file: " << cfgFileName << std::endl;
  if( !sim.readConfigFile( cfgFileName.data() ) ){
    return -1;
  }
  


	if (!sim.readConfigFile(cfgFileName.data())) {
		return -1;
	}

	std::cout << "Trajectory: " << trajNum << std::endl;
	sim.generateTrajectory(trajNum);

	sim.generateLandmarks();
	sim.generateOdometry();
	sim.generateMeasurements();

	sim.exportSimData();

	sim.setup();
  if( parser.is_used("seed") ){
    std::cout << "Simulation random seed manually set to: " << seed << std::endl;
  }
  else{
    std::cout << "Simulation random seed set to: " << seed << std::endl;
  }
	srand48(seed);
	initializeGaussianGenerators();

	// boost::timer::auto_cpu_timer *timer = new boost::timer::auto_cpu_timer(6, "Simulation run time: %ws\n");

	sim.run();

	// std::cout << "mem use: " << MemProfile::getCurrentRSS() << "(" << MemProfile::getPeakRSS() << ")\n";
	//delete timer;

	return 0;

}
