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

#include "RBPHDSLAM_2D.hpp"
#include "MeasurementModel_XY.hpp"
#include "external/argparse.hpp"





int main(int argc, char* argv[]){

  RBPHDSLAM_2D<MeasurementModel_XY> sim;

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
  
  
  std::cout << "Trajectory: " << trajNum << std::endl;
  //sim.generateTrajectory( trajNum );  
  
  //sim.generateLandmarks();
  //sim.generateOdometry();
  //sim.generateMeasurements();
  sim.readISAMFile("data/isam/cityTrees10000.txt");
  sim.readISAMGroundTruth("data/isam/groundtruth/cityTrees10000_groundtruth.txt");
  std::cout << "Configuration file: " << cfgFileName << std::endl;
  if( !sim.readConfigFile( cfgFileName.data() ) ){
    return -1;
  }
  sim.exportSimData();
  sim.setupRBPHDFilter();

  if( parser.is_used("seed") ){
    std::cout << "Simulation random seed manually set to: " << seed << std::endl;
  }
  else{
    std::cout << "Simulation random seed set to: " << seed << std::endl;
  }
  srand48( seed );
  initializeGaussianGenerators();

  // boost::timer::auto_cpu_timer *timer = new boost::timer::auto_cpu_timer(6, "Simulation run time: %ws\n");

  sim.run();

  // std::cout << "mem use: " << MemProfile::getCurrentRSS() << "(" << MemProfile::getPeakRSS() << ")\n";
  //delete timer;

  return 0;

}
