#include "VectorGLMBSLAM2D.hpp"
#include "external/argparse.hpp"


int main(int argc, char* argv[]){

   rfs::VectorGLMBSLAM2D vglmb;


  std::string cfgFileName,g2oFileName;
  bool printHelp = false;
  argparse::ArgumentParser parser("This is a test program for argparse");
  parser.add_argument("-h", "--help").help("produce this help message").store_into(printHelp);
  parser.add_argument("-c", "--cfg").help("configuration xml file").default_value("cfg/rbphdslam2dSim.xml").store_into(cfgFileName);
  parser.add_argument("-g", "--g2ofile").help("g2o style 2D simulation output").store_into(g2oFileName).default_value("g2ofile");
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
  

  rfs::initializeGaussianGenerators();

  vglmb.loadConfig(cfgFileName);
  vglmb.init(vglmb.gt_graph);
  vglmb.load(g2oFileName);
  //vglmb.calculateWeight(vglmb.gt_graph);
  //std::cout << "GROUND TRUTH WEIGHT:             " <<vglmb.gt_graph.logweight_ << "\n";
  //std::cout << "weight: " << vglmb.gt_graph.logweight_ << "   chi2:  " <<vglmb.gt_graph.optimizer_->chi2() << "  determinant: " << vglmb.gt_graph.linearSolver_->_determinant<< "\n";

  vglmb.initComponents();

#ifdef _PERFTOOLS_CPU
		std::string perfCPU_file =  "vglmb.prof";
		ProfilerStart(perfCPU_file.data());
#endif
#ifdef _PERFTOOLS_HEAP
		std::string perfHEAP_file = "vglmb_heap.prof";
		HeapProfilerStart(perfHEAP_file.data());
#endif
  vglmb.run(vglmb.config.numIterations_);
  #ifdef _PERFTOOLS_HEAP
		HeapProfilerStop();
#endif
#ifdef _PERFTOOLS_CPU
		ProfilerStop();
#endif

  vglmb.components_[0].optimizer_->save(vglmb.config.finalStateFile_.c_str() , 0);
  vglmb.components_[0].DA_bimap_ = vglmb.best_DA_;
  vglmb.updateGraph(vglmb.components_[0]);
  vglmb.components_[0].optimizer_->initializeOptimization();
  vglmb.components_[0].optimizer_->optimize(50);
  vglmb.components_[0].optimizer_->save("beststate.g2o" , 0);


}
