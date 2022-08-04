#include "VectorGLMBSLAM2D.hpp"
#include <boost/program_options.hpp>


int main(int argc, char* argv[]){

   rfs::VectorGLMBSLAM2D vglmb;


  std::string cfgFileName,g2oFileName;
  boost::program_options::options_description desc("Options");
  desc.add_options()
    ("help,h", "produce this help message")
    ("cfg,c", boost::program_options::value<std::string>(&cfgFileName)->default_value("cfg/vglmbslam2d.yaml"), "configuration xml file")
    ("g2ofile,g", boost::program_options::value<std::string>(&g2oFileName)->default_value("g2ofile"), "g2o style 2D simulation output");
  boost::program_options::variables_map vm;
  boost::program_options::store( boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if( vm.count("help") ){
    std::cout << desc << "\n";
    return 1;
  }

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
