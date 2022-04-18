#include "NNG2O_2D.hpp"
#include <boost/program_options.hpp>


int main(int argc, char* argv[]){

   rfs::NNG2O2D solver;


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

  solver.loadConfig(cfgFileName);
  solver.init(solver.gt_graph);
  solver.load(g2oFileName);
  //vglmb.calculateWeight(vglmb.gt_graph);
  //std::cout << "GROUND TRUTH WEIGHT:             " <<vglmb.gt_graph.logweight_ << "\n";
  //std::cout << "weight: " << vglmb.gt_graph.logweight_ << "   chi2:  " <<vglmb.gt_graph.optimizer_->chi2() << "  determinant: " << vglmb.gt_graph.linearSolver_->_determinant<< "\n";

  solver.initComponents();

  solver.run(solver.config.numIterations_);



}
