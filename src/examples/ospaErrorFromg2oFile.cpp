/*
 * Software License Agreement (New BSD License)
 *
 * Copyright (c) 2014, Keith Leung
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

#include <iostream>
#include <math.h>
#include <vector>
#include "OSPA.hpp"
#include <fstream>
#include <filesystem>


class Pos2d{

public:
  
  Pos2d(double x, double y): x_(x), y_(y){}
  ~Pos2d(){}

  static double distance( const Pos2d x1 , const Pos2d x2 ){
    double dx = x1.x_ - x2.x_;
    double dy = x1.y_ - x2.y_;
    return sqrt(dx*dx + dy*dy);
  }
  
  double x_;
  double y_;

};

std::vector<Pos2d> readg2oFile(std::string filename){
	std::vector<Pos2d> set;
	std::ifstream file(filename);
	if (file.is_open())
	  {
		  std::string line;
	    while ( getline (file,line) )
	    {

	    	std::vector <std::string> tokens;
	    	std::stringstream check1(line);
	    	std::string intermediate;
	        // Tokenizing w.r.t. space ' '
	        while(getline(check1, intermediate, ' '))
	        {
	            tokens.push_back(intermediate);
	        }
	        if (tokens[0] ==  "VERTEX_XY"){
	        	set.push_back(Pos2d(std::stof(tokens[2]),std::stof(tokens[3])) );
	        }
	    }
	    file.close();
	  }
	return set;

}

inline bool file_exists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

int main(int argc, char *argv[]){

  const double cutoff = 10;
  const double order = 2;


  std::ifstream in1("file1");
  std::ifstream in2("file2");


  std::vector<Pos2d> gtset = readg2oFile("/home/finostro/g2osim/001.g2o");

  int i=0;
  while (true){
	  std::stringstream filename;
	  filename << "beststate_" ;
	  filename << std::setw(5)  << std::setfill('0') << i ;
	  filename << ".g2o";
	  if (!file_exists(filename.str())){
		  break;
	  }


	  std::vector<Pos2d> bestSet = readg2oFile(filename.str());
	  rfs::OSPA<Pos2d> ospa(gtset, bestSet, cutoff, order, &Pos2d::distance);
	    double e, e_d, e_c;
	    e = ospa.calcError(&e_d, &e_c, true);
	  i++;
	  std::cout << e  <<"  "<< e_d  <<"  "<< e_c  <<"  "<< std::endl;

  }








};
