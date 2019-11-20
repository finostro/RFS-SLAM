#include "AssociationSampler.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <yaml-cpp/yaml.h>

using namespace Eigen;
using namespace std;

Sample::Sample(const Sample& sample_)
{
	associations = std::vector<std::pair<size_t, size_t>>(sample_.associations);
	remaining_rows = std::vector<std::pair<size_t, size_t>>(sample_.remaining_rows);
	remaining_cols = std::vector<std::pair<size_t, size_t>>(sample_.remaining_cols);
}

bool Sample::operator==(const Sample& rhs) const
{
	if(this->associations.size() != rhs.associations.size())
		return false;
	if(this->remaining_cols.size() != rhs.remaining_cols.size())
		return false;
	if(this->remaining_rows.size() != rhs.remaining_rows.size())
		return false;
	//si las cantidades son las mismas hay que comparar los elementos
	//asegurarse de que los elementos esten ordenados, sino el operador
	//de comparacion no funciona
	for(size_t i = 0; i < this->associations.size(); i++)
	{
		if(this->associations[i].first != rhs.associations[i].first)
			return false;
		if(this->associations[i].second != rhs.associations[i].second)
			return false;
	}
	for(size_t i = 0; i < this->remaining_cols.size(); i++)
	{
		if(this->remaining_cols[i].first != rhs.remaining_cols[i].first)
			return false;
		if(this->remaining_cols[i].second != rhs.remaining_cols[i].second)
			return false;
	}
	for(size_t i = 0; i < this->remaining_rows.size(); i++)
	{
		if(this->remaining_rows[i].first != rhs.remaining_rows[i].first)
			return false;
		if(this->remaining_rows[i].second != rhs.remaining_rows[i].second)
			return false;
	}

	return true;
}

void Sample::sort()
{
	// Using simple sort() function to sort
	std::sort(associations.begin(), associations.end());
	std::sort(remaining_rows.begin(), remaining_rows.end());
	std::sort(remaining_cols.begin(), remaining_cols.end());
}


double GenericSampler::eval_assignment(const CostMat &cost_mat, const Sample &sample_)
{
	double weight = 0;

	for(auto &a: sample_.associations)
	{
		weight += cost_mat.cost_main(a.first, a.second);
	}

	for(auto &r: sample_.remaining_rows)
	{
		weight += cost_mat.cost_rows(r.first, r.second);
	}

	for(auto &c: sample_.remaining_cols)
	{
		weight += cost_mat.cost_rows(c.second, c.first);
	}

	return weight;
}

std::vector<Sample> GenericSampler::samples(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_, size_t number_of_samples,  SampleBy flag)
{
	std::vector<Sample> samples_;
	samples_.reserve(number_of_samples);
	for(size_t n = 0; n < number_of_samples; n++)
	{
		samples_.push_back(sample(gen, cost_mat, sample_, flag));//DEFINIR FLAG CUANDO flag == ANY
	}
	return samples_;
}


Sample GibbsSampler::sample(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample,  SampleBy flag )
{
	//
	if(flag == SampleBy::ROW)
		return sample_by_row(gen, cost_mat, sample);
	else if(flag == SampleBy::COL)
		return sample_by_column(gen, cost_mat, sample);
	else if(flag == SampleBy::ROWG)
		return sample_by_row_g(gen, cost_mat, sample);
	else if(flag == SampleBy::COLG)
		return sample_by_column_g(gen, cost_mat, sample);
	else
		throw std::runtime_error("No puede samplear ANY");

	//
}

Sample GibbsSampler::sample_by_row(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_)
{
	const size_t MM = cost_mat.cost_main.cols();
	const size_t N = sample_.associations.size() + sample_.remaining_rows.size();
	const size_t V = cost_mat.cost_cols.cols();

	std::vector<size_t> indices(N);
	std::vector<size_t> associations(N);
	Sample sample_out;
	sample_out.remaining_cols = std::vector<std::pair<size_t, size_t>>(sample_.remaining_cols);

	//se escribe el sample inicial
	{
		size_t t = 0;
		for(auto &a: sample_.associations)
		{
			indices[t] = a.first;
			associations[t] = a.second;
			++t;
		}

		for(auto &rr: sample_.remaining_rows)
		{
			indices[t] = rr.first;
			associations[t] = MM + rr.second;
			++t;
		}
	}

	double w, wmax;
	int number_associated = sample_.associations.size();

	for(size_t t = 0; t < N; t++)
	{
		size_t i = indices[t];				//indice de la fila en la matriz de asociaciones
		size_t jc = associations[t];		//indice de la columna actual asociada a la fila "i"
		const size_t R = sample_out.remaining_cols.size();

		vector<double> weights(R + V + 1);

		bool is_detected = jc < MM;

		//se llenan los pesos de los pesos correspondientes a las mis-detecciones
		wmax = cost_mat.cost_cols(i, 0);
		weights[R] = wmax;	//v=0
		for(size_t v = 1; v < V; v++)
		{
			w = cost_mat.cost_cols(i, v);
			if(w > wmax) wmax = w;
			weights[R + v] = w;
		}

		//se llenan los pesos de los estados "libres" que son misdetection
		for(size_t r = 0; r < R; r++)
		{
			size_t j = sample_out.remaining_cols[r].first;
			w = cost_mat.cost_main(i, j);
			if(w > wmax) wmax = w;
			weights[r] = w;
		}

		if(is_detected)
		{
			w = cost_mat.cost_main(i, jc);
			if(w > wmax) wmax = w;
			weights[R + V] = std::exp(w - wmax);	//se normaliza el peso
		}
		else
		{
			weights[R + V] = 0;
		}



		//se normalizan todos los pesos
		for(size_t r = 0; r < R + V; r++)
		{
			weights[r] = std::exp(weights[r] - wmax);
		}



		size_t k;
		k = sample(gen,weights);


//		boost::random::discrete_distribution<> assignment(weights.begin(), weights.end());
//		k = assignment(gen);	//se realiza el muestreo

		size_t js = (k < R)?(sample_out.remaining_cols[k].first):(MM + k - R);
		if(js == MM + V) js = jc;

		if(js == jc) continue;		//se obtuvo el mismo muestreo anterior, por lo tanto no se hace nada

		if(is_detected)
		{
			std::pair<size_t, size_t> new_col;
			new_col.first = jc;
			sample_out.remaining_cols.push_back(new_col);
		}

		if(k < R)
		{
			associations[t] = sample_out.remaining_cols[k].first;
			// Deletes the element of index k
			sample_out.remaining_cols.erase(sample_out.remaining_cols.begin() + k);
			if(!is_detected)
				number_associated++;
		}
		else
		{
			associations[t] = js;
			if(is_detected)
				number_associated--;
		}
	}

	//se escribe la solucion en formato de salida (clase Sample)
	sample_out.associations.resize(number_associated);
	sample_out.remaining_rows.resize(N - number_associated);

	auto a = sample_out.associations.begin();
	auto rr = sample_out.remaining_rows.begin();

	for(size_t t = 0; t < N; t++)
	{
		size_t i = indices[t];
		size_t j = associations[t];
		if(j < MM)
		{
			a->first = i;
			a->second = j;
			++a;
		}
		else
		{
			rr->first = i;
			rr->second = j - MM;
			++rr;
		}
	}

	return sample_out;
}

Sample GibbsSampler::sample_by_column(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_)
{
	const size_t NN = cost_mat.cost_main.rows();
	const size_t M = sample_.associations.size() + sample_.remaining_cols.size();
	const size_t V = cost_mat.cost_rows.rows();

	std::vector<size_t> indices(M);
	std::vector<size_t> associations(M);
	Sample sample_out;

	sample_out.remaining_rows = std::vector<std::pair<size_t, size_t>>(sample_.remaining_rows);

	//se escribe el sample inicial
	{
		size_t t = 0;
		for(auto &a: sample_.associations)
		{
			indices[t] = a.second;
			associations[t] = a.first;
			++t;
		}

		for(auto &rc: sample_.remaining_cols)
		{
			indices[t] = rc.first;
			associations[t] = NN + rc.second;
			++t;
		}
	}

	double w, wmax;
	int number_associated = sample_.associations.size();

	for(size_t t = 0; t < M; t++)
	{
		size_t j = indices[t];				//indice de la columna en la matriz de asociaciones
		size_t ic = associations[t];		//indice de la columna actual asociada a la fila "j"

		const size_t R = sample_out.remaining_rows.size();

		vector<double> weights(R + V + 1);

		bool is_detected = ic < NN;

		//se llenan los pesos de los pesos correspondientes a las mis-detecciones
		wmax = cost_mat.cost_rows(0, j);
		weights[R] = wmax;	//v=0
		for(size_t v = 1; v < V; v++)
		{
			w = cost_mat.cost_rows(v, j);
			if(w > wmax) wmax = w;
			weights[R + v] = w;
		}

		//se llenan los pesos de los estados "libres" que son misdetection
		for(size_t r = 0; r < R; r++)
		{
			size_t i = sample_out.remaining_rows[r].first;
			w = cost_mat.cost_main(i, j);
			if(w > wmax) wmax = w;
			weights[r] = w;
		}

		if(is_detected)
		{
			w = cost_mat.cost_main(ic, j);
			if(w > wmax) wmax = w;
			weights[R + V] = std::exp(w - wmax);	//se normaliza el peso
		}
		else
		{
			weights[R + V] = 0;
		}

		//se normalizan todos los pesos
		for(size_t r = 0; r < R + V; r++)
		{
			weights[r] = std::exp(weights[r] - wmax);
		}

		boost::random::discrete_distribution<> assignment(weights.begin(), weights.end());
		size_t k = assignment(gen);	//se realiza el muestreo
		size_t is = (k < R)?(sample_out.remaining_rows[k].first):(NN + k - R);

		if(is == NN + V) is = ic;

		if(is == ic) continue;		//se obtuvo el mismo muestreo anterior, por lo tanto no se hace nada

		if(is_detected)
		{
			std::pair<size_t, size_t> new_row;
			new_row.first = ic;
			sample_out.remaining_rows.push_back(new_row);
		}

		if(k < R)
		{
			associations[t] = sample_out.remaining_rows[k].first;
			// Deletes the element of index k
			sample_out.remaining_rows.erase(sample_out.remaining_rows.begin() + k);
			if(!is_detected)
				number_associated++;
		}
		else
		{
			associations[t] = is;
			if(is_detected)
				number_associated--;
		}
	}


	//se escribe la solucion en formato de salida (clase Sample)
	sample_out.associations.resize(number_associated);
	sample_out.remaining_cols.resize(M - number_associated);

	auto a = sample_out.associations.begin();
	auto rc = sample_out.remaining_cols.begin();

	for(size_t t = 0; t < M; t++)
	{
		size_t j = indices[t];
		size_t i = associations[t];
		if(i < NN)
		{
			a->first = i;
			a->second = j;
			++a;
		}
		else
		{
			rc->first = j;
			rc->second = i - NN;
			++rc;
		}
	}

	return sample_out;
}


Sample GibbsSampler::sample_by_row_g(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_)
{
	const size_t MM = cost_mat.cost_main.cols();
	const size_t N = sample_.associations.size() + sample_.remaining_rows.size();
	const size_t V = cost_mat.cost_cols.cols();

	std::vector<size_t> indices(N);
	std::vector<size_t> associations(N);
	Sample sample_out;
	sample_out.remaining_cols = std::vector<std::pair<size_t, size_t>>(sample_.remaining_cols);

	//se escribe el sample inicial
	{
		size_t t = 0;
		for(auto &a: sample_.associations)
		{
			indices[t] = a.first;
			associations[t] = a.second;
			++t;
		}

		for(auto &rr: sample_.remaining_rows)
		{
			indices[t] = rr.first;
			associations[t] = MM + rr.second;
			++t;
		}
	}

	double w, wmax;
	int number_associated = sample_.associations.size();
	std::vector<double> weights_remaining(V);							//*general*//

	for(size_t t = 0; t < N; t++)
	{
		size_t i = indices[t];				//indice de la fila en la matriz de asociaciones
		size_t jc = associations[t];		//indice de la columna actual asociada a la fila "i"
		const size_t R = sample_out.remaining_cols.size();

		vector<double> weights(R + V + 1);

		bool is_detected = jc < MM;

		//se llenan los pesos de los pesos correspondientes a las mis-detecciones
		wmax = cost_mat.cost_cols(i, 0);
		weights[R] = wmax;	//v=0
		for(size_t v = 1; v < V; v++)
		{
			w = cost_mat.cost_cols(i, v);
			if(w > wmax) wmax = w;
			weights[R + v] = w;
		}

		//se llenan los pesos de los estados "libres" que son misdetection
		for(size_t r = 0; r < R; r++)
		{
			size_t j = sample_out.remaining_cols[r].first;
			size_t v = sample_out.remaining_cols[r].second;				//*general*//
			w = cost_mat.cost_main(i, j) - cost_mat.cost_rows(v, j);	//*general*//
			if(w > wmax) wmax = w;
			weights[r] = w;
		}

		if(is_detected)
		{
			w = cost_mat.cost_main(i, jc) - cost_mat.sum_cost_rows(jc);		//*general*//
			if(w > wmax) wmax = w;
			weights[R + V] = std::exp(w - wmax);	//se normaliza el peso
		}
		else
		{
			weights[R + V] = 0;
		}

		//se normalizan todos los pesos
		for(size_t r = 0; r < R + V; r++)
		{
			weights[r] = std::exp(weights[r] - wmax);
		}

		boost::random::discrete_distribution<> assignment(weights.begin(), weights.end());
		size_t k = assignment(gen);	//se realiza el muestreo

		size_t js = (k < R)?(sample_out.remaining_cols[k].first):(MM + k - R);
		if(js == MM + V) js = jc;

		if(js == jc) continue;		//se obtuvo el mismo muestreo anterior, por lo tanto no se hace nada

		if(is_detected)
		{
			std::pair<size_t, size_t> new_col;
			new_col.first = jc;

			//***************************general***************************//
			//se samplea a que fila de la matriz cost_rows correspondiente
			wmax = cost_mat.cost_rows(0, jc);
			weights_remaining[0] = wmax;
			for(size_t v = 1; v < V; v++)
			{
				w = cost_mat.cost_rows(v, jc);
				if(w > wmax) wmax = w;
				weights_remaining[v] = w;
			}
			for(size_t v = 0; v < V; v++)
			{
				weights_remaining[v] = std::exp(weights_remaining[v] - wmax);
			}
			boost::random::discrete_distribution<> assignment_remaining(weights_remaining.begin(), weights_remaining.end());
			new_col.second = assignment_remaining(gen);	//se realiza el muestreo
			//***************************general***************************//

			sample_out.remaining_cols.push_back(new_col);
		}

		if(k < R)
		{
			associations[t] = sample_out.remaining_cols[k].first;
			// Deletes the element of index k
			sample_out.remaining_cols.erase(sample_out.remaining_cols.begin() + k);
			if(!is_detected)
				number_associated++;
		}
		else
		{
			associations[t] = js;
			if(is_detected)
				number_associated--;
		}
	}

	//se escribe la solucion en formato de salida (clase Sample)
	sample_out.associations.resize(number_associated);
	sample_out.remaining_rows.resize(N - number_associated);

	auto a = sample_out.associations.begin();
	auto rr = sample_out.remaining_rows.begin();

	for(size_t t = 0; t < N; t++)
	{
		size_t i = indices[t];
		size_t j = associations[t];
		if(j < MM)
		{
			a->first = i;
			a->second = j;
			++a;
		}
		else
		{
			rr->first = i;
			rr->second = j - MM;
			++rr;
		}
	}

	return sample_out;
}

size_t GibbsSampler::sample(boost::mt19937 &gen, std::vector<double> &weights){

	size_t size=weights.size();
	double sumW=0;
	for(size_t r = 0; r < size; r++)
	{

		sumW += weights[r];
	}


	vector<uint32_t> cumulative_weights( weights.size());
	uint32_t rand=gen();
	uint32_t prevCum=0;
	size_t k=0;
	for(size_t r = 0; r <  size; r++)
	{

		cumulative_weights[r] = prevCum + (uint32_t)((weights[r] /sumW) * std::numeric_limits<uint32_t>::max());
		prevCum = cumulative_weights[r];

		k=r;
		if(cumulative_weights[r] >= rand){
			break;
		}
	}
	return k;
}
Sample GibbsSampler::sample_by_column_g(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_)
{
	const size_t NN = cost_mat.cost_main.rows();
	const size_t M = sample_.associations.size() + sample_.remaining_cols.size();
	const size_t V = cost_mat.cost_rows.rows();

	std::vector<size_t> indices(M);
	std::vector<size_t> associations(M);
	Sample sample_out;
	sample_out.remaining_rows = std::vector<std::pair<size_t, size_t>>(sample_.remaining_rows);
	std::vector<double> weights_remaining(V);							//*general*//

	//se escribe el sample inicial
	{
		size_t t = 0;
		for(auto &a: sample_.associations)
		{
			indices[t] = a.second;
			associations[t] = a.first;
			++t;
		}

		for(auto &rc: sample_.remaining_cols)
		{
			indices[t] = rc.first;
			associations[t] = NN + rc.second;
			++t;
		}
	}

	double w, wmax;
	int number_associated = sample_.associations.size();

	for(size_t t = 0; t < M; t++)
	{
		size_t j = indices[t];				//indice de la columna en la matriz de asociaciones
		size_t ic = associations[t];		//indice de la columna actual asociada a la fila "j"

		const size_t R = sample_out.remaining_rows.size();

		vector<double> weights(R + V + 1);

		bool is_detected = ic < NN;

		//se llenan los pesos de los pesos correspondientes a las mis-detecciones
		wmax = cost_mat.cost_rows(0, j);
		weights[R] = wmax;	//v=0
		for(size_t v = 1; v < V; v++)
		{
			w = cost_mat.cost_rows(v, j);
			if(w > wmax) wmax = w;
			weights[R + v] = w;
		}

		//se llenan los pesos de los estados "libres" que son misdetection
		for(size_t r = 0; r < R; r++)
		{
			size_t i = sample_out.remaining_rows[r].first;
			size_t v = sample_out.remaining_rows[r].second;				//*general*//
			w = cost_mat.cost_main(i, j) - cost_mat.cost_cols(i, v);	//*general*//
			if(w > wmax) wmax = w;
			weights[r] = w;
		}

		if(is_detected)
		{
			w = cost_mat.cost_main(ic, j) - cost_mat.sum_cost_cols(ic);	//*general*//
			if(w > wmax) wmax = w;
			weights[R + V] = std::exp(w - wmax);	//se normaliza el peso
		}
		else
		{
			weights[R + V] = 0;
		}

		//se normalizan todos los pesos
		for(size_t r = 0; r < R + V; r++)
		{
			weights[r] = std::exp(weights[r] - wmax);
		}

		boost::random::discrete_distribution<> assignment(weights.begin(), weights.end());
		size_t k = assignment(gen);	//se realiza el muestreo
		size_t is = (k < R)?(sample_out.remaining_rows[k].first):(NN + k - R);

		if(is == NN + V) is = ic;

		if(is == ic) continue;		//se obtuvo el mismo muestreo anterior, por lo tanto no se hace nada

		if(is_detected)
		{
			std::pair<size_t, size_t> new_row;
			new_row.first = ic;

			//***************************general***************************//
			//se samplea a que columna de la matriz cost_cols correspondiente
			wmax = cost_mat.cost_cols(ic, 0);
			weights_remaining[0] = wmax;
			for(size_t v = 1; v < V; v++)
			{
				w = cost_mat.cost_cols(ic, v);
				if(w > wmax) wmax = w;
				weights_remaining[v] = w;
			}
			for(size_t v = 0; v < V; v++)
			{
				weights_remaining[v] = std::exp(weights_remaining[v] - wmax);
			}
			boost::random::discrete_distribution<> assignment_remaining(weights_remaining.begin(), weights_remaining.end());
			new_row.second = assignment_remaining(gen);	//se realiza el muestreo
			//***************************general***************************//

			sample_out.remaining_rows.push_back(new_row);
		}

		if(k < R)
		{
			associations[t] = sample_out.remaining_rows[k].first;
			// Deletes the element of index k
			sample_out.remaining_rows.erase(sample_out.remaining_rows.begin() + k);
			if(!is_detected)
				number_associated++;
		}
		else
		{
			associations[t] = is;
			if(is_detected)
				number_associated--;
		}
	}


	//se escribe la solucion en formato de salida (clase Sample)
	sample_out.associations.resize(number_associated);
	sample_out.remaining_cols.resize(M - number_associated);

	auto a = sample_out.associations.begin();
	auto rc = sample_out.remaining_cols.begin();

	for(size_t t = 0; t < M; t++)
	{
		size_t j = indices[t];
		size_t i = associations[t];
		if(i < NN)
		{
			a->first = i;
			a->second = j;
			++a;
		}
		else
		{
			rc->first = j;
			rc->second = i - NN;
			++rc;
		}
	}

	return sample_out;
}

CostMat::CostMat(Eigen::ArrayXXd &cost_main, Eigen::ArrayXXd &cost_rows, Eigen::ArrayXXd &cost_cols)
{
	this->cost_main = Eigen::ArrayXXd(cost_main);
	this->cost_rows = Eigen::ArrayXXd(cost_rows);
	this->cost_cols = Eigen::ArrayXXd(cost_cols);
	//this->sum_cost_rows = cost_rows.colwise().sum();
	//this->sum_cost_cols = cost_cols.rowwise().sum();
	this->logsumexp();
}

void CostMat::logsumexp()
{
	auto logsumexp = [] (const Eigen::ArrayXd &logw)
	{
		double maxw = logw.maxCoeff();
		return std::log((logw - maxw).exp().sum()) + maxw;
	};
	// El calculo por filas y columnas se puede hacer en threads separados
	// ya que son independientes

	size_t P = cost_rows.rows();
	size_t M = cost_rows.cols();
	sum_cost_rows.resize(M);
	// suma de matriz de filas
	// esto tambien se puede paralelizar
	for(size_t j = 0; j < M; ++j)
	{
		sum_cost_rows(j) = logsumexp(cost_rows.block(0, j, P, 1));
	}

	size_t V = cost_cols.cols();
	size_t N = cost_cols.rows();
	sum_cost_cols.resize(N);
	// suma de matriz de columnas
	// esto tambien se puede paralelizar
	for(size_t i = 0; i < N; ++i)
	{
		//Mejorar implementacion para mejor performance
		sum_cost_cols(i) = logsumexp(cost_cols.block(i, 0, 1, V).transpose());
	}
}


