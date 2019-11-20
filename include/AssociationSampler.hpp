#ifndef ASSOCIATIONSAMPLER_HPP_
#define ASSOCIATIONSAMPLER_HPP_
#include <Eigen/Dense>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <memory>

class CostMat
{
public:
	Eigen::ArrayXXd cost_main;
	Eigen::ArrayXXd cost_rows;
	Eigen::ArrayXXd cost_cols;
	Eigen::ArrayXd sum_cost_rows;
	Eigen::ArrayXd sum_cost_cols;

	/**
	 * Constructor
	 */
	CostMat() {}

	/**
	 * Destructor
	 */
	virtual ~CostMat() {}

	/**
	 * Constructor
	 * @param cost_main matriz (principal) de costos de asociaciones
	 * @param cost_rows matriz de costos de filas, usualmente 2xM
	 * @param cost_cols matriz de costos de columnas, usualmente Nx2
	 */
	CostMat(Eigen::ArrayXXd &cost_main, Eigen::ArrayXXd &cost_rows, Eigen::ArrayXXd &cost_cols);

	/**
	 * Calcula (internamente) las matrices de suma de cost_rows y cost_cols
	 */
	void logsumexp(); //se calculan las sumas de filas y columnas sum_cost_rows y sum_cost_cols


};

class Sample
{
public:
	std::vector<std::pair<size_t, size_t>> associations;
	std::vector<std::pair<size_t, size_t>> remaining_rows;
	std::vector<std::pair<size_t, size_t>> remaining_cols;

	/**
	 * Constructor
	 */
	Sample() {};

	/**
	 * Destructor
	 */
	virtual ~Sample() {}

	/**
	 *
	 * @param
	 */
	Sample(const Sample& sample_);

	/**
	 * Order the associations
	 */
	void sort();

	/**
	 * Assignment operator
	 * \param[in] rhs the right-hand-side from which data is compared
	 */
	bool operator==(const Sample& rhs) const;

};


enum class SampleBy {ANY, ROW, COL, ROWG, COLG}; /**<Tipos de posibles formas de samplear */


class GenericSampler
{
public:
	/**
	 * Constructor
	 */
	GenericSampler() {}

	/**
	 * Destructor
	 */
	virtual ~GenericSampler() {}

	/**
	 * Funcion que evalua una ssignacion
	 * @param cost_mat matrices de costos
	 * @param sample_ un sample
	 * @return peso de la asignacion
	 */
	double eval_assignment(const CostMat &cost_mat, const Sample &sample_);

	/**
	 * Funcion que hace un sample generico.
	 * No esta implementada y se espera que la clase hija la implemente
	 * @param gen generador de numeros aleatorios
	 * @param cost_mat matrices de costos
	 * @param sample_ sample de entrada
	 * @param flag tipo de sampling (por filas o por columnas). Ver "sample_by"
	 * @return un sample de salida
	 */
	virtual Sample sample(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_,  SampleBy flag = SampleBy::ROW) = 0;

	/**
	 * Funcion que hace multiples samples dado un sample inicial
	 * @param gen generador de numeros aleatorios
	 * @param cost_mat matrices de costos
	 * @param sample_ sample de entrada
	 * @param number_of_samples
	 * @param flag tipo de sampling (por filas o por columnas). Ver "sample_by"
	 * @return multiples samples de salida
	 */
	std::vector<Sample> samples(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_, size_t number_of_samples, SampleBy flag = SampleBy::ROW);



};

class GibbsSampler : public GenericSampler
{
public:
	/**
	 * Constructor
	 */
	GibbsSampler() {}

	/**
	 * Destructor
	 */
	virtual ~GibbsSampler() {}

	/**
	 * Implementacion de Gibbs del sampling.
	 * Los parametros son los mismos de GenericSampler
	 * @param gen
	 * @param cost_mat
	 * @param sample_
	 * @param flag
	 * @return
	 */
	virtual Sample sample(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_,  SampleBy flag = SampleBy::ROW);

//private:
	/**
	 * Hace lo mismo que sample, pero por fila, por lo que no usa flag
	 * Se utiliza cuando la matriz de costos (cost_rows) por fila no existe
	 * @param gen
	 * @param cost_mat
	 * @param sample_
	 * @return
	 */
	Sample sample_by_row(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_);

	/**
	 * Hace lo mismo que sample, pero por columna, por lo que no usa flag
	 * Se utiliza cuando la matriz de costos (cost_cols) por columna no existe
	 * @param gen
	 * @param cost_mat
	 * @param sample_
	 * @return
	 */
	Sample sample_by_column(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_);

	/**
	 * Hace lo mismo que sample, pero por fila, por lo que no usa flag
	 * Se utiliza cuando la matriz de costos (cost_rows) por fila tiene multiples filas
	 * POR IMPLEMENTAR
	 * @param gen
	 * @param cost_mat
	 * @param sample_
	 * @return
	 */
	Sample sample_by_row_g(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_);

	/**
	 * Hace lo mismo que sample, pero por columna, por lo que no usa flag
	 * Se utiliza cuando la matriz de costos (cost_cols) por columna tiene multiples columnas
	 * POR IMPLEMENTAR
	 * @param gen
	 * @param cost_mat
	 * @param sample_
	 * @return
	 */
	Sample sample_by_column_g(boost::mt19937 &gen, const CostMat &cost_mat, const Sample &sample_);

	/**
	 * Sample a random number k with probabilities proportional to the weights
	 * @param weights probabilities will be proportional to this
	 * @return random k from 0 to weights.size()-1
	 */
	static size_t sample(boost::mt19937 &gen, std::vector<double> &weights);
};

#endif //SAMPLER_H_
