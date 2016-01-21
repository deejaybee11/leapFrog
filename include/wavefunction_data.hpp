/**The MIT License (MIT)
*
*Copyright (c) 2016 Dylan
*
*Permission is hereby granted, free of charge, to any person obtaining a copy
*of this software and associated documentation files (the "Software"), to deal
*in the Software without restriction, including without limitation the rights
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the Software is
*furnished to do so, subject to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all
*copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*SOFTWARE.
*/

#ifndef _WAVEFUNCTION_DATA_H
#define _WAVEFUNCTION_DATA_H

#include <stdlib.h>

#include "mkl.h"

#include "simulation_data.hpp"

class WavefunctionData {
public:
	WavefunctionData(SimulationData &sim_data);
	~WavefunctionData();
	MKL_Complex16 *psi;
	MKL_Complex16 *psi_old;
	MKL_Complex16 *psi_new;
	MKL_Complex16 *conj_psi;
	MKL_Complex16 *psi_temp;	
	double *psi_abs2;

	void calc_norm(SimulationData &sim_data, MKL_Complex16 *wf);
	void normalize_wf(SimulationData &sim_data, MKL_Complex16 *wf);
	double get_norm() { return this->wavefunction_norm; };
	
private:
	double wavefunction_norm;

};

#endif    //    _WAVEFUNCTION_DATA_H
