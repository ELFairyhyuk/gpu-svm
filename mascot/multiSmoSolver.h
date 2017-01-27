//
// Created by ss on 16-12-14.
//

#ifndef MASCOT_SVM_MULTISMOSOLVER_H
#define MASCOT_SVM_MULTISMOSOLVER_H

#include "svmProblem.h"
#include "svmModel.h"
#include "../svm-shared/Cache/cache.h"
#include "../svm-shared/HessianIO/baseHessian.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"
#include "../svm-shared/Cache/gpuCache.h"
#include "../svm-shared/baseSMO.h"
#include "../svm-shared/svmParam.h"

class MultiSmoSolver: public BaseSMO
{
public:
    MultiSmoSolver(const SvmProblem &problem, SvmModel &model, const SVMParam &param);

    ~MultiSmoSolver(){
    };
    void solve();

private:
    const SvmProblem &problem;
    SvmModel &model;
    const SVMParam &param;

    CCache *gpuCache;
	GpuCache cache;
    DeviceHessianOnFly *hessianCalculator;

    void init4Training(const SvmProblem &subProblem);

    void SelectFirst(int begin1, int end1, int begin2, int end2, float_point CforPositive);
    void SelectSecond(int begin1, int end1, int begin2, int end2, float_point CforNegative);
    bool iterate(SvmProblem &subProblem, float_point C, int begin1, int end1, int begin2, int end2);
    int getHessianRow(int rowIndex);

    void extractModel(const SvmProblem &subProblem, vector<int> &svIndex, vector<float_point> &coef, float_point &rho) const;
    void deinit4Training();

    virtual float_point *ObtainRow(int numTrainingInstance)
    {
    	cache.getHessianRow(IdofInstanceOne, devHessianInstanceRow1);
    	return devHessianInstanceRow1;
    }

    float_point *devHessianMatrixCache;
    int numOfElementEachRowInCache;
    int cacheSize;
    float_point *devSteep;

};


#endif //MASCOT_SVM_MULTISMOSOLVER_H
