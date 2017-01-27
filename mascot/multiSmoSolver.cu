//
// Created by ss on 16-12-14.
//

#include <zconf.h>
#include <sys/time.h>
#include <cfloat>
#include "multiSmoSolver.h"
#include "../svm-shared/constant.h"
#include "cuda_runtime.h"
#include "../svm-shared/smoGPUHelper.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"
#include "../SharedUtility/Timer.h"
#include "trainClassifier.h"

#include "../svm-shared/devUtility.h"

__global__ void GetBlockMinYiGValue2(float_point *pfYiFValue, float_point *pfAlpha, int *pnLabel, float_point fPCost,
                                     int begin1, int end1, int begin2, int end2, float_point *pfBlockMin,
                                     int *pnBlockMinGlobalKey) {
    __shared__ float_point fTempLocalYiFValue[BLOCK_SIZE];
    __shared__ int nTempLocalKeys[BLOCK_SIZE];

    int nGlobalIndex;
    int nThreadId = threadIdx.x;
    //global index for thread
    nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    fTempLocalYiFValue[nThreadId] = FLT_MAX;
    if ((nGlobalIndex >= begin1 && nGlobalIndex < end1) || (nGlobalIndex >= begin2 && nGlobalIndex < end2))
//        if(nGlobalIndex < end2 )
    {
        float_point fAlpha;
        int nLabel;
        fAlpha = pfAlpha[nGlobalIndex];
        nLabel = pnLabel[nGlobalIndex];
        //fill yi*GValue in a block
        if ((nLabel > 0 && fAlpha < fPCost) || (nLabel < 0 && fAlpha > 0)) {
            //I_0 is (fAlpha > 0 && fAlpha < fCostP). This condition is covered by the following condition
            //index set I_up
            fTempLocalYiFValue[nThreadId] = pfYiFValue[nGlobalIndex];
            nTempLocalKeys[nThreadId] = nGlobalIndex;
        }
    }
    __syncthreads();    //synchronize threads within a block, and start to do reduce

    GetMinValueOriginal(fTempLocalYiFValue, nTempLocalKeys, blockDim.x);

    if (nThreadId == 0) {
        int nBlockId = blockIdx.y * gridDim.x + blockIdx.x;
        pfBlockMin[nBlockId] = fTempLocalYiFValue[0];
        pnBlockMinGlobalKey[nBlockId] = nTempLocalKeys[0];
    }
}

__global__ void GetBlockMinLowValue2(float_point *pfYiFValue, float_point *pfAlpha, int *pnLabel, float_point fNCost,
                                     int begin1, int end1, int begin2, int end2, float_point *pfDiagHessian,
                                     float_point *pfHessianRow,
                                     float_point fMinusYiUpValue, float_point fUpValueKernel, float_point *pfBlockMin,
                                     int *pnBlockMinGlobalKey, float_point *pfBlockMinYiFValue) {
    __shared__ int nTempKey[BLOCK_SIZE];
    __shared__ float_point fTempMinValues[BLOCK_SIZE];

    int nThreadId = threadIdx.x;
    int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread

    fTempMinValues[nThreadId] = FLT_MAX;
    //fTempMinYiFValue[nThreadId] = FLT_MAX;

    //fill data (-b_ij * b_ij/a_ij) into a block
    float_point fYiGValue;
    float_point fBeta;
    int nReduce = NOREDUCE;
    float_point fAUp_j;
    float_point fBUp_j;

    if ((nGlobalIndex >= begin1 && nGlobalIndex < end1) || (nGlobalIndex >= begin2 && nGlobalIndex < end2))
//        if(nGlobalIndex < end2)
    {
        float_point fUpValue = fMinusYiUpValue;
        fYiGValue = pfYiFValue[nGlobalIndex];
        float_point fAlpha = pfAlpha[nGlobalIndex];

        nTempKey[nThreadId] = nGlobalIndex;

        int nLabel = pnLabel[nGlobalIndex];
        /*************** calculate b_ij ****************/
        //b_ij = -Gi + Gj in paper, but b_ij = -Gi + y_j * Gj in the code of libsvm. Here we follow the code of libsvm
        fBUp_j = fUpValue + fYiGValue;

        if (((nLabel > 0) && (fAlpha > 0)) ||
            ((nLabel < 0) && (fAlpha < fNCost))
                ) {
            fAUp_j = fUpValueKernel + pfDiagHessian[nGlobalIndex] - 2 * pfHessianRow[nGlobalIndex];

            if (fAUp_j <= 0) {
                fAUp_j = TAU;
            }

            if (fBUp_j > 0) {
                nReduce = REDUCE1 | REDUCE0;
            } else
                nReduce = REDUCE0;

            //for getting optimized pair
            //fBeta = -(fBUp_j * fBUp_j / fAUp_j);
            fBeta = __fdividef(__powf(fBUp_j, 2.f), fAUp_j);
            fBeta = -fBeta;
            //fTempMinYiFValue[nThreadId] = -fYiGValue;
        }
    }

    if ((nReduce & REDUCE0) != 0) {
        fTempMinValues[threadIdx.x] = -fYiGValue;
    }
    __syncthreads();
    GetMinValueOriginal(fTempMinValues, blockDim.x);
    int nBlockId;
    if (nThreadId == 0) {
        nBlockId = blockIdx.y * gridDim.x + blockIdx.x;
        pfBlockMinYiFValue[nBlockId] = fTempMinValues[0];
    }

    fTempMinValues[threadIdx.x] = (((nReduce & REDUCE1) != 0) ? fBeta : FLT_MAX);

    //block level reduce
    __syncthreads();
    GetMinValueOriginal(fTempMinValues, nTempKey, blockDim.x);
    __syncthreads();

    if (nThreadId == 0) {
        pfBlockMin[nBlockId] = fTempMinValues[0];
        pnBlockMinGlobalKey[nBlockId] = nTempKey[0];
    }
}

__global__ void getSteep(const float_point *YiFValue, const int *labels, const float_point *alpha, int numOfInstances,
                         float_point *steep, float_point C) {
    int globalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread
    if (globalIndex < numOfInstances) {
        float_point gradient = -YiFValue[globalIndex] * labels[globalIndex];
        if (gradient > 0)
            steep[globalIndex] = gradient * (C - alpha[globalIndex]);
        else
            steep[globalIndex] = gradient * (-alpha[globalIndex]);
    }
}

void MultiSmoSolver::solve() {
    int nrClass = problem.getNumOfClasses();

    if (model.vC.size() == 0) {//initialize C for all the binary classes
        model.vC = vector<float_point>(nrClass * (nrClass - 1) / 2, param.C);
    }

    //train nrClass*(nrClass-1)/2 binary models
    int k = 0;
    for (int i = 0; i < nrClass; ++i) {
        for (int j = i + 1; j < nrClass; ++j) {
            printf("training classifier with label %d and %d\n", i, j);
            SvmProblem subProblem = problem.getSubProblem(i, j);
            init4Training(subProblem);
            cache.enable(i, j, subProblem);
            cacheSize = 2000;
            assert(cacheSize % 2 == 0);
            assert(cacheSize <= subProblem.count[0]);
            assert(cacheSize <= subProblem.count[1]);
            int begin1 = 0;
            int begin2 = subProblem.start[1];
            bool isOptimal = false;
            vector<float_point> steep(subProblem.getNumOfSamples());
            printf("#positive ins %d, #negatice ins %d\n", subProblem.count[0], subProblem.count[1]);
            for (int l = 0; l < 20; ++l) {
                bool mature = false;
                getSteep << < gridSize, BLOCK_SIZE >> >
                                        (devYiGValue, devLabel, devAlpha, subProblem.getNumOfSamples(), devSteep, model.vC[k]);
                checkCudaErrors(
                        cudaMemcpy(steep.data(), devSteep, sizeof(float_point) * steep.size(), cudaMemcpyDeviceToHost));
                float_point maxSteep = -1;
                for (int m = 0; m < subProblem.count[0] - cacheSize / 2; ++m) {
                    float_point sumSteep = 0;
                    for (int n = m; n < m + cacheSize / 2; ++n) {
                        sumSteep += steep[n];
                    }
                    if (sumSteep > maxSteep) {
                        maxSteep = sumSteep;
                        begin1 = m;
                    }
                }
                maxSteep = -1;
                for (int m = subProblem.start[1]; m < subProblem.start[1] + subProblem.count[1] - cacheSize / 2; ++m) {
                    float_point sumSteep = 0;
                    for (int n = m; n < m + cacheSize / 2; ++n) {
                        sumSteep += steep[n];
                    }
                    if (sumSteep > maxSteep) {
                        maxSteep = sumSteep;
                        begin2 = m;
                    }
                }
                int maxIter = (subProblem.getNumOfSamples() > INT_MAX / ITERATION_FACTOR
                               ? INT_MAX
                               : ITERATION_FACTOR * subProblem.getNumOfSamples()) * 4;
                int numOfIter;
                printf("begin1 %d, end1 %d, begin2 %d, begin2 %d\n", begin1, begin1 + cacheSize / 2, begin2,
                       begin2 + cacheSize / 2);
                TIMER_START(iterationTimer)
                for (numOfIter = 0; numOfIter < maxIter; numOfIter++) {
                    if (numOfIter % 100 == 0) {
                    }
                    if (iterate(subProblem, model.vC[k], begin1, begin1 + cacheSize / 2, begin2,
                                begin2 + cacheSize / 2)) {
                        if (numOfIter < cacheSize)
                            mature = true;
                        break;
                    }
                    if (numOfIter % 1000 == 0 && numOfIter != 0) {
                        std::cout << ".";
                        std::cout.flush();
                    }
                }
                cout << "# of iteration: " << numOfIter << endl;
                TIMER_STOP(iterationTimer)
                printf("local up + low = %f\n", upValue + lowValue);
                SelectFirst(0, subProblem.getNumOfSamples(), 0, 0, model.vC[k]);
                SelectSecond(0, subProblem.getNumOfSamples(), 0, 0, model.vC[k]);
                lowValue = -hostBuffer[3];
                printf("global up + low = %f\n", upValue + lowValue);
                if (upValue + lowValue <= EPS)
                    isOptimal = true;
                if (mature) break;
            }
            if (!isOptimal) {
                printf("start convergence\n");
                int maxIter = (subProblem.getNumOfSamples() > INT_MAX / ITERATION_FACTOR
                               ? INT_MAX
                               : ITERATION_FACTOR * subProblem.getNumOfSamples()) * 4;
                int numOfIter;
                cacheSize = subProblem.getNumOfSamples();
                TIMER_START(iterationTimer)
                for (numOfIter = 0; numOfIter < maxIter; numOfIter++) {
                    if (iterate(subProblem, model.vC[k], 0, subProblem.count[0], subProblem.start[1],
                                subProblem.start[1] + subProblem.count[1]))
                        break;
                    if (numOfIter % 1000 == 0 && numOfIter != 0) {
                        std::cout << ".";
                        std::cout.flush();
                    }

                }
                cout << "# of iteration: " << numOfIter << endl;
                TIMER_STOP(iterationTimer)
            }
            cache.disable(i, j);

            vector<int> svIndex;
            vector<float_point> coef;
            float_point rho;
            extractModel(subProblem, svIndex, coef, rho);

            model.addBinaryModel(subProblem, svIndex, coef, rho, i, j);
            k++;
            deinit4Training();
        }
    }
}

bool MultiSmoSolver::iterate(SvmProblem &subProblem, float_point C, int begin1, int end1, int begin2, int end2) {
    int trainingSize = subProblem.getNumOfSamples();
    SelectFirst(begin1, end1, begin2, end2, C);
    SelectSecond(begin1, end1, begin2, end2, C);


    IdofInstanceTwo = int(hostBuffer[0]);

    //get kernel value K(Sample1, Sample2)
    float_point fKernelValue = 0;
    float_point fMinLowValue;
    fMinLowValue = hostBuffer[1];
    fKernelValue = hostBuffer[2];


    cache.getHessianRow(IdofInstanceTwo, devHessianInstanceRow2);


    lowValue = -hostBuffer[3];
    //check if the problem is converged
    if (upValue + lowValue <= EPS) {
        //cout << upValue << " : " << lowValue << endl;
        //m_pGPUCache->PrintCachingStatistics();
        return true;
    }

    float_point fY1AlphaDiff, fY2AlphaDiff;
    float_point fMinValue = -upValue;
    TIMER_START(updateAlphaTimer)
    UpdateTwoWeight(fMinLowValue, fMinValue, IdofInstanceOne, IdofInstanceTwo, fKernelValue,
                    fY1AlphaDiff, fY2AlphaDiff, subProblem.v_nLabels.data(), C);

    TIMER_STOP(updateAlphaTimer)
    TIMER_START(updateGTimer)
    UpdateYiGValue(trainingSize, fY1AlphaDiff, fY2AlphaDiff);
    TIMER_STOP(updateGTimer)

    return false;
}

void MultiSmoSolver::init4Training(const SvmProblem &subProblem) {
    unsigned int trainingSize = subProblem.getNumOfSamples();

    checkCudaErrors(cudaMalloc((void **) &devAlpha, sizeof(float_point) * trainingSize));
    checkCudaErrors(cudaMalloc((void **) &devYiGValue, sizeof(float_point) * trainingSize));
    checkCudaErrors(cudaMalloc((void **) &devLabel, sizeof(int) * trainingSize));

    checkCudaErrors(cudaMemset(devAlpha, 0, sizeof(float_point) * trainingSize));
    vector<float_point> negatedLabel(trainingSize);
    for (int i = 0; i < trainingSize; ++i) {
        negatedLabel[i] = -subProblem.v_nLabels[i];
    }
    checkCudaErrors(cudaMemcpy(devYiGValue, negatedLabel.data(), sizeof(float_point) * trainingSize,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(devLabel, subProblem.v_nLabels.data(), sizeof(int) * trainingSize, cudaMemcpyHostToDevice));

    InitSolver(trainingSize);//initialise base solver

    checkCudaErrors(cudaMalloc((void **) &devHessianInstanceRow1, sizeof(float_point) * trainingSize));
    checkCudaErrors(cudaMalloc((void **) &devHessianInstanceRow2, sizeof(float_point) * trainingSize));

    for (int j = 0; j < trainingSize; ++j) {
        hessianDiag[j] = 1;//assume using RBF kernel
    }
    checkCudaErrors(
            cudaMemcpy(devHessianDiag, hessianDiag, sizeof(float_point) * trainingSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &devSteep, sizeof(float_point) * trainingSize));
}

void MultiSmoSolver::deinit4Training() {
    checkCudaErrors(cudaFree(devAlpha));
    checkCudaErrors(cudaFree(devYiGValue));
    checkCudaErrors(cudaFree(devLabel));

    DeInitSolver();

    checkCudaErrors(cudaFree(devHessianInstanceRow1));
    checkCudaErrors(cudaFree(devHessianInstanceRow2));
    checkCudaErrors(cudaFree(devSteep));
}

int MultiSmoSolver::getHessianRow(int rowIndex) {
    int cacheLocation;
    bool cacheFull = false;
    bool cacheHit = gpuCache->GetDataFromCache(rowIndex, cacheLocation, cacheFull);
    if (!cacheHit) {
        if (cacheFull)
            gpuCache->ReplaceExpired(rowIndex, cacheLocation, NULL);
        hessianCalculator->ReadRow(rowIndex, devHessianMatrixCache + cacheLocation * numOfElementEachRowInCache, 0,
                                   100);
    }
    return cacheLocation * numOfElementEachRowInCache;
}


void MultiSmoSolver::extractModel(const SvmProblem &subProblem, vector<int> &svIndex, vector<float_point> &coef,
                                  float_point &rho) const {
    const unsigned int trainingSize = subProblem.getNumOfSamples();
    vector<float_point> alpha(trainingSize);
    const vector<int> &label = subProblem.v_nLabels;
    checkCudaErrors(cudaMemcpy(alpha.data(), devAlpha, sizeof(float_point) * trainingSize, cudaMemcpyDeviceToHost));
    for (int i = 0; i < trainingSize; ++i) {
        if (alpha[i] != 0) {
            coef.push_back(label[i] * alpha[i]);
            svIndex.push_back(i);
        }
    }
    rho = (lowValue - upValue) / 2;
    printf("# of SV %lu\nbias = %f\n", svIndex.size(), rho);
}

void
MultiSmoSolver::SelectFirst(int begin1, int end1, int begin2, int end2, float_point CforPositive) {
    TIMER_START(selectTimer)
    GetBlockMinYiGValue2 << < gridSize, BLOCK_SIZE >> >
                                        (devYiGValue, devAlpha, devLabel, CforPositive,
                                                begin1, end1, begin2, end2, devBlockMin, devBlockMinGlobalKey);
    //global reducer
    GetGlobalMin << < 1, BLOCK_SIZE >> >
                         (devBlockMin, devBlockMinGlobalKey, numOfBlock, devYiGValue, NULL, devBuffer);

    //copy result back to host
    cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 2, cudaMemcpyDeviceToHost);
    IdofInstanceOne = (int) hostBuffer[0];
    TIMER_STOP(selectTimer)

    cache.getHessianRow(IdofInstanceOne, devHessianInstanceRow1);
}

void
MultiSmoSolver::SelectSecond(int begin1, int end1, int begin2, int end2, float_point CforNegative) {
    TIMER_START(selectTimer)
    float_point fUpSelfKernelValue = 0;
    fUpSelfKernelValue = hessianDiag[IdofInstanceOne];

    //for selecting the second instance
    float_point fMinValue;
    fMinValue = hostBuffer[1];
    upValue = -fMinValue;

    //get block level min (-b_ij*b_ij/a_ij)
    GetBlockMinLowValue2 << < gridSize, BLOCK_SIZE >> >
                                        (devYiGValue, devAlpha, devLabel, CforNegative, begin1, end1, begin2, end2, devHessianDiag,
                                                devHessianInstanceRow1, upValue, fUpSelfKernelValue, devBlockMin, devBlockMinGlobalKey,
                                                devBlockMinYiGValue);

    //get global min
    GetGlobalMin << < 1, BLOCK_SIZE >> >
                         (devBlockMin, devBlockMinGlobalKey,
                                 numOfBlock, devYiGValue, devHessianInstanceRow1, devBuffer);

    //get global min YiFValue
    //0 is the size of dynamically allocated shared memory inside kernel
    GetGlobalMin << < 1, BLOCK_SIZE >> > (devBlockMinYiGValue, numOfBlock, devBuffer);

    //copy result back to host
    cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 4, cudaMemcpyDeviceToHost);
    TIMER_STOP(selectTimer)
}

MultiSmoSolver::MultiSmoSolver(const SvmProblem &problem, SvmModel &model, const SVMParam &param) :
        problem(problem), model(model), param(param), cache(problem, param, problem.isBinary()) {
}
