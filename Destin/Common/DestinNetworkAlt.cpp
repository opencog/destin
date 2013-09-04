#include "DestinNetworkAlt.h"
#include <stdio.h>

void DestinNetworkAlt::initTemperatures(int layers, uint * centroids){
    temperatures = new float[layers];

    for(int l = 0 ; l < layers ; l++){
        temperatures[l] = centroids[l];
    }

    if(layers >= 2){
        temperatures[layers - 1] *= 2;
    }
    if(layers >=3){
        temperatures[layers - 2] *= 2;
    }
}

float *** DestinNetworkAlt::getCentroidImages(){
    if(centroidImages==NULL){
        centroidImages = Cig_CreateCentroidImages(destin, centroidImageWeightParameter);
    }
    return centroidImages;
}

//TODO: comment extRatio parameter
DestinNetworkAlt::DestinNetworkAlt(SupportedImageWidths width, unsigned int layers,
        unsigned int centroid_counts [], bool isUniform, int extRatio) :
        training(true),
        beta(.01),
        lambda(.1),
        gamma(.1),
        isUniform(isUniform),
        centroidImages(NULL),
        centroidImageWeightParameter(1.0),
        inputImageWidth(width)
        {

    uint input_dimensionality = 16;
    uint c, l;
    callback = NULL;
    initTemperatures(layers, centroid_counts);
    float starv_coef = 0.05;
    uint n_classes = 0;//doesn't look like its used
    uint num_movements = 0; //this class does not use movements

    //figure out how many layers are needed to support the given
    //image width.
    bool supported = false;
    for (c = 4, l = 1; c <= MAX_IMAGE_WIDTH ; c *= 2, l++) {
        if (c == width) {
            supported = true;
            break;
        }
    }
    if(!supported){
        throw std::logic_error("given image width is not supported.");
    }
    if (layers != l) {
        throw std::logic_error("Image width does not match the given number of layers.");
    }
    destin = InitDestin(
            input_dimensionality,
            layers,
            centroid_counts,
            n_classes,
            beta,
            lambda,
            gamma,
            temperatures,
            starv_coef,
            num_movements,
            isUniform,
            extRatio
     );

    setBeliefTransform(DST_BT_NONE);
    ClearBeliefs(destin);
    SetLearningStrat(destin, CLS_FIXED);
    destin->fixedLearnRate = 0.1;

    //SetLearningStrat(destin, CLS_DECAY_c1);

    isTraining(true);
}

// 2013.6.3
// CZT
// If adding centroids; Only for uniform!
void DestinNetworkAlt::updateDestin_add(SupportedImageWidths width, unsigned int layers,
        unsigned int centroid_counts [], bool isUniform, int extRatio, int currLayer)
{
    if(!isUniform)
    {
        printf("The adding action NOW is only for Uniform DeSTIN!\n");
        return;
    }

    training = true;
    beta = .01;
    lambda = .1; // 0.1
    gamma = .1;  // 0.1
    isUniform = isUniform;
    centroidImages = NULL;
    centroidImageWeightParameter = 1.0;

    uint input_dimensionality = 16;
    uint c, l;
    callback = NULL;
    initTemperatures(layers, centroid_counts);
    float starv_coef = 0.05;
    uint n_classes = 0;//doesn't look like its used
    uint num_movements = 0; //this class does not use movements

    //figure out how many layers are needed to support the given
    //image width.
    bool supported = false;
    for (c = 4, l = 1; c <= MAX_IMAGE_WIDTH ; c *= 2, l++) {
        if (c == width) {
            supported = true;
            break;
        }
    }
    if(!supported){
        throw std::logic_error("given image width is not supported.");
    }
    if (layers != l) {
        throw std::logic_error("Image width does not match the given number of layers.");
    }
    addCentroid(
            destin,
            input_dimensionality,
            layers,
            centroid_counts,
            n_classes,
            beta,
            lambda,
            gamma,
            temperatures,
            starv_coef,
            num_movements,
            isUniform,
            extRatio,
            currLayer,
            getSharedCentroids(),
            getStarv(),
            getSigma(),
            getPersistWinCounts(),
            getPersistWinCounts_detailed(),
            getAbsvar()
     );

    setBeliefTransform(DST_BT_NONE);
    ClearBeliefs(destin);
    SetLearningStrat(destin, CLS_DECAY_c1); //
    isTraining(true);
}

// 2013.6.6
// CZT
// If killing centroids; Only for uniform!
void DestinNetworkAlt::updateDestin_kill(SupportedImageWidths width, unsigned int layers,
        unsigned int centroid_counts [], bool isUniform, int extRatio, int currLayer, int kill_ind)
{
    if(!isUniform)
    {
        printf("The killing action NOW is only for Uniform DeSTIN!\n");
        return;
    }

    training = true;
    beta = .01;
    lambda = .1; // 0.1
    gamma = .1;  // 0.1
    isUniform = isUniform;
    centroidImages = NULL;
    centroidImageWeightParameter = 1.0;

    uint input_dimensionality = 16;
    uint c, l;
    callback = NULL;
    initTemperatures(layers, centroid_counts);
    float starv_coef = 0.05;
    uint n_classes = 0;//doesn't look like its used
    uint num_movements = 0; //this class does not use movements

    //figure out how many layers are needed to support the given
    //image width.
    bool supported = false;
    for (c = 4, l = 1; c <= MAX_IMAGE_WIDTH ; c *= 2, l++) {
        if (c == width) {
            supported = true;
            break;
        }
    }
    if(!supported){
        throw std::logic_error("given image width is not supported.");
    }
    if (layers != l) {
        throw std::logic_error("Image width does not match the given number of layers.");
    }
    killCentroid(
            destin,
            input_dimensionality,
            layers,
            centroid_counts,
            n_classes,
            beta,
            lambda,
            gamma,
            temperatures,
            starv_coef,
            num_movements,
            isUniform,
            extRatio,
            currLayer,
            kill_ind,
            getSharedCentroids(),
            getStarv(),
            getSigma(),
            getPersistWinCounts(),
            getPersistWinCounts_detailed(),
            getAbsvar()
     );

    setBeliefTransform(DST_BT_NONE);
    ClearBeliefs(destin);
    SetLearningStrat(destin, CLS_DECAY_c1); //
    isTraining(true);
}

/*****************************************************************************/
// 2013.6.4
// CZT
// Get sharedCentroids or mu; Only for uniform!
float ** DestinNetworkAlt::getSharedCentroids()
{
    float ** sharedCentroids;
    MALLOC(sharedCentroids, float *, destin->nLayers);
    int i;
    for(i=0; i<destin->nLayers; ++i)
    {
        Node * currNode = getNode(i, 0, 0);
        MALLOC(sharedCentroids[i], float, currNode->nb*currNode->ns);
        for(int j=0; j<currNode->nb*currNode->ns; ++j)
        {
            sharedCentroids[i][j] = currNode->mu[j];
        }
    }
    return sharedCentroids;
}

// 2013.6.5
// CZT
// Get uf_starv; Only for uniform!
float ** DestinNetworkAlt::getStarv()
{
    float ** starv;
    MALLOC(starv, float *, destin->nLayers);
    int i;
    for(i=0; i<destin->nLayers; ++i)
    {
        MALLOC(starv[i], float, destin->nb[i]);
        for(int j=0; j<destin->nb[i]; ++j)
        {
            starv[i][j] = destin->uf_starv[i][j];
        }
    }
    return starv;
}

// 2013.6.6
// CZT
// Get uf_persistWinCounts; Only for uniform!
long ** DestinNetworkAlt::getPersistWinCounts()
{
    long ** persistWinCounts;
    MALLOC(persistWinCounts, long *, destin->nLayers);
    int i;
    for(i=0; i<destin->nLayers; ++i)
    {
        MALLOC(persistWinCounts[i], long, destin->nb[i]);
        for(int j=0; j<destin->nb[i]; ++j)
        {
            persistWinCounts[i][j] = destin->uf_persistWinCounts[i][j];
        }
    }
    return persistWinCounts;
}

// 2013.6.5
// CZT
// Get uf_sigma; Only for uniform!
float ** DestinNetworkAlt::getSigma()
{
    float ** sigma;
    MALLOC(sigma, float *, destin->nLayers);
    int i;
    for(i=0; i<destin->nLayers; ++i)
    {
        Node * currNode = getNode(i, 0, 0);
        MALLOC(sigma[i], float, currNode->nb*currNode->ns);
        for(int j=0; j<currNode->nb*currNode->ns; ++j)
        {
            sigma[i][j] = destin->uf_sigma[i][j];
        }
    }
    return sigma;
}

// 2013.6.13
// CZT: get uf_persistWinCounts_detailed, only for uniform!
long ** DestinNetworkAlt::getPersistWinCounts_detailed()
{
    long ** persistWinCounts_detailed;
    MALLOC(persistWinCounts_detailed, long *, destin->nLayers);
    int i;
    for(i=0; i<destin->nLayers; ++i)
    {
        MALLOC(persistWinCounts_detailed[i], long, destin->nb[i]);
        for(int j=0; j<destin->nb[i]; ++j)
        {
            persistWinCounts_detailed[i][j] = destin->uf_persistWinCounts_detailed[i][j];
        }
    }
    return persistWinCounts_detailed;
}

// 2013.7.4
// CZT: get uf_absvar, only for uniform;
float ** DestinNetworkAlt::getAbsvar()
{
    float ** absvar;
    MALLOC(absvar, float *, destin->nLayers);
    int i, j;
    for(i=0; i<destin->nLayers; ++i)
    {
        Node * currNode = getNode(i, 0, 0);
        MALLOC(absvar[i], float, currNode->nb*currNode->ns);
        for(j=0; j<currNode->nb*currNode->ns; ++j)
        {
            absvar[i][j] = destin->uf_absvar[i][j];
        }
    }
    return absvar;
}

// 2013.7.4
// CZT: get sep;
float DestinNetworkAlt::getSep(int layer)
{
    Node * currNode = getNode(layer, 0, 0);
    std::vector<float> sep(currNode->nb, 1.0); //vector of length nb, each element initialized to 1.0
    int i,j,k;
    for(i=0; i<currNode->nb; ++i)
    {
        for(j=0; j<currNode->nb; ++j)
        {
            if( i != j )
            {
                float fSum = 0.0;
                float fTemp;
                for(k=0; k<currNode->ni; ++k)
                {
                    fSum += fabs(currNode->mu[i*currNode->ns + k]
                                 - currNode->mu[j*currNode->ns + k]);
                }
                for(k=currNode->ni + currNode->nb + currNode->np + currNode->nc;
                    k < currNode->ns; ++k)
                {
                    fSum += fabs(currNode->mu[i*currNode->ns + k]
                                 - currNode->mu[j*currNode->ns + k]);
                }
                fTemp = fSum / (currNode->ni * (layer==0 ? destin->extRatio : 1));

                if(fTemp < sep[i])
                {
                    sep[i] = fTemp;
                }
            }
        }
    }
    float fSum = 0.0;
    for(i=0; i<currNode->nb; ++i)
    {
        fSum += sep[i];
    }
    return fSum/currNode->nb;
}

// 2013.7.4
// CZT: get var;
float DestinNetworkAlt::getVar(int layer)
{
    Node * currNode = getNode(layer, 0, 0);
    std::vector<float> var(currNode->nb);
    int i,j;
    float fSum;
    for(i=0; i<currNode->nb; ++i)
    {
        fSum = 0.0;
        for(j=0; j<currNode->ni; ++j)
        {
            fSum += destin->uf_absvar[layer][i*currNode->ns + j];
        }
        for(j=currNode->ni+currNode->nb+currNode->np+currNode->nc;
            j<currNode->ns; ++j)
        {
            fSum += destin->uf_absvar[layer][i*currNode->ns + j];
        }
        var[i] = fSum / (currNode->ni * (layer == 0 ? destin->extRatio : 1));
    }
    fSum = 0.0;
    for(i=0; i<currNode->nb; ++i)
    {
        fSum += var[i];
    }
    return fSum/currNode->nb;
}

// 2013.7.4
// CZT: get quality;
float DestinNetworkAlt::getQuality(int layer)
{
    return getSep(layer)-getVar(layer);
}

// CZT: to display all centroids
void DestinNetworkAlt::displayFloatCentroids(int layer)
{
    Node * currNode = getNode(layer, 0, 0);

    // Display all centroids on the selected layer
    printf("---All centroids on layer %d:\n", layer);
    for(int i=0; i<currNode->nb; ++i)
    {
        for(int j=0; j<currNode->ni; ++j)
        {
            printf("%f  ", currNode->mu[i*currNode->ns + j]);
        }
        printf("\n");
        for(int j=0; j<currNode->nb; ++j)
        {
            printf("%f  ", currNode->mu[i*currNode->ns + j + currNode->ni]);
        }
        printf("\n");
        for(int j=0; j<currNode->np; ++j)
        {
            printf("%f  ", currNode->mu[i*currNode->ns + j + currNode->ni + currNode->nb]);
        }
        printf("\n");
        if(i != currNode->nb-1)
        {
            printf("---\n");
        }
    }
    printf("\n\n");
}

// CZT: to display the vector with a given title
void DestinNetworkAlt::displayFloatVector(std::string title, std::vector<float> vec)
{
    printf("%s", title.c_str());
    for(int i=0; i<vec.size(); ++i)
    {
        printf("%f  ", vec[i]);
    }
    printf("\n\n");
}

void DestinNetworkAlt::getSelectedCentroid(int layer, int idx, std::vector<float> & outCen)
{
    // Clear the vector
    outCen.clear();

    Node * currNode = getNode(layer, 0, 0);
    for(int i=currNode->ns*idx; i<currNode->ns*(idx+1); ++i)
    {
        outCen.push_back(currNode->mu[i]);
    }
}

void DestinNetworkAlt::getSelectedSigma(int layer, int idx, std::vector<float> & outSigma)
{
    // Clear the vector
    outSigma.clear();

    Node * currNode = getNode(layer, 0, 0);
    for(int i=currNode->ns*idx; i<currNode->ns*(idx+1); ++i)
    {
        outSigma.push_back(destin->uf_sigma[layer][i]);
    }
}

void DestinNetworkAlt::normalizeChildrenPart(std::vector<float> & inCen, int ni)
{
    for(int i=0; i<4; ++i)
    {
        float sum = 0;
        for(int j=0; j<ni/4; ++j)
        {
            sum += inCen[i*(ni/4) + j];
        }

        for(int j=0; j<ni/4; ++j)
        {
            inCen[i*(ni/4) + j] = inCen[i*(ni/4) + j] / sum;
        }
    }
}

#define EPSILON     1e-8
#define MAX_INTERMEDIATE_BELIEF (1.0 / EPSILON)

// CZT: rescaleCentroid
// 2013.9.2
//   Remove the rescale_up and rescale_down; Focus on the recursive steps;
//   Bug now: the value bigger than 1???
void DestinNetworkAlt::rescaleCentroid(int srcLayer, int idx, int dstLayer)
{
    if(!isUniform)
    {
        printf("The rescaling action NOW is only for Uniform DeSTIN!\n");
        return;
    }

    if(srcLayer == dstLayer)
    {
        printf("You are joking!\n");
        return;
    }

    if(srcLayer < dstLayer)
    {
        std::vector<float> selCen;
        getSelectedCentroid(srcLayer, idx, selCen);
        //
        rescaleRecursiveUp(srcLayer, selCen, dstLayer);
    }
    else if(srcLayer > dstLayer)
    {
        std::vector<float> selCen;
        getSelectedCentroid(srcLayer, idx, selCen);
        //
        rescaleRecursiveDown(srcLayer, selCen, dstLayer);
    }
}

void DestinNetworkAlt::rescaleRecursiveUp(int srcLayer, std::vector<float> selCen, int dstLayer)
{
    if(srcLayer == dstLayer)
    {
        displayFloatVector("---Result is:\n", selCen);
        //displayFloatCentroids(dstLayer);
        return;
    }
    else
    {
        Node * currNode = getNode(srcLayer, 0, 0);
        Node * parentNode = getNode(srcLayer+1, 0, 0);
        // Contain the final result
        std::vector<float> newSelCen;
        // Contain the 4 new made children
        std::vector<std::vector<float> > extendCen;

        // Display all centroids on source layer
        displayFloatCentroids(srcLayer);
        // Display the selected centroid
        displayFloatVector("---The selected centroid is:\n", selCen);

        if(srcLayer == 0)
        {
            // Generate the index for every quarter, specific for layer 0
            std::vector<std::vector<int> > vecIdx;
            // 0,1,4,5
            // 2,3,6,7
            // 8,9,12,13
            // 10,11,14,15
            for(int i=0; i<2; ++i)
            {
                for(int j=0; j<=2; j+=2)
                {
                    std::vector<int> vecTemp;
                    vecTemp.push_back(i*2*4 + j);
                    vecTemp.push_back(i*2*4 + j + 1);
                    vecTemp.push_back(i*2*4 + j + 4);
                    vecTemp.push_back(i*2*4 + j + 4 + 1);
                    vecIdx.push_back(vecTemp);
                }
            }

            // Generate the 4 new observations
            for(int i=0; i<vecIdx.size(); ++i)
            {
                std::vector<float> quaCen;

                for(int j=0; j<vecIdx[i].size(); ++j)
                {
                    for(int k=0; k<4; ++k)
                    {
                        quaCen.push_back(selCen[vecIdx[i][j]]);
                    }
                }
                for(int j=0; j<currNode->nb; ++j)
                {
                    quaCen.push_back(1/(float)currNode->nb);
                }
                for(int j=0; j<currNode->np; ++j)
                {
                    quaCen.push_back(1/(float)currNode->np);
                }
                extendCen.push_back(quaCen);
            }

            // Display the 4 new observations
            printf("---The generated 4 new observations:\n");
            for(int i=0; i<extendCen.size(); ++i)
            {
                for(int j=0; j<currNode->ni; ++j)
                {
                    printf("%f  ", extendCen[i][j]);
                }
                printf("\n");
                for(int j=0; j<currNode->nb; ++j)
                {
                    printf("%f  ", extendCen[i][j + currNode->ni]);
                }
                printf("\n");
                for(int j=0; j<currNode->np; ++j)
                {
                    printf("%f  ", extendCen[i][j + currNode->ni + currNode->nb]);
                }
                printf("\n");
                if(i != extendCen.size()-1)
                {
                    printf("---\n");
                }
            }
            printf("\n\n");/**/

            //normalizeChildrenPart(newSelCen, extendCen.size()*currNode->nb);
        }
        else
        {
            // Generate the 4 new observations
            int numParts = 4;
            Node * childNode = getNode(srcLayer-1, 0, 0);
            for(int i=0; i<numParts; ++i)
            {
                std::vector<float> quaCen;
                for(int j=0; j<numParts; ++j)
                {
                    for(int k=0; k<childNode->nb; ++k)
                    {
                        quaCen.push_back(selCen[i*childNode->nb + k]);
                    }
                }
                for(int j=0; j<currNode->nb; ++j)
                {
                    quaCen.push_back( 1/(float)currNode->nb );
                }
                for(int j=0; j<currNode->np; ++j)
                {
                    quaCen.push_back( 1/(float)currNode->np );
                }

                extendCen.push_back(quaCen);

                // testing
                //displayFloatVector("\n", quaCen);
            }
        }


        // Use the 4 new observations to construct the belief vector for layer 1
        // 4 children part
        for(int i=0; i<extendCen.size(); ++i)
        {
            float delta;
            float sumMal;
            for(int j=0; j<currNode->nb; ++j)
            {
                sumMal = 0;

                // TODO: decrease from 'ns' to 'ni';
                for(int k=0; k<currNode->ni; ++k)
                //for(int k=0; k<currNode->ns; ++k)
                {
                    delta = extendCen[i][k] - currNode->mu[j*currNode->ns + k];

                    delta *= delta;

                    // Assume starv is 1
                    delta *= 1;

                    sumMal += delta/destin->uf_sigma[srcLayer][j*currNode->ns + k];
                }

                sumMal = sqrt(sumMal);

                //newSelCen.push_back( ( sumMal < EPSILON ) ? MAX_INTERMEDIATE_BELIEF : (1.0 / sumMal) );
                newSelCen.push_back( 1 / (1+sumMal) );

                // Because the faked observations are too close to the centroids, the sumMal is
                // less than 1. Then the belief will be larger than 1 according to 1.0/sumMal.
                // In the original DeSTIN, the belief vector will make centroids 'move' even the
                // belief is larger than 1. But what we get directly is not centroid, but belief.
            }
        }
        // current part
        for(int i=0; i<parentNode->nb; ++i)
        {
            newSelCen.push_back( 1/(float)parentNode->nb );
        }
        // parent part
        for(int i=0; i<parentNode->np; ++i)
        {
            newSelCen.push_back( 1/(float)parentNode->np );
        }

        rescaleRecursiveUp(srcLayer+1, newSelCen, dstLayer);
    }
}

void DestinNetworkAlt::rescaleRecursiveDown(int srcLayer, std::vector<float> selCen, int dstLayer)
{
    if(srcLayer == dstLayer)
    {
        displayFloatVector("---Result is:\n", selCen);
        //displayFloatCentroids(dstLayer);
        return;
    }
    else
    {
        Node * childNode = getNode(srcLayer-1, 0, 0);

        std::vector<float> newSelCen;
        std::vector<float> normSelCen;

        // Display all centroids on source layer
        displayFloatCentroids(srcLayer);
        // Display the selected centroid
        displayFloatVector("---The selected centroid is:\n", selCen);

        if(srcLayer == 1)
        {
            // Use every quarter as a single lower level centroid, then downsample
            //int level0 = 0;
            //Node * childNode = getNode(level0, 0, 0);
            //displayFloatCentroids(level0);

            // Normalize for every quarter
            // This is inspired by 'cent_image_gen.c';
            for(int i=0; i<4; ++i)
            {
                float sum = 0.0;
                for(int j=0; j<childNode->nb; ++j)
                {
                    sum += selCen[i*childNode->nb + j];
                }
                for(int j=0; j<childNode->nb; ++j)
                {
                    normSelCen.push_back(selCen[i*childNode->nb + j] / sum);
                }
            }

            // Display the selected centroid
            //displayFloatVector("---The selected centroid is:\n", selCen);
            //displayFloatVector("---The normalized selected centroid is:\n", normSelCen);

            // Downsampling method: pickping left-up one
            std::vector<int> vecIdx;
            vecIdx.push_back(0);
            vecIdx.push_back(2);
            vecIdx.push_back(8);
            vecIdx.push_back(10);

            for(int i=0; i<4; ++i)
            {
                std::vector<float> tempCen(childNode->ni, 0);
                for(int j=0; j<childNode->nb; ++j)
                {
                    for(int k=j*childNode->ns; k<j*childNode->ns+childNode->ni; ++k)
                    {
                        //
                        tempCen[k-j*childNode->ns] += normSelCen[i*childNode->nb+j] * childNode->mu[k];
                    }
                }
                //displayFloatVector("\n", tempCen);
                for(int j=0; j<vecIdx.size(); ++j)
                {
                    newSelCen.push_back(tempCen[vecIdx[j]]);
                }
            }
        }
        else
        {
            // Normalize for every quarter
            for(int i=0; i<4; ++i)
            {
                float sum = 0.0;
                for(int j=0; j<childNode->nb; ++j)
                {
                    sum += selCen[i*childNode->nb + j];
                }
                for(int j=0; j<childNode->nb; ++j)
                {
                    normSelCen.push_back(selCen[i*childNode->nb + j] / sum);
                }
            }

            for(int i=0; i<4; ++i)
            {
                std::vector<float> tempCen(childNode->ni, 0);
                for(int j=0; j<childNode->nb; ++j)
                {
                    for(int k=j*childNode->ns; k<j*childNode->ns+childNode->ni; ++k)
                    {
                        //
                        tempCen[k - j*childNode->ns] += normSelCen[i*childNode->nb + j] * childNode->mu[k];
                    }
                }

                for(int j=0; j<childNode->ni/4; ++j)
                {
                    newSelCen.push_back(tempCen[j]);
                }
            }
        }

        for(int i=0; i<childNode->nb; ++i)
        {
            newSelCen.push_back(1 / (float)childNode->nb);
        }
        for(int i=0; i<childNode->np; ++i)
        {
            newSelCen.push_back(1 / (float)childNode->np);
        }

        rescaleRecursiveDown(srcLayer-1, newSelCen, dstLayer);
    }
}

/*****************************************************************************/

DestinNetworkAlt::~DestinNetworkAlt() {
    if(centroidImages != NULL){
        Cig_DestroyCentroidImages(destin,  centroidImages);
    }

    if(destin!=NULL){
        DestroyDestin(destin);
        destin = NULL;
    }
    if(temperatures!=NULL){
        delete [] temperatures;
        temperatures = NULL;
    }
}

void DestinNetworkAlt::setTemperatures(float temperatures[]){
    memcpy(this->temperatures, temperatures, sizeof(float) * getLayerCount());
    memcpy(destin->temp, temperatures, sizeof(float) * getLayerCount());
    for(int l = 0; l < getLayerCount(); l++){
        for(int n = 0 ; n < destin->layerSize[l]; n++){
            GetNodeFromDestinI(destin, l, n)->temp = temperatures[l];
        }
    }
}

void DestinNetworkAlt::doDestin( //run destin with the given input
        float * input_dev //pointer to input memory on device
        ) {
    FormulateBelief(destin, input_dev);

    if(this->callback != NULL){
        this->callback->callback(*this );
    }
}

void DestinNetworkAlt::isTraining(bool isTraining) {
    this->training = isTraining;
    for(int l = 0 ; l < destin->nLayers ; l++){
       destin->layerMask[l] = isTraining ? 1 : 0;
    }
}

void DestinNetworkAlt::printNodeCentroidPositions(int layer, int row, int col){
    Node * n = getNode(layer, row, col);
    for(int centroid  = 0 ; centroid < n->nb ; centroid++){
        printf("centroid %i: input ", centroid);
        int dimension;
        for(dimension = 0 ; dimension < n->ni ; dimension++){
            if(dimension % (n->ni / 4) == 0){
                printf("\n");
            }
            printf("%.5f ", n->mu[centroid*n->ns + dimension]);
        }
        printf("\n");
        printf(" self ");
        for(int end = dimension + n->nb ; dimension < end ; dimension++){
            printf("%.5f ", n->mu[centroid*n->ns + dimension]);
        }
        printf(" parent ");
        for(int end = dimension + n->np ; dimension < end ; dimension++){
            printf("%.5f ", n->mu[centroid*n->ns + dimension]);
        }
        printf("\n");
    }
}

void DestinNetworkAlt::printWinningCentroidGrid(int layer){
    uint width = (uint)sqrt(destin->layerSize[layer]);
    uint nidx = 0;
    Node * nodelist = GetNodeFromDestin(destin, layer, 0, 0);
    printf("Winning centroids grid - layer %i:\n", layer);
    for( int r  = 0 ; r < width ; r++){
        for(int c = 0 ; c < width ; c++ , nidx++){
            printf("%i ", nodelist[nidx].winner);
        }
        printf("\n");
    }
}

void DestinNetworkAlt::imageWinningCentroidGrid(int layer, int zoom, const string& window_name){
    uint width = (uint)sqrt(destin->layerSize[layer]);

    //initialize or create new grid if needed
    if(winningGrid.rows != width || winningGrid.cols != width){
        winningGrid = cv::Mat(width, width, CV_32FC1);
    }
    int nb = destin->nb[layer];
    float * data = (float *)winningGrid.data;
    for(int n = 0 ; n < destin->layerSize[layer] ; n++){
        data[n]  = (float)GetNodeFromDestinI(destin, layer, n)->winner / (float) nb;
    }
    cv::resize(winningGrid,winningGridLarge, cv::Size(), zoom, zoom, cv::INTER_NEAREST);
    cv::imshow(window_name, winningGridLarge);
    return;
}

void DestinNetworkAlt::printNodeObservation(int layer, int row, int col){
    Node * n = getNode(layer, row, col);
    printf("Node Observation: layer %i, row %i, col: %i\n", layer, row, col);
    printf("input: ");
    int i = 0;
    for(int c = 0 ; c < n->ni ; c++ ){
        if(c % (n->ni / 4) == 0){
            printf("\n");
        }
        printf("%f ", n->observation[i]);
        i++;
    }
    printf("\nPrevious Belief: ");
    for(int c  = 0 ; c < n->nb ; c++ ){
        printf("%f ", n->observation[i]);
        i++;
    }
    printf("\nParent prev belief: ");
    for(int c  = 0 ; c < n->np ; c++ ){
        printf("%f ", n->observation[i]);
        i++;
    }
    printf("\n");
}

void DestinNetworkAlt::printNodeBeliefs(int layer, int row, int col){
    Node * n = getNode(layer, row, col);
    printf("Node beliefs: layer%i, row %i, col: %i\n", layer, row, col);
    for(int i = 0; i < n->nb ; i++){
        printf("%f ", n->pBelief[i]);
    }
    printf("\n");

}

void DestinNetworkAlt::setParentBeliefDamping(float gamma){
    if(gamma < 0 || gamma > 1.0){
        throw std::domain_error("setParentBeliefDamping: gamma must be between 0 and 1");
    }
    for(int n = 0; n < destin->nNodes ; n++){
        destin->nodes[n].gamma = gamma;
    }
}

void DestinNetworkAlt::setPreviousBeliefDamping(float lambda){
    if(gamma < 0 || gamma > 1.0){
        throw std::domain_error("setParentBeliefDamping: lambda must be between 0 and 1");
    }
    for(int n = 0; n < destin->nNodes ; n++){
        destin->nodes[n].nLambda = lambda;
    }

}


void DestinNetworkAlt::load(const char * fileName){
    if(centroidImages != NULL){
        Cig_DestroyCentroidImages(destin,  centroidImages);
        centroidImages = NULL;
    }

    destin = LoadDestin(destin, fileName);
    if(destin == NULL){
        throw std::runtime_error("load: could not open destin file.\n");
    }


    Node * n = GetNodeFromDestin(destin, 0, 0, 0);
    this->beta = n->beta;
    this->gamma = n->gamma;
    this->isUniform = destin->isUniform;
    this->lambda = n->nLambda;

    if(temperatures != NULL){
        delete [] temperatures;
    }
    temperatures = new float[getLayerCount()];
    memcpy(temperatures, destin->temp, sizeof(float) * getLayerCount());

}

void DestinNetworkAlt::moveCentroidToInput(int layer, int row, int col, int centroid){
    Node * n = getNode(layer, row, col);
    if(centroid >= destin->nb[layer]){
        throw std::domain_error("moveCentroidToInput: centroid out of bounds.");
    }
    float * cent = &n->mu[centroid * n->ns];
    memcpy(cent, n->observation, n->ns * sizeof(float));
}

float * DestinNetworkAlt::getCentroidImage(int layer, int centroid){
    if(!destin->isUniform){
        printf("getCentroidImage: must be uniform");
        return NULL;
    }

    displayCentroidImage(layer, centroid);
    return getCentroidImages()[layer][centroid];
}


cv::Mat DestinNetworkAlt::getCentroidImageM(int layer, int centroid, int disp_width, bool enhanceContrast){
    if(!isUniform){
        throw std::logic_error("can't displayCentroidImage with non uniform DeSTIN.");
    }
    if(layer > destin->nLayers ||  centroid > destin->nb[layer]){
        throw std::domain_error("displayCentroidImage: layer, centroid out of bounds\n");
    }

    uint width = Cig_GetCentroidImageWidth(destin, layer);

    //initialize or create new grid if needed
    if(centroidImage.rows != width || centroidImage.cols != width){
        centroidImage = cv::Mat(width, width, CV_32FC1);
    }
    float * data = (float *)centroidImage.data;

    memcpy(data, getCentroidImages()[layer][centroid], width*width*sizeof(float));

    cv::Mat toShow;
    centroidImage.convertTo(toShow, CV_8UC1, 255); // make it suitable for equalizeHist
    if(enhanceContrast){
        cv::equalizeHist(toShow, toShow);
    }

    cv::resize(toShow,centroidImageResized, cv::Size(disp_width, disp_width), 0, 0, cv::INTER_NEAREST);

    return centroidImageResized;
}

void DestinNetworkAlt::displayCentroidImage(int layer, int centroid, int disp_width, bool enhanceContrast, string window_name){
    getCentroidImageM(layer, centroid, disp_width, enhanceContrast);
    cv::imshow(window_name.c_str(), centroidImageResized);
}

void DestinNetworkAlt::saveCentroidImage(int layer, int centroid, string filename, int disp_width, bool enhanceContrast){
    getCentroidImageM(layer, centroid, disp_width, enhanceContrast);
    cv::imwrite(filename, centroidImageResized);
}

void DestinNetworkAlt::displayLayerCentroidImages(int layer,
                                int scale_width,
                                int border_width,
                                string window_title
                                ){

    if(layer < 0 || layer >= getLayerCount()){

        std::cerr << "displayLayerCentroidImages: layer out of bounds " << std::endl;
        return;
    }
    cv::imshow(window_title, getLayerCentroidImages(layer, scale_width, border_width));
    return;
}

cv::Mat DestinNetworkAlt::getLayerCentroidImages(int layer,
                              int scale_width,
                              int border_width){
    if(!isUniform){
        throw std::logic_error("can't displayLayerCentroidImages with non uniform DeSTIN.");
    }

    int images = getBeliefsPerNode(layer);
    int images_wide = ceil(sqrt(images));
    int sub_img_width = (int)((double)scale_width / (double)images_wide - (double)border_width);

    int wpb = sub_img_width + border_width; // sub image width plus boarder. Each image gets a right and bottom boarder only.

    int images_high = ceil((float)images / (float)images_wide);

    // initialize the big image as solid black
    cv::Mat big_img = cv::Mat::zeros(wpb*images_high, wpb*images_wide, CV_32FC1);

    int r, c, x, y;
    // copies the subimages into the correct place in the big image
    for(int i = 0 ; i < images ; i++){
            r = i  / images_wide;
            c = i - r * images_wide;
            x = c * wpb;
            y = r * wpb;
            int w = Cig_GetCentroidImageWidth(destin, layer);
            cv::Mat subimage(w, w, CV_32FC1, getCentroidImages()[layer][i]);
            cv::Mat subimage_resized;
            cv::resize(subimage, subimage_resized, cv::Size(sub_img_width, sub_img_width), 0,0,cv::INTER_NEAREST);
            cv::Rect roi( cv::Point( x, y ), subimage_resized.size() );
            cv::Mat dest = big_img( roi );

            subimage_resized.copyTo( dest );
    }

    //cv::Mat toShow;
    big_img.convertTo(layerCentroidsImage, CV_8UC1, 255);

    //layerCentroidsImage = big_img;
    return layerCentroidsImage;
}

void DestinNetworkAlt::saveLayerCentroidImages(int layer, const string & filename,
                              int scale_width,
                              int border_width
                              ){
    cv::imwrite(filename, getLayerCentroidImages(layer, scale_width, border_width) );
    return;
}

/*****************************************************************************/
