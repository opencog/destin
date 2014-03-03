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

float **** DestinNetworkAlt::getCentroidImages(){
    if(centroidImages==NULL){
        centroidImages = Cig_CreateCentroidImages(destin, centroidImageWeightParameter);
    }
    return centroidImages;
}

DestinNetworkAlt::DestinNetworkAlt(SupportedImageWidths width, unsigned int layers,
                                   unsigned int centroid_counts [], bool isUniform,
                                   unsigned int layer_widths[],
                                   DstImageMode imageMode):
        training(true),
        lambdaCoeff(.1),
        gamma(.1),
        isUniform(isUniform),
        centroidImages(NULL),
        centroidImageWeightParameter(1.0),
        inputImageWidth(width),
        imageMode(imageMode)
        {
    int extRatio = getExtendRatio(imageMode);
    init(width, layers, centroid_counts, isUniform, extRatio, layer_widths);
}

void DestinNetworkAlt::init(SupportedImageWidths width, unsigned int layers,
                            unsigned int centroid_counts [], bool isUniform,
                            int extRatio, unsigned int layer_widths[]){
    callback = NULL;
    initTemperatures(layers, centroid_counts);

    DestinConfig *dc = CreateDefaultConfig(layers);
    dc->addCoeff = 0; // disabled addition of centroids
    dc->beta = 0.01;
    std::copy(centroid_counts, centroid_counts + layers, dc->centroids); // initial number of centroids
    dc->extRatio = extRatio;
    dc->freqCoeff = 0.05;
    dc->freqTreshold = 0; // disabled deletion of centroids
    dc->gamma = gamma;
    dc->isUniform = isUniform;
    dc->lambdaCoeff = lambdaCoeff;
    std::copy(centroid_counts, centroid_counts + layers, dc->layerMaxNb); // max number of centroids

    if(layer_widths != NULL){ // provide support for non standard heirarchy
        std::copy(layer_widths, layer_widths + layers, dc->layerWidths);
    }

    if(width % dc->layerWidths[0] == 0){
        int ratio = width / dc->layerWidths[0];
        dc->inputDim = ratio * ratio;
    } else {
        DestroyConfig(dc);
        throw std::runtime_error("Input image width must be evenly divisible by the bottom layer width.");
    }

    dc->nClasses = 0;
    dc->nMovements = 0; //this class does not use movements
    dc->starvCoeff = 0.05;
    std::copy(temperatures, temperatures + layers, dc->temperatures);

    destin = InitDestinWithConfig(dc);
    if(destin == NULL){
        throw std::runtime_error("Could not create the destin structure. Perhaps the given layer widths is not supported.");
    }

    DestroyConfig(dc);

    setBeliefTransform(DST_BT_NONE);
    ClearBeliefs(destin);
    SetLearningStrat(destin, CLS_FIXED);
    destin->fixedLearnRate = 0.1;
    isTraining(true);
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
                    fSum += fabs(currNode->mu[i][k]
                                 - currNode->mu[j][k]);
                }
                for(k=currNode->ni + currNode->nb + currNode->np + currNode->nc;
                    k < currNode->ns; ++k)
                {
                    fSum += fabs(currNode->mu[i][k]
                                 - currNode->mu[j][k]);
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

std::vector<float> DestinNetworkAlt::getLayersSeparations()
{
    std::vector<float> separations(destin->nLayers);
    for (int i = 0; i < destin->nLayers; i++)
    {
        separations[i] = getSep(i);
    }
    return separations;
}

// 2013.7.4
// CZT: get var;
float DestinNetworkAlt::getVar(int layer)
{
    Node * currNode = getNode(layer, 0, 0);
    int j;
    float fSum = 0.0;
    for(j=0; j<currNode->ni; ++j)
    {
        fSum += destin->uf_absvar[layer][j];
    }
    for(j=currNode->ni+currNode->nb+currNode->np+currNode->nc;
        j<currNode->ns; ++j)
    {
        fSum += destin->uf_absvar[layer][j];
    }
    return fSum / (currNode->ni * (layer == 0 ? destin->extRatio : 1));
}

std::vector<float> DestinNetworkAlt::getLayersVariances()
{
    std::vector<float> variances(destin->nLayers);
    for (int i = 0; i < destin->nLayers; i++)
    {
        variances[i] = getVar(i);
    }
    return variances;
}

float DestinNetworkAlt::getQuality(int layer)
{
    return getSep(layer)-getVar(layer);
}

std::vector<float> DestinNetworkAlt::getLayersQualities()
{
    std::vector<float> qualities(destin->nLayers);
    for (int i = 0; i < destin->nLayers; i++)
    {
        qualities[i] = getQuality(i);
    }
    return qualities;
}

void DestinNetworkAlt::printFloatCentroids(int layer)
{
    Node * currNode = getNode(layer, 0, 0);

    // Display all centroids on the selected layer
    printf("---All centroids on layer %d:\n", layer);
    for(int i=0; i<currNode->nb; ++i)
    {
        for(int j=0; j<currNode->ni; ++j)
        {
            printf("%f  ", currNode->mu[i][j]);
        }
        printf("\n");
        for(int j=0; j<currNode->nb; ++j)
        {
            printf("%f  ", currNode->mu[i][j + currNode->ni]);
        }
        printf("\n");
        for(int j=0; j<currNode->np; ++j)
        {
            printf("%f  ", currNode->mu[i][j + currNode->ni + currNode->nb]);
        }
        printf("\n");
        if(i != currNode->nb-1)
        {
            printf("---\n");
        }
    }
    printf("\n\n");
}

void DestinNetworkAlt::printFloatVector(std::string title, std::vector<float> vec)
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
    for(int i=0; i<currNode->ns; ++i)
    {
        outCen.push_back(currNode->mu[idx][i]);
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
        printFloatVector("---Result is:\n", selCen);
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
        printFloatCentroids(srcLayer);
        // Display the selected centroid
        printFloatVector("---The selected centroid is:\n", selCen);

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
                    delta = extendCen[i][k] - currNode->mu[j][k];

                    delta *= delta;

                    // Assume starv is 1
                    delta *= 1;

                    sumMal += delta/destin->uf_sigma[srcLayer][j][k];
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
        printFloatVector("---Result is:\n", selCen);
        return;
    }
    else
    {
        Node * childNode = getNode(srcLayer-1, 0, 0);

        std::vector<float> newSelCen;
        std::vector<float> normSelCen;

        // Display all centroids on source layer
        printFloatCentroids(srcLayer);
        // Display the selected centroid
        printFloatVector("---The selected centroid is:\n", selCen);

        if(srcLayer == 1)
        {
            // Use every quarter as a single lower level centroid, then downsample
            //int level0 = 0;
            //Node * childNode = getNode(level0, 0, 0);

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
                        tempCen[k-j*childNode->ns] += normSelCen[i*childNode->nb+j] * childNode->mu[j][k-j*childNode->ns];
                    }
                }
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
                        tempCen[k - j*childNode->ns] += normSelCen[i*childNode->nb + j] * childNode->mu[j][k-j*childNode->ns];
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

void DestinNetworkAlt::setFrequencyCoefficients(float freqCoeff, float freqTreshold, float addCoeff)
{
    destin->freqCoeff = freqCoeff;
    destin->freqTreshold = freqTreshold;
    destin->addCoeff = addCoeff;
}

void DestinNetworkAlt::setStarvationCoefficient(float starvCoeff)
{
    for (int n = 0; n < destin->nNodes; n++)
    {
        destin->nodes[n].starvCoeff = starvCoeff;
    }
}

void DestinNetworkAlt::setMaximumCentroidCounts(int count)
{
    for (int l = 0; l < destin->nLayers; l++)
    {
        destin->layerMaxNb[l] = count;
    }
}

void DestinNetworkAlt::doDestin(float * input_array ) {

    if (destin->isUniform){
        Uniform_DeleteCentroids(destin);
        Uniform_AddNewCentroids(destin);
    }

    FormulateBelief(destin, input_array);

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
            printf("%.5f ", n->mu[centroid][dimension]);
        }
        printf("\n");
        printf(" self ");
        for(int end = dimension + n->nb ; dimension < end ; dimension++){
            printf("%.5f ", n->mu[centroid][dimension]);
        }
        printf(" parent ");
        for(int end = dimension + n->np ; dimension < end ; dimension++){
            printf("%.5f ", n->mu[centroid][dimension]);
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
        printf("%f ", n->belief[i]);
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

void DestinNetworkAlt::setPreviousBeliefDamping(float lambdaCoeff){
    if(lambdaCoeff < 0 || lambdaCoeff > 1.0){
        throw std::domain_error("setParentBeliefDamping: lambdaCoeff must be between 0 and 1");
    }
    for(int n = 0; n < destin->nNodes ; n++){
        destin->nodes[n].lambdaCoeff = lambdaCoeff;
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
    this->gamma = n->gamma;
    this->isUniform = destin->isUniform;
    this->lambdaCoeff = n->lambdaCoeff;

    if(temperatures != NULL){
        delete [] temperatures;
    }
    temperatures = new float[getLayerCount()];
    memcpy(temperatures, destin->temp, sizeof(float) * getLayerCount());
    imageMode = extRatioToImageMode(destin->extRatio);
}

void DestinNetworkAlt::moveCentroidToInput(int layer, int row, int col, int centroid){
    Node * n = getNode(layer, row, col);
    if(centroid >= destin->nb[layer]){
        throw std::domain_error("moveCentroidToInput: centroid out of bounds.");
    }
    float * cent = n->mu[centroid];
    memcpy(cent, n->observation, n->ns * sizeof(float));
}

float * DestinNetworkAlt::getCentroidImage(int channel, int layer, int centroid){
    if(!destin->isUniform){
        printf("getCentroidImage: must be uniform");
        return NULL;
    }

    displayCentroidImage(layer, centroid);
    return getCentroidImages()[channel][layer][centroid];
}

int DestinNetworkAlt::getCvFloatImageType(){
    if(imageMode == DST_IMG_MODE_GRAYSCALE || imageMode == DST_IMG_MODE_GRAYSCALE_STEREO){
        return CV_32FC1;
    } else if (imageMode == DST_IMG_MODE_RGB){
        return CV_32FC3;
    } else {
        throw std::runtime_error("getCvFloatImageType: unsupported image mode.\n");
    }
}

int DestinNetworkAlt::getCvByteImageType(){
    if(imageMode == DST_IMG_MODE_GRAYSCALE || imageMode == DST_IMG_MODE_GRAYSCALE_STEREO){
        return CV_8UC1;
    } else if (imageMode == DST_IMG_MODE_RGB){
        return CV_8UC3;
    } else {
        throw std::runtime_error("getCvByteImageType: unsupported image mode.\n");
    }
}

DstImageMode DestinNetworkAlt::extRatioToImageMode(int extRatio){
    switch(extRatio){
        case 1:
            return DST_IMG_MODE_GRAYSCALE;
        case 2:
            return DST_IMG_MODE_GRAYSCALE_STEREO;
        case 3:
            return DST_IMG_MODE_RGB;
        default:
            throw std::runtime_error("extRatioToImageMode: unsupported extRatio\n.");
    }
}

int DestinNetworkAlt::getExtendRatio(DstImageMode imageMode){
    if(imageMode == DST_IMG_MODE_GRAYSCALE){
        return 1;
    } else if (imageMode == DST_IMG_MODE_GRAYSCALE_STEREO){
        return 2;
    } else if (imageMode == DST_IMG_MODE_RGB){
        return 3;
    } else {
        throw std::runtime_error("DstImageMode: unsupported image mode.\n");
    }
}

cv::Mat DestinNetworkAlt::convertCentroidImageToMatImage(int layer, int centroid, bool toByteType){
    uint width = Cig_GetCentroidImageWidth(destin, layer);

    int height = width;

    if(imageMode == DST_IMG_MODE_GRAYSCALE_STEREO){
        width = width * 2;
    }

    cv::Mat centroidImage(height, width, getCvFloatImageType());

    cv::Mat toShow;
    if(imageMode == DST_IMG_MODE_GRAYSCALE){
        float * data = (float *)centroidImage.data;
        // Copy generated centroid image to the Opencv mat.
        memcpy(data, getCentroidImages()[0][layer][centroid], width*width*sizeof(float));
    } else if(imageMode == DST_IMG_MODE_RGB){
        // Copy  generated centroid to the color OpenCV mat.
        for(int channel = 0 ; channel < 3 ; channel++){ // iterate over R, G, B
            float * centroid_image = getCentroidImages()[channel][layer][centroid];
            cv::Point p;
            int pixel = 0;
            for(p.y = 0 ; p.y < centroidImage.rows ; p.y++){
                for(p.x = 0 ; p.x < centroidImage.cols; p.x++){
                    centroidImage.at<cv::Vec3f>(p)[channel] = centroid_image[pixel];
                    pixel++;
                }
            }
        }
    } else if(imageMode == DST_IMG_MODE_GRAYSCALE_STEREO){

        // This section copies left and right pair of stereo images into a rectangle shapped image.

        cv::Rect left_region_rect(cv::Point(0, 0), cv::Size(width, width));
        cv::Rect right_region_rect(cv::Point(width, 0), cv::Size(width, width));

        cv::Mat left_region = centroidImage(left_region_rect);
        cv::Mat right_region = centroidImage(right_region_rect);

        cv::Mat leftImage (width, width, getCvFloatImageType());
        memcpy(leftImage.data,  getCentroidImages()[0][layer][centroid], width*width*sizeof(float));

        cv::Mat rightImage(width, width, getCvFloatImageType());
        memcpy(rightImage.data, getCentroidImages()[1][layer][centroid], width*width*sizeof(float));

        leftImage.copyTo(left_region);   // copy left centroid image to left side of the bigger image
        rightImage.copyTo(right_region); // copy right centroid image to right side of the bigger image

        //TODO: test this section and remove the throw
        throw std::runtime_error("convertCentroidImageToMatImage: TODO: need to test the code for DST_IMG_MODE_GRAYSCALE_STEREO");
    } else {
        throw std::runtime_error("fillMatWithCentroidImage: unsupported image mode.\n");
    }
    // make it suitable for equalizeHist
    if(toByteType){
        centroidImage.convertTo(toShow, getCvByteImageType(), 255);
    } else {
        toShow  = centroidImage;
    }

    return toShow;
}

cv::Mat DestinNetworkAlt::getCentroidImageM(int layer, int centroid, int disp_width, bool enhanceContrast){
    if(!isUniform){
        throw std::logic_error("can't displayCentroidImage with non uniform DeSTIN.");
    }

    if(layer > destin->nLayers ||  centroid > destin->nb[layer]){
        throw std::domain_error("displayCentroidImage: layer, centroid out of bounds\n");
    }

    cv::Mat toShow = convertCentroidImageToMatImage(layer, centroid, true);

    if(enhanceContrast){
        if(getCvByteImageType() == CV_8UC1){
            cv::equalizeHist(toShow, toShow);
        } else if(getCvByteImageType() == CV_8UC3){
            // This section is borrowed from http://stackoverflow.com/a/14709331
            // on how to do equalization on color images
            std::vector<cv::Mat> hsv_planes;
            cvtColor(toShow,toShow,CV_BGR2HSV);
            cv::split(toShow, hsv_planes);
            cv::equalizeHist(hsv_planes[2], hsv_planes[2]);
            cv::merge(hsv_planes,toShow);
            cvtColor(toShow, toShow, CV_HSV2BGR);
        } else {
            std::cerr << __PRETTY_FUNCTION__ << ": unsupported image type for cv::equalizeHist" << std::endl;
        }
    }

    cv::resize(toShow, centroidImageResized, cv::Size(disp_width, disp_width), 0, 0, cv::INTER_NEAREST);

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
                                string window_title,
                                std::vector<int> sort_order
                                ){

    if(layer < 0 || layer >= getLayerCount()){

        std::cerr << "displayLayerCentroidImages: layer out of bounds " << std::endl;
        return;
    }
    cv::imshow(window_title, getLayerCentroidImages(layer, scale_width, border_width, sort_order));
    return;
}

float DestinNetworkAlt::distanceBetweenCentroids(int layer, int centroid1, int centroid2){

    float distance = 0;

    int width = Cig_GetCentroidImageWidth(destin, layer);
    int size = width * width;
    for(int channel = 0 ; channel < destin->extRatio ; channel++ ){
        float * first = getCentroidImages()[channel][layer][centroid1];
        float * second = getCentroidImages()[channel][layer][centroid2];
        for(int pix = 0 ; pix < size ; pix++){
            distance += fabs(first[pix] - second[pix]);
        }
    }
    return distance;
}

std::vector<int> DestinNetworkAlt::sortLayerCentroids(int layer){
    int centroids = getBeliefsPerNode(layer);
    std::vector<int> order;
    std::vector<int> candidates;
    for(int i = 1 ; i < centroids ; i++){
        candidates.push_back(i);
    }

    order.push_back(0); // centroid 0 is always first
    for(int first = 0 ; first < centroids - 1; first++){
        int first_centroid = order.at(order.size() - 1);
        if(first == 125){
            first_centroid = first_centroid;
        }
        int min_cand_index = 0;
        int min_centroid = candidates.at(min_cand_index);
        float min_dist = distanceBetweenCentroids(layer, first_centroid, min_centroid);
        for(int cand_index = 1 ; cand_index < candidates.size() ; cand_index++){
            float  dist = distanceBetweenCentroids(layer, first_centroid, candidates.at(cand_index));
            if(dist < min_dist){
                min_cand_index = cand_index;
                min_centroid = candidates.at(cand_index);
                min_dist = dist;
            }
        }

        order.push_back(min_centroid);
        if(min_cand_index >= candidates.size() || min_cand_index < 0){
            std::logic_error("DestinNetworkAlt::sortLayerCentroids min index out of bounds");
        }
        candidates.erase(candidates.begin() + min_cand_index);
    }
    if(order.size() != centroids){
        throw std::logic_error("DestinNetworkAlt::sortLayerCentroids: order did not match number of centroids\n");
    }
    return order;
}

cv::Mat DestinNetworkAlt::getLayerCentroidImages(int layer,
                              int scale_width,
                              int border_width, std::vector<int> sort_order){
    if(!isUniform){
        throw std::logic_error("can't displayLayerCentroidImages with non uniform DeSTIN.");
    }

    int centroids = getBeliefsPerNode(layer);
    int images_wide = ceil(sqrt(centroids));
    int sub_img_width = (int)((double)scale_width / (double)images_wide - (double)border_width);

    // sub image width plus boarder. Each image gets a right and bottom boarder only.
    int wpb = sub_img_width + border_width;

    int images_high = ceil((float)centroids / (float)images_wide);

    // initialize the big image as solid black
    cv::Mat big_img = cv::Mat::zeros(wpb*images_high, wpb*images_wide, getCvFloatImageType());

    // specify centroid sort order
    std::vector<int> centroid_sort_order;
    if(sort_order.size() > 0){
        centroid_sort_order = sort_order;
    } else { // otherwize, default to centroid number
        for(int i = 0 ; i < centroids ; i ++){
            centroid_sort_order.push_back(i);
        }
    }

    int r, c, x, y;
    // copies the subimages into the correct place in the big image
    for(int centroid = 0 ; centroid < centroids ; centroid++){
            r = centroid  / images_wide;
            c = centroid - r * images_wide;
            x = c * wpb;
            y = r * wpb;

            int centroid_sorted = centroid_sort_order.at(centroid);
            cv::Mat subimage = convertCentroidImageToMatImage(layer, centroid_sorted, false);
            cv::Mat subimage_resized;
            cv::resize(subimage, subimage_resized, cv::Size(sub_img_width, sub_img_width), 0,0,cv::INTER_NEAREST);
            cv::Rect roi( cv::Point( x, y ), subimage_resized.size() );
            cv::Mat dest = big_img( roi );

            // copy centroid image into big image
            subimage_resized.copyTo( dest );
    }

    //cv::Mat toShow;
    big_img.convertTo(layerCentroidsImage, getCvByteImageType(), 255);

    //layerCentroidsImage = big_img;
    return layerCentroidsImage;
}

void DestinNetworkAlt::saveLayerCentroidImages(int layer, const string & filename,
                                               int scale_width,
                                               int border_width,
                                               std::vector<int> sort_order){
    cv::imwrite(filename, getLayerCentroidImages(layer, scale_width, border_width, sort_order) );
    return;
}

/*****************************************************************************/
