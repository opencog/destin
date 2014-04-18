#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

#include<quadmath.h>

#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/types_c.h"

#include "macros.h"
#include "node.h"
#include "destin.h"


#define NITS 20000
#define NMEANS 1

#define ASTK 15
#define NCOLS 8
void PrintIt( Destin *d )
{
    uint i, j, n;
    uint l, nIdx, k;
    Node *nPtr;

    /*
    printf("\033[2J\033[1;1H");

    for( nIdx=0, l=0; l < d->nLayers; nIdx += d->layerSize[l++] )
    {
        uint nb = d->nodes[nIdx].nb;

        for( n=nIdx; n < nIdx + d->layerSize[l]; n += NCOLS )
        {
            for( i=0; i < nb; i++ )
            {
                for( k=0; k < NCOLS && n-nIdx+k < d->layerSize[l]; k++ )
                {
                    nPtr = &d->nodes[n+k];
                    uint nAstk = (uint) (nPtr->belief[i] * ASTK);

                    for( j=0; j < nAstk; j++ )
                    {
                        printf("*");
                    }
                    for( j=nAstk; j < ASTK; j++ )
                    {
                        printf("-");
                    }

                    if( i == nPtr->winner && i == nPtr->genWinner )
                    {
                        printf(" W\t");
                    } else if (i == nPtr->winner) {
                        printf(" w\t");
                    } else if (i == nPtr->genWinner ) {
                        printf(" g\t");
                    } else {
                        printf("  \t");
                    }
                }
                printf("\n");
            }
            printf("\n");
        }

        printf("\n");
    }
    */

    for( n=0; n < d->nNodes; n++ )
    {
        printf("node %d\n", n);
        nPtr = &d->nodes[n];
        for( i=0; i < nPtr->nb; i++ )
        {
            printf("%0.3f ", nPtr->belief[i]);
        }
        printf("\n");
    }
}

void TrainDestin( Destin *d, char *dataFileName, char *labelsFileName )
{
    size_t freadResult; // Not used, added to remove compiler warnings.
    // check if destin passed is initialized
    if( d == NULL )
    {
        fprintf(stderr, "Destin 0x%p not initialized!\n", d);
        exit(1);
    }
    
    FILE *dataFile;
    FILE *labelFile;
    
    dataFile = fopen(dataFileName, "rb");
    if( !dataFile ) {
        fprintf(stderr, "Cannot open data file %s\n", dataFileName);
        exit(1);
    }

    labelFile = fopen(labelsFileName, "rb");
    if( !labelFile ) {
        fprintf(stderr, "Cannot open data file %s\n", labelsFileName);
        exit(1);
    }

    size_t inputFrameSize = d->layerSize[0] * 16;

    // allocate space for data set
    MALLOC(d->dataSet, float, inputFrameSize);

    uint i;
    uint j;
    uint lTrain;
    uint label;

    uint nBatches = 6;
    uint batch;

    float muSumSqDiff = 0;

    SaveDestin( d, "destintmp.des");

    lTrain = -1;
    for( batch=0; batch < nBatches; batch++ )
    {
        i = 0;

        // while the whole file hasn't been read...
        while( i < NITS * 3 )//!feof(dataFile) )
        {
            freadResult = fread( d->dataSet, sizeof(float), inputFrameSize, dataFile );
            freadResult =fread( &label, sizeof(uint), 1, labelFile );

            if( label > d->nc )
            {
                fprintf(stderr, "bad input label -- %d > %d!\n", label, d->nc);
            }

            for( j=0; j < d->nc; j++ )
            {
                d->inputLabel[j] = 0;
            }

            d->inputLabel[label] = 1;

            // a new digit is picked up every d->nMovements movements.
            if( i % d->nMovements == 0 )
            {
                ClearBeliefs( d );
            }

            if( i % NITS == 0 && batch == 0 )
                lTrain++;

            for( j=0; j < d->nLayers; j++ )
            {
                if( j <= lTrain )
                    d->layerMask[j] = 1 && ((i % d->nLayers) >= j);
            }

            FormulateBelief( d, d->dataSet );
            muSumSqDiff += d->muSumSqDiff;

            if( i % 1000 == 0 )
            {
                printf("iteration %d: %0.5f\n", i, muSumSqDiff);
                SaveDestin( d, "destintmp.des");
                DisplayFeatures( d );
                cvWaitKey(25);

                uint nIdx;
                for( nIdx=0; nIdx < d->nodes[d->nNodes-1].nb; nIdx++ )
                {
                    printf("%0.2f ", d->nodes[d->nNodes-1].belief[nIdx]);
                }
                printf("\n");

                muSumSqDiff = 0;
            }

            i++;
        }

        SaveDestin( d, "destintmp.des");

        for( j=0; j < d->nc; j++ )
        {
            d->inputLabel[j] = 0;
        }


        rewind(dataFile);
        rewind(labelFile);
    }

    return;
}

void TestDestin( Destin *d, char *dataFileName, char *labelsFileName, bool generative )
{
    FILE *beliefsFile;
    if (!generative)
    {
        beliefsFile = fopen("beliefs.dat", "wb");
    }
    size_t freadResult; // Not used, added to remove compiler warnings.
    
    FILE *dataFile;
    FILE *labelFile;
    
    dataFile = fopen(dataFileName, "rb");
    if( !dataFile ) {
        fprintf(stderr, "Cannot open data file %s\n", dataFileName);
        exit(1);
    }

    labelFile = fopen(labelsFileName, "rb");
    if( !labelFile ) {
        fprintf(stderr, "Cannot open data file %s\n", labelsFileName);
        exit(1);
    }

    size_t inputFrameSize = d->layerSize[0] * 16;

    // allocate space for data set
    MALLOC(d->dataSet, float, inputFrameSize);

    uint j;
    uint lTrain;
    uint label;

    uint nBatches = 5;
    uint batch;

    float muSumSqDiff = 0;
    uint n;
    uint i;

    // set up stuff for generative display
    float *outFrame;
    MALLOC( outFrame, float, d->layerSize[0] * d->nci[0]);

    CvSize size;
    size.height = 16;
    size.width = 16;
    IplImage* inFrame_ipl  = cvCreateImageHeader(size, IPL_DEPTH_32F, 1);
    IplImage* outFrame_ipl = cvCreateImageHeader(size, IPL_DEPTH_32F, 1);

    outFrame_ipl->imageData = (char *) outFrame;
    outFrame_ipl->imageDataOrigin = outFrame_ipl->imageData;

    size.height = 128;
    size.width = 128;

    IplImage *destinIn  = cvCreateImage( size, IPL_DEPTH_32F, 1 );
    IplImage *destinOut = cvCreateImage( size, IPL_DEPTH_32F, 1 );

    i=0;

    ResetStarvTrace( d );

    while( i < d->nMovements * 15000 )
    {
        freadResult = fread( d->dataSet, sizeof(float), inputFrameSize, dataFile );
        freadResult = fread( &label, sizeof(uint), 1, labelFile );

        // a new digit is picked up every d->nMovements movements.
        if( i % d->nMovements == 0 )
        {
            ClearBeliefs( d );
        }
        
        FormulateBelief( d, d->dataSet );
    
        printf("iteration %d\n", i);

        if( generative )
        {
            uint m;
            if( i % d->nMovements == d->nMovements - 1 )
            {
                uint n;

                for( m=0; m < 10; m++ )
                {
                    GenerateInputFromBelief( d, outFrame );

                    inFrame_ipl->imageData = (char *) d->dataSet;
                    inFrame_ipl->imageDataOrigin = inFrame_ipl->imageData;

                    cvResize(inFrame_ipl, destinIn, CV_INTER_NN);
                    cvResize(outFrame_ipl, destinOut, CV_INTER_NN);

                    cvShowImage("Destin In", destinIn);
                    cvShowImage("Destin Out", destinOut);

                    cvWaitKey(10);
                }
            }
        } else {
            if( i % d->nMovements == d->nMovements - 1 )
            {
                uint j;
                fprintf(beliefsFile, "%d, ", label);
                for( j=0; j < d->nodes[d->nNodes-1].nb; j++ )
                {
                    fprintf(beliefsFile, "%0.5f, ", d->nodes[d->nNodes-1].belief[j]);
                }
                fprintf(beliefsFile, "\n");
            }
        }

        i++;
    }

    fclose( dataFile );
    fclose( labelFile );
    if (! generative)
    {
        fclose( beliefsFile );
    }

    cvReleaseImage(&destinIn);
    cvReleaseImage(&destinOut);
    cvReleaseImageHeader(&inFrame_ipl);
    cvReleaseImageHeader(&outFrame_ipl);

    FREE( d->dataSet );
    FREE( outFrame );
}

// run an iteration with the CPU implementations of the kernels
void FormulateBelief( Destin *d, float *image )
{
    // n = node index
    // l = layer index
    uint n, l;

    // Set stability to 0
    d->muSumSqDiff = 0;


    if(d->isUniform){
        Uniform_ResetStats(d);
    }
    // n_start = first node index in a layer
    // n_end = first node index of 'next' layer
    uint n_start, n_end;
    n_start = 0;
    n_end = 0;
    
    for(l = 0; l < d->nLayers; l++ )
    {
        // Set start of node indexes for current layer to n_end (first node index of 'next' layer)
        n_start = n_end;
        // Set end of node index to start index + size of current layer
        n_end = n_start + d->layerSize[l];
        
        // Loop through nodes in current layer]
        #pragma omp parallel for private(n) //openmp to do multithreaded processing
        for(n = n_start; n < n_end; n++ )
        {
            // Get an observation from the source image for the current node if in layer 0, else from the child nodes.
            GetObservation( d->nodes, image, n );

            // Calculate the distances between the centroids of the current node and the new observation
            CalculateDistances( d->nodes, n );

            // Normalize the calculations and get the winning centroid
            NormalizeBeliefGetWinner( d->nodes, n );

            // Check if centroids should be updated (1 = yes)
            if( d->layerMask[l] == 1 )
            {
                // Calculate the required centroid movement
                //TODO: merge CalculateDistances with CalcCentroidMovement
                CalcCentroidMovement( d->nodes, d->inputLabel, n );

                // Check if the network is uniform
                if(!d->isUniform){
                    // Apply the deltas to move the centroids
                    MoveCentroids( d->nodes,n );

                    // Update the starvation for the centroids
                    UpdateStarvation(d->nodes, n);

                    // Update the network's stability by adding the node's stability
                    d->muSumSqDiff += d->nodes[n].muSqDiff;
                }
            }
        }

        // Check if the network is uniform and if centroids should be updated
        if(d->isUniform && d->layerMask[l] == 1  ){

            // Loop through the nodes in the current layer
            for(n = n_start ; n < n_end ; n++){
                // Average shared centroid's movements
                Uniform_AverageDeltas(d->nodes, n);
            }
            // Move the shared centroids
            Uniform_ApplyDeltas(d, l, d->uf_sigma[l] );
            // In uniform destin, muSqDiff for the layer is stored in the 0th node of the layer.
            d->muSumSqDiff += GetNodeFromDestin(d, l, 0, 0)->muSqDiff;

            // Update shared centroids starvation
            Uniform_UpdateStarvation(d, l, d->uf_starv[l], d->uf_winCounts[l], d->nodes[0].starvCoeff );

            // Update shared centroids estimated frequencies
            Uniform_UpdateFrequency(d, l, d->uf_winFreqs[l], d->uf_winCounts[l], d->freqCoeff);
        }
    }

    // copy node's belief to parent node's input
    CopyOutputBeliefs( d );
}

uint GenFromPMF( float *pmf, uint len )
{
    uint i;

    float n = 0;                    // normalizing const
    float c = 0;                    // cumulative sum
    float r = (float) rand() / RAND_MAX;    // random num chosen

    // get norm const
    for( i=0; i < len; i++ )
    {
        n += pmf[i];
    }

    // sample from pmf
    for( i=0; i < len; i++ )
    {
        c += pmf[i] / n;
        if( r < c )
            break;
    }

    return i;
}

#define __special double

// box-muller normal rng
__special NormRND(__special m, __special s)
{
    __special u1,u2;

    u1 = (__special) rand() / RAND_MAX;
    u2 = (__special) rand() / RAND_MAX;

    __special val = sqrt(-2*log(u1)) * cos(2*M_PI*u2) * s + m;

    return val;
}

#define KAHAN(sum, input, y, t, c)  \
    y = input - c;                  \
    t = sum + y;                    \
    c = (t - sum) - y;              \
    sum = t;

void Boltzmannize( Node *n, __special *b )
{
    uint i;

    __special maxBoltz = 0;
    for( i=0; i < n->nb; i++ )
    {
        if( b[i] * (__special) n->temp > maxBoltz )
        {
            maxBoltz = b[i] * (__special) n->temp;
        }
    }
    
    __special boltzSum = 0;
    __special comp = 0;
    __special t, y;

    for( i=0; i < n->nb; i++ )
    {
        b[i] = exp( (__special) n->temp * b[i] - maxBoltz);
        KAHAN( boltzSum, b[i], y, t, comp );
    }


    for( i=0; i < n->nb; i++ )
    {
        b[i] /= boltzSum;
    }
}

void CalculateBelief( Node *n, __special *x, __special *b )
{
    uint i, j;
    __special bSum, bDiff, bNorm;

    bNorm = 0;

    __special outerComp = 0;
    __special outerT, outerY;

    for( i=0; i < n->nb; i++ )
    {
        bSum = 0;
        __special innerComp = 0;
        __special innerT, innerY;

        for( j=0; j < n->ns - n->nc; j++ )
        {
            bDiff = x[j] - (__special) n->mu[i][j];
            bDiff *= bDiff;
            bDiff *= (__special) n->starv[i];

            KAHAN( bSum, bDiff, innerY, innerT, innerComp );
        }
        b[i] = 1 / sqrt(bSum);

        KAHAN( bNorm, b[i], outerY, outerT, outerComp );
    }

    for( i=0; i < n->nb; i++ )
    {
        b[i] /= bNorm;
    }

    Boltzmannize( n, b );
}

__special BeliefMSE( uint n, __special *b, __special *bx )
{
    uint i;
    __special diff;
    __special mse = 0;
    __special comp = 0;
    __special t, y;

    for( i=0; i < n; i++ )
    {
        diff = b[i] - bx[i];
        diff *= diff;

        KAHAN( mse, diff, y, t, comp );
    }

    return mse;
}

void ConstrainInput( Node *nTmp, __special *x )
{
    if( nTmp->ni % 4 != 0 )
        fprintf(stderr, "there was a porblm\n");

    uint nc = nTmp->ni / 4;

    uint nOffset;
    uint n, i;
    float norm;
    for( n=0; n < 4; n++ )
    {
        norm = 0;

        nOffset = n*nc;

        for( i=0; i < nc; i++ )
        {
            if( x[nOffset+i] < 0 ) x[nOffset+i] *= -1;
            if( x[nOffset+i] > 1 ) x[nOffset+i] = 1;
            norm += x[nOffset+i];
        }
        for( i=0; i < nc; i++ )
        {
            x[nOffset+i] /= norm;
        }
    }
    
    norm = 0;

    nOffset = nTmp->ni;

    for( i=0; i < nTmp->nb; i++ )
    {
        if( x[nOffset+i] < 0 ) x[nOffset+i] *= -1;
        if( x[nOffset+i] > 1 ) x[nOffset+i] = 1;
        norm += x[nOffset + i];
    }
    
    for( i=0; i < nTmp->nb; i++ )
    {
        x[nOffset + i] /= norm;
    }
    
    nOffset = nTmp->ni + nTmp->nb;
    for( i=0; i < nTmp->np; i++ )
    {
        if( x[nOffset+i] < 0 ) x[nOffset+i] *= -1;
        if( x[nOffset+i] > 1 ) x[nOffset+i] = 1;
        norm += x[nOffset + i];
    }
    
    for( i=0; i < nTmp->np; i++ )
    {
        x[nOffset + i] /= norm;
    }
}

// generate a belief sample by metropolis
void SampleInputFromBelief( Node *n, float *xf )
{
    uint i, j;
    __special x[n->ns];
    __special xGrad[n->ns];

    __special xEpsP[n->ns];
    __special xEpsN[n->ns];

    __special belief[n->nb];
    __special bx[n->nb];
    __special bxEps[n->nb];

    __special bMSE, bEpsPMSE, bEpsNMSE;

    __special gradEps = 1e-9;

    uint nIts, nit;

    nIts = 50; 

    // init x to fuzzy distance of winning centroid
    for( i=0; i < n->ns; i++ )
    {
        x[i] = NormRND(n->mu[n->genWinner][i], 0.1);//n->sigma[n->genWinner][i]);

        if( x[i] > 1 ) x[i] = 1;
        if( x[i] < 0 ) x[i] = 0;
    }
    
    for( i=0; i < n->nb; i++) 
    {
        belief[i] = (__special) n->belief[i];
    }
    //printf("\n");


    __special g;

    ConstrainInput( n, x );
    
    CalculateBelief( n, x, bx );
    bMSE = BeliefMSE( n->nb, bx, belief );
    for( nit=0; nit < nIts; nit++ )
    {

        /*
        printf("belief:\n");
        for( i=0; i < n->nb; i++ )
        {
            printf("%0.3f ", belief[i]);
        }
        printf("\n");
        printf("bx:\n");
        for( i=0; i < n->nb; i++ )
        {
            printf("%0.3f ", bx[i]);
        }
        printf("\n");
        printf("x:\n");
        for( i=0; i < n->ns; i++ )
        {
            printf("%0.3f ", x[i]);
        }
        printf("\n");
        */

        for( i=0; i < n->ns; i++ )
        {
            for( j=0; j < n->ns; j++ )
            {
                xEpsN[j] = x[j];
                xEpsP[j] = x[j];
            }
            xEpsN[i] -= gradEps;
            xEpsP[i] += gradEps;

            CalculateBelief( n, xEpsP, bxEps );
            bEpsPMSE = BeliefMSE( n->nb, bxEps, belief );
            CalculateBelief( n, xEpsN, bxEps );
            bEpsNMSE = BeliefMSE( n->nb, bxEps, belief );

            xGrad[i] = (bEpsPMSE - bEpsNMSE) / (2*gradEps);
        }
        
        CalculateBelief( n, x, bx );
        bMSE = BeliefMSE( n->nb, bx, belief );
        
        log_info("%f\n", bMSE);

        g = 0.001;
        __special xTmp[n->ns];
        __special gMSE = bMSE;
        __special prev_gMSE = bMSE;
        while( true )
        {
            for( i=0; i < n->ns; i++ )
            {
                xTmp[i] = x[i];
            }

            for( i=0; i < n->ns; i++ )
            {
                xTmp[i] -= g * xGrad[i];
            }

            __special bxTmp[n->nb];

            CalculateBelief( n, xTmp, bxTmp );
            gMSE = BeliefMSE( n->nb, bxTmp, belief );
            
            if( gMSE > bMSE || gMSE > prev_gMSE || g > 16 ) break;
            
            prev_gMSE = gMSE;
            g *= 2;
        }

        g /= 2;

        for( i=0; i < n->ns; i++ )
        {
            x[i] -= g * xGrad[i];
        }

        ConstrainInput( n, x );
    }

#ifndef UNIT_TEST
    printf("bx:\n");
    for( i=0; i < n->nb; i++ )
    {
        printf("%0.3f ", bx[i]);
    }
    printf("\n");
    printf("final x:\n");
    for( i=0; i < n->ns; i++ )
    {
        printf("%0.3f ", x[i]);
    }
    printf("\n");
#endif

    for( i=0; i < n->ns; i++ )
    {
        xf[i] = (float) x[i];
    }
}

void GenerateInputFromBelief( Destin *d, float *frame )
{
    uint n, j, nMeans, bIdx;
    int i;
    float sampledInput[d->maxNs], sample;
    Node *nTmp;

    // initialize the frame
    for( i=0; i < d->layerSize[0] * d->nci[0]; i++ )
    {
        frame[i] = 0;
    }

    for( nMeans=0; nMeans < NMEANS; nMeans++ )
    {
        // proceed down the network to the input layer
        for( n = d->nNodes-1; n >= d->layerSize[0]; n-- )
        {
            nTmp = &d->nodes[n];

            SampleInputFromBelief( nTmp, sampledInput );

            // pass sampled input to children's previous belief
            for( i=0; i < nTmp->nChildren; i++ )
            {
                if (nTmp->children[i] == NULL)
                {
                    continue;
                }

                uint muCol = i*nTmp->children[i]->nb;
                float maxBelief = 0;
                uint genWinner = 0;

                for( j=0; j < nTmp->children[i]->nb; j++ )
                {
                    nTmp->children[i]->belief[j] = sampledInput[muCol+j];
                    if( nTmp->children[i]->belief[j] > maxBelief )
                    {
                        genWinner = j;
                        maxBelief = nTmp->children[i]->belief[j];
                    }
                }
                nTmp->children[i]->genWinner = genWinner;
            }
        }

        // output winning centroids to frame
        for( n=0; n < d->layerSize[0]; n++ )
        {
            nTmp = &d->nodes[n];

            //SampleInputFromBelief( nTmp, sampledInput );

            for( i=0; i < nTmp->ni; i++ )
            {
                sampledInput[i] = nTmp->mu[nTmp->genWinner][i];
                frame[nTmp->inputOffsets[i]] += log(sampledInput[i]);
            }
        }
    }

    // normalize output to 0-1.
    float frameMax = 0;
    float frameMin = 1;

    for( i=0; i < d->layerSize[0] * d->nci[0]; i++ )
    {
        frame[i] = exp(frame[i] / NMEANS);

        if( frame[i] <= frameMin )
        {
            frameMin = frame[i];
        }

        if( frame[i] > frameMax )
        {
            frameMax = frame[i];
        }
    }

    frameMax -= frameMin;

    for( i=0; i < d->layerSize[0] * d->nci[0]; i++ )
    {
        frame[i] -= frameMin;
        frame[i] /= frameMax;
    }

    // feed frame up
    /*
    for( i=0; i < d->nLayers; i++ )
    {
        FormulateBelief( d, frame );
    }
     */
}



void DisplayLayerFeatures( Destin *d, int layer, int node_start, int nodes )
{
    uint i, j, n, u, b, sqrtPatch;

    uint width, height;

    Node * startNode = GetNodeFromDestin(d, layer, 0,0);
    Node * node;

    sqrtPatch = (uint) sqrt(startNode->ni);
    int ncount = nodes == 0 ? d->layerSize[layer] : nodes;

    height = ncount * sqrtPatch;
    width = startNode->nb * sqrtPatch;

    float *frame;

    MALLOC( frame, float, width*height );

    for( n=node_start; n < node_start + ncount; n++ )
    {
        node = GetNodeFromDestinI(d, layer, n);
        for( b=0; b < node->nb; b++ )
        {
            for( u=0, i=0; i < sqrtPatch; i++ )
            {
                for( j=0; j < sqrtPatch; j++, u++ )
                {
                    frame[(n * sqrtPatch + i) * sqrtPatch * node->nb + j + b * sqrtPatch ] = node->mu[b][u];
                }
            }
        }
    }

    CvSize size;
    size.height = height;
    size.width = width;
    IplImage *featuresOut = cvCreateImageHeader(size, IPL_DEPTH_32F, 1);
    featuresOut->imageData = (char *) frame;
    featuresOut->imageDataOrigin = featuresOut->imageData;

    size.height = height * 8;
    size.width = width * 8;
    IplImage *bigImg = cvCreateImage(size, IPL_DEPTH_32F, 1);
    cvResize(featuresOut, bigImg, CV_INTER_NN);

    cvShowImage("Features!!", bigImg);

    cvReleaseImageHeader(&featuresOut);
    cvReleaseImage(&bigImg);

    FREE( frame );
}

void DisplayFeatures( Destin *d )
{
    uint i, j, n, u, b, sqrtPatch;

    uint width, height;

    sqrtPatch = (uint) sqrt(d->nci[0]);

    height = d->layerSize[0] * sqrtPatch;
    width = d->nb[0] * sqrtPatch;

    float *frame;

    MALLOC( frame, float, width*height );

    for( n=0; n < d->layerSize[0]; n++ )
    {
        for( b=0; b < d->nodes[n].nb; b++ )
        {
            for( u=0, i=0; i < sqrtPatch; i++ )
            {
                for( j=0; j < sqrtPatch; j++, u++ )
                {
                    frame[(n*sqrtPatch+i) * sqrtPatch*d->nodes[n].nb + j + b*sqrtPatch] = d->nodes[n].mu[b][u];
                }
            }
        }
    }

    CvSize size;
    size.height = height;
    size.width = width;
    IplImage *featuresOut = cvCreateImageHeader(size, IPL_DEPTH_32F, 1);
    featuresOut->imageData = (char *) frame;
    featuresOut->imageDataOrigin = featuresOut->imageData;

    size.height = height * 2;
    size.width = width * 2;
    IplImage *bigImg = cvCreateImage(size, IPL_DEPTH_32F, 1);
    cvResize(featuresOut, bigImg, CV_INTER_CUBIC);

    cvShowImage("Features!!", bigImg);

    cvReleaseImageHeader(&featuresOut);
    cvReleaseImage(&bigImg);

    FREE( frame );
}

void ResetStarvTrace( Destin *d )
{
    uint i, j;
    Node *nTmp;

    for( i=0; i < d->nNodes; i++ )
    {
        nTmp = &d->nodes[i];

        for( j=0; j < nTmp->nb; j++ )
        {
            nTmp->starv[j] = 1;
        }
    }
}


// grab a node at a particular layer, row, and column
struct Node * GetNodeFromDestin( Destin *d, uint l, uint r, uint c )
{
    //TODO: make this function faster

    // check layer bounds
    if( l >= d->nLayers )
    {
        fprintf(stderr, "GetNodeFromDestin(): layer requested is out of range!\n");
        return NULL;
    }

    uint layerSizeSqRoot = d->layerWidth[l];

    // check row bounds
    if( r >= layerSizeSqRoot )
    {
        fprintf(stderr, "GetNodeFromDestin(): row requested is out of range!\n");
        return NULL;
    }

    // check column bounds
    if( c >= layerSizeSqRoot )
    {
        fprintf(stderr, "GetNodeFromDestin(): column requested is out of range!\n");
        return NULL;
    }

    // grab the node index
    return GetNodeFromDestinI(d,l, r * layerSizeSqRoot + c);
}

// grab a node at a particular layer, node index
struct Node * GetNodeFromDestinI( Destin *d, uint l, uint nIdx)
{
    return &d->nodes[d->layerNodeOffsets[l] + nIdx];
}

void Uniform_ResetStats(Destin * d){
    int l, c, ns, s;
    for(l = 0 ; l < d->nLayers ; l++){
        ns = GetNodeFromDestin(d,l,0,0)->ns;
        for(c = 0 ; c < d->nb[l]; c++){
            d->uf_winCounts[l][c] = 0;
            for(s = 0 ; s < ns ;s++){
                d->uf_avgDelta[l][c][s] = 0;
                d->uf_avgSquaredDelta[l][c][s] = 0;
            }
        }
        for(s = 0 ; s < ns; s++)
        {
            d->uf_avgAbsDelta[l][s] = 0;
        }
    }
    return;
}

void GetLayerBeliefs( Destin * d, uint layer, float * beliefs )
{
    uint n, offset, i;
    uint nodeOffset = d->layerNodeOffsets[layer];

    offset = 0;
    for( n=0; n < d->layerSize[layer]; ++n )
    {
        float * belief = d->nodes[nodeOffset + n].outputBelief;
        for ( i=0; i < d->nb[layer]; ++i, ++offset)
        {
            beliefs[offset] = belief[i];
        }
    }

}
