
#include "VideoSource.h"
#include "DestinNetworkAlt.h"
#include "Transporter.h"
#include "stdio.h"

int main(int argc, char ** argv){
    //VideoSource vs(false, "./destin_video_test.avi");
    VideoSource vs(true, "./destin_video_test.avi");
    SupportedImageWidths siw = W512;

    uint centroid_counts[]  = {20,16,14,12,10,8,4,2};

    INetwork * network = new DestinNetworkAlt(siw, 8, centroid_counts);

    vs.enableDisplayWindow();

    Transporter t;



    float * beliefs;
    while(vs.grab()){

        t.setSource(vs.getOutput());
        t.transport(); //move video from host to card

        network->doDestin(t.getDest());

        beliefs = network->getNodeBeliefs(7,0,0);

        //printf("%c[2A", 27); //earase two lines so window doesn't scroll
        //printf("0: %f\n", beliefs[0]);
        //printf("1: %f\n", beliefs[1]);

        //printf("\033[%d;%dH", 10, 20);  /* move cursor (row 10, col 20) */
        //printf("\033[2J");      /* clear screen */
        network->printBeliefGraph(0,0,0);
    }

    delete network;
    return 0;
}
