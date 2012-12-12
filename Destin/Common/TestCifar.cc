

#include "CifarSource.h"

#include  <stdio.h>

int main(int argc, char ** argv){
    CifarSource cs("/home/ted/destin_git_repo/Destin/Data/CIFAR/cifar-10-batches-bin", 1);



    printf("argc: %i\n", argc);
    int im = argc > 1 ? atoi(argv[1]) : 1000;

    cs.setCurrentImage(im);
    cv::imshow("output", cs.getColorImageMat(128, 128));
    cv::waitKey();

    cv::imshow("output", cs.getGrayImageMat(128, 128));
    cv::waitKey();
    return 0;
}
