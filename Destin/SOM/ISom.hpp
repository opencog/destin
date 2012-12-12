#ifndef ISOM_HPP
#define ISOM_HPP
#include<string>

using namespace std;

typedef void (*ItemToColorFunc)(float * item, int &r, int &g, int &b);

class ISom {

public:
    virtual ~ISom(){}

    virtual void train_iterate(float * data) = 0;

    virtual void showSimularityMap(
                    string,     // window name
                    uint,       // 1/2 width of pixel neighborhood
                    int,        // window height, does not affect the map dimensions only output window size
                    int         // window width,  does not affect the map dimensions only output window size
        ) = 0;

    virtual void showMap(string windowName) = 0;

    virtual void setItemToColorFuncPointer(ItemToColorFunc function) = 0;



};

#endif // ISOM_HPP
