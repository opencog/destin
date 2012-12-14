#include "BrlySom.hpp"

CvPoint BrlySom::findBestMatchingUnit(float * data){
    int r, c;
    som.find_bmu(data, r, c);
    CvPoint p;
    p.x = c;
    p.y = r;
    return p;
}

bool BrlySom::saveSom(string filename){
    FILE * f;
    f = fopen(filename.c_str(), "w");
    if(f == NULL){
        fprintf(stderr, "Could not open file %s for writing.\n", filename.c_str());
        return false;
    }

    size_t wc  = 0; //write count

    wc += fwrite(&rows, sizeof(uint), 1, f );
    wc += fwrite(&cols, sizeof(uint), 1, f );
    wc += fwrite(&vector_dim, sizeof(uint), 1, f );
    uint t = som.getT();
    wc += fwrite(&t, sizeof(uint), 1, f);

    for(int r = 0 ; r < rows ; r++){
        for(int c = 0 ; c < cols ; c++){
            wc += fwrite(&(som.data()[r][c][0]), sizeof(float), vector_dim, f);
        }
    }

    fclose(f);

    size_t should_write = 4 + rows * cols  * vector_dim;

    if(wc != should_write){
        fprintf(stderr, "Trouble saving file %s\n", filename.c_str());
        fprintf(stderr, "Expected %lu, but saved %lu\n", should_write, wc);
        return false;
    }
    return true;

}

bool BrlySom::loadSom(string filename){
    FILE * f;
    f = fopen(filename.c_str(), "r");
    if(f == NULL){
        fprintf(stderr, "Could not open file %s for reading.\n", filename.c_str());
        return false;
    }

    size_t rc  = 0; //read count

    rc += fread(&rows, sizeof(uint), 1, f );
    rc += fread(&cols, sizeof(uint), 1, f );
    rc += fread(&vector_dim, sizeof(uint), 1, f );

    som = brly_som::Som<float>(rows, cols, vector_dim);
    uint t;
    rc += fread(&t, sizeof(uint), 1, f);
    som.setT(t);

    for(int r = 0 ; r < rows ; r++){
        for(int c = 0 ; c < cols ; c++){

            float fl[vector_dim];

            rc += fread(fl, sizeof(float), vector_dim, f);

            som.data()[r][c].clear();
            for(int i = 0; i < vector_dim ; i++){
                som.data()[r][c].push_back(fl[i]);
            }
        }
    }

    fclose(f);

    size_t should_read = 4 + rows * cols  * vector_dim;

    if(rc != should_read){
        fprintf(stderr, "Trouble loading file %s\n", filename.c_str());
        fprintf(stderr, "Expected %lu, but loaded %lu\n", should_read, rc);
        return false;
    }
    return true;

}


