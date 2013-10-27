#ifndef __MACROS
#define __MACROS

// unsigned int macro
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PRINTF printf

#define _MIN(a, b)       (a < b ? a : b)
#define _MAX(a, b)       (a > b ? a : b)

#ifdef UNIT_TEST
#define log_info(s...)do{}while(0)
#else
#define log_info(s...)do{printf(s);}while(0)
#endif

#define oops(s...) { fprintf(stderr, ## s); exit(1); }

#define MALLOC(s,t,n) {                                 \
    if((s = (t *) malloc(n*sizeof(t))) == NULL) {       \
        oops("error: malloc()\n");                      \
    }                                                   \
}

#define FREE(t) {                                       \
    if( t != NULL ) {                                   \
        free(t);                                        \
    } else {                                            \
        oops("error: free()\n");                        \
    }                                                   \
}

#ifdef _WIN32
typedef unsigned int uint;
#endif

#define MIN_SIZEV       16
#define SIZEV(n) ((n < MIN_SIZEV) ? MIN_SIZEV : 1 << ((uint)(log(n-1)/log(2)) + 1))

#define MALLOCV(s,t,n) {                                    \
    if((s = (t *) malloc(SIZEV(n*sizeof(t)))) == NULL) {    \
        oops("error: malloc()\n");                          \
    }                                                       \
}

#define REALLOCV(s,t,n,c) {                                 \
    if (SIZEV((n+c)*sizeof(t)) > SIZEV(n*sizeof(t))) {      \
        if((s = (t *) realloc(s, SIZEV((n+c)*sizeof(t)))) == NULL) {   \
            oops("error: realloc()\n");                     \
        }                                                   \
    }                                                       \
}

#endif
