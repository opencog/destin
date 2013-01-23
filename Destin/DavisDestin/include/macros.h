#ifndef __MACROS
#define __MACROS

// unsigned int macro
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>

#define PRINTF printf


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

#endif
