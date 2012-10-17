#ifndef DestinIterationFinishedCallback_H
#define DestinIterationFinishedCallback_H


#include "INetwork.h"

class DestinIterationFinishedCallback {
	public:
		virtual ~DestinIterationFinishedCallback(){}
		virtual void callback(INetwork & network) = 0;
};

#endif
