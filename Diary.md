Development Diary
=================

10/19 - 10/21/2012
------------------

Continued writing a unit test for destin so I could figure out why the beliefs
were not updating like before. Found a few things that changed that affected it.
The default starvation trace coefficient was set lower so it would take longer
for other centroids to be adjusted. The default  temperature I set for my destin
network using the new boltzman  distribution was set too low so that over time
the beliefs would be smoothed out to uniform and stop evolving. Andrew
implemented a new rule for the rate of adjusting centroids to be  proportional
to 1/(centroid update count) so the centroids would eventually stop moving. Will
have to think about what the best time is to stop centroid training in the robot
vision experiments.

So to get the beliefs more reactive again I reverted the starvation trace
coefficient to be higher, turned off applying the bolzman distribution to the
beliefs and  set the learning rate to be fixed (not going got check that in
because I dont want to mess up Andrew's experiments, but I suppose we can make
that stuff configurable which some already is). Basically there's lots of
factors to play around with. 

Going to move on again to implementing uniform destin.

10/21 - 10/30/2012
------------------
Making good progress on uniform destin. Continued updating unit test, somewhat
following test driven development. Nodes are using the shared centroids to
compare their inputs with. Working through the logic to average centroid update
vectors in case multiple nodes choose the same shared centroid to update. Also
working through the logic on the starvation trace.

10/31 - 11/16/2012
------------------
Uniform DeSTIN translation invariance through shared centroids is pretty much
complete ( see
http://wiki.opencog.org/w/DestinOpenCog#Uniform_DeSTIN.2C_Part_One )

For each iteration of the DeSTIN algorithm, each node picks a winning centroid
from the shared pool of centroids for the layer. If multiple nodes from a layer
pick the same centroid as winner then their movement vectors are averaged and
then moved at the end. If a shared centroid is picked by at least one node in
the level then its starvation trace credit is reset. 

The SaveDestin, LoadDestin, and CreateDestin ( loading from a config file) now
works with uniform destin. To switch between uniform destin and classic destin a
boolean flag isUniform was added as an argument to InitDestin. 

Unfortunately having shared centroids will require a different parallelization
strategy when we port it back to CUDA. 

11/17 - 12/12/2012
------------------
Made the centroid update strategy configurable via function pointers to make it
easy to switch between a fixed learning rate or a decaying one.

Added Python bindings to DeSTIN to make it easier to interativly experiment with
it.

Worked on creating a visualization for DeSTIN. I simply created an image for
each layer by taking the winning centroid index of each node and converting it
into grayscale pixel. You can watch in time how long it takes for each layer to
come "onlnie" as the bottom layers learn first. With uniform DeSTIN turned on a
clear picture come through whereas on classic DeSTIN it looks like static
because the winning centroid labels at each node mean something different. 

Throug the visualization work I discovered that our heirarcy wasn't a 4x1
reduction in a square fashion at each level. Instead it was take in 4x4 square
regions in the bottom layer, but then the upperalyers were doing a 4x1 reduction
in a linear fashion instead. This was fixed by adding a lookup array for the
node inputs to force them to have childing arranged in a square.

I've spent about two days creating code to stream DeSTIN states to a self
organizing map to see if the belief states cluster with certain images. Will
need to spend more time to interpret the results. Currently however I'm going to
be moving on to implementing scale invariance as outlined here:
http://wiki.opencog.org/w/DestinOpenCog#Uniform_DeSTIN.2C_Part_Two

