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
