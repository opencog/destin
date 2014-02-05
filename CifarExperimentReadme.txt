# Download waffles machine learning linux binary from http://sourceforge.net/projects/waffles/files/waffles/2013-12-09/
# Add the bin directory that contains waffles_train to your PATH environment variable.
# You can do this by running the command and adding to the end of your ~/.bashrc file and restarting your shell:
export PATH="$PATH:~/waffles/bin"

cd ~/
cd destin
git submodule init
git submodule update

# Add a remote to Ted's git repo
git remote add ted https://github.com/tpsjr7/destin.git
git fetch ted

git stash # stash any outgoing changes
git checkout cifar_run_13 # Experiment runs numbered 12 and 13.
git checkout -b cifar_run # create a branch at that commit incase you want to make further commits.

cd Destin/
cmake .
make clean
make -j8
cd Bindings/Python/

# Download the cifar dataset from http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz and unzip it
# Edit cifar_experiment_config.py and modify the cifar_dir variable at the top to point to the unzipped cifar data directory that contains the data batches.
# The cifar_experiment_config.py has incremental variable changes for each of the runs, so be aware that each variable may be set further
# down in the file.

# This is tested with python 2.7.x
# Run the cifar experiment:
python cifar_experiment.py

# You will see a graph change as it moves. It shows the "Quality" measurement of each layer as it is training. 
# Right now it only trains each layer one by one. On my cpu ( Intel Core I7 ), this take about 20 minutes. 
# It will say "Done" at then end and then pop a window of centroid images. You can use your arrow keys to 
# move between the layers. Pressing escape will close the window and end the cifar_experimeny.py script

# The script generates OutputTrainBeliefs.txt and OutputTestBeliefs.txt. 
# OutputTrainBeliefs.txt are the beliefs output by destin when it was fed the images that destin was used to train on.
# OutputTestBeliefs.txt are the beliefs output by destin when fed images from the test cifar batch ( specified by the 

# The first column of the two files is the CIFAR image class label ( 0 to 9 ) of the image that was used to generate
# the output beliefs for that row.

# Right now, cifar_experiment_config.py is set to run_id="013"
# so can view the centroid images generated here:

ls cifar_experiment_runs/013

# some scripts to run waffles 
cd waffles_scripts

# Setup the experiment run directory. Copies and formats the OutputBeliefs from the python directory.
./setup_run.sh 13 # takes about a minute

# Uses waffles to train a neural network. It will create a small log of the run called n.run, where n is an incrementing integer for
# each time train_test.sh is ran.
# On my CPU, with these particular parameters, this runs for 7 minutes. There will be no output while running.

./train_test.sh -addlayer 40 -addlayer 80 -learningrate .002 -momentum 0.2 -windowepochs 100 -minwindowimprovement 0.001 &

# We launched it in the background with the & at the end.  Check up on our runnings training jobs ( assuming we're using bash shell on Unbuntu: )

jobs

# My Intel I7 has 8 hardware threads so I can run up to 8 train jobs simultaneously without slowing down.

# After it's finished running, take a look at the results

ls *.run
cat 1.run

# Should see something like this:
    Start:  Sun Jan 26 19:11:46 CST 2014

    waffles_learn train -seed 0 OutputTrainBeliefs.txt.arff -labels 0 neuralnet -addlayer 40 -addlayer 80 -learningrate .002 -momentum 0.2 -windowepochs 100 -minwindowimprovement 0.001
    ./train_test.sh -addlayer 40 -addlayer 80 -learningrate .002 -momentum 0.2 -windowepochs 100 -minwindowimprovement 0.001
    waffles_learn test 1.nn.model OutputTestBeliefs.txt.arff -labels 0
    Mean squared error: 0.5893
    End:  Sun Jan 26 19:18:28 CST 2014
    Time: 402 seconds

# It reports 0.5893 as the error, since we're using nominals as the labels, waffles is actually showing the accuracy as 1 - 0.5893 = 41.07%

# The best results I've gotten so far is %44.02 accuracy with these settings
# however the runtime was 33 hours.

./train_test.sh -addlayer 400 -addlayer 400 -learningrate .0002 -momentum 0.2 -windowepochs 200 -minwindowimprovement 0.0001


