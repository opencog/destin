#!/bin/bash

if [ "$#" = "0" ] ; then
    echo "Usage: $0 <run_id> <waffles_learn train params>"
	echo "Example: 1234 -labels 0 neuralnet -addlayer 400 -learningrate 0.005 -momentum 0.3 -windowepochs 100 -minwindowimprovement 0.002"
    exit
fi

run_id="$1"
shift

waffles_learn_params="$@"

run_dir="run_${run_id}/"
template_dir="./template/"

test_set="OutputTestBeliefs.txt"
train_set="OutputTrainBeliefs.txt"

#setup directories
function setup_dirs {
    cp -r "$template_dir" "$run_dir"
	cp ../"$test_set" "$run_dir"
	cp ../"$train_set" "$run_dir"
    cd "$run_dir"
}

# Format DeSTIN beliefs to be used by waffles
function format_beliefs {
	the_data_set="$1"
    # convers OutputBeliefs.txt from destin BeliefExporter into a format that waffles can use
    waffles_transform import "$the_data_set" -whitespace > "$the_data_set".arff 

    # generate a comma sperated list of class labels
    # from the first column
    classes=`cat "$the_data_set".arff | awk -F ',' '{print $1}' | sort -u  | grep -v '@' | grep '[0-9]' | paste -s -d,`

    # replaces
    # @ATTRIBUTE attr0 real
    # With
    # @ATTRIBUTE attr0 {0,4}
    sed -i "s/\(@ATTRIBUTE attr0.*\)real/\1{$classes}/" "$the_data_set".arff
}

function train {
    # Train the model.
    echo "Training..."
    waffles_cmd="waffles_learn train -seed 0 ${train_set}.arff ${waffles_learn_params}"
    echo "$waffles_cmd > nn.model" > nn_train.sh
    chmod +x ./nn_train.sh
    time $waffles_cmd > nn.model
}

# Report the Results
function report_results {
    echo "Predicting..."
    
    #run the prediction
    waffles_learn predict nn.model "$1".arff -labels 0 > nn_predict.arff
    
    # get the original labels
    cat "$1" | cut -f1 > labels.txt
    
    # get the predicted labels
    waffles_transform export nn_predict.arff > predictions.txt
  
    
    # calculate and report the accuracy
    # count where prediction != actual 
    failures=`paste labels.txt predictions.txt | awk '$1 != $2' | wc -l`
    total=`cat  labels.txt | wc -l`
    echo "Total samples: $total"
    echo "Failures: $failures"
    accuracy=$(echo "scale=4; 1.0 - ($failures / $total )" | bc )
    echo "Accuracy: $accuracy"
}

function main  {
    setup_dirs
	
    format_beliefs "$test_set"
	format_beliefs "$train_set"
	
    train
	
    report_results "$test_set"
}

#main
