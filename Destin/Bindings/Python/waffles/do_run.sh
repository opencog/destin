#!/bin/bash

if [ "$#" != "1" ] ; then
    echo "Usage: $0 <run_id>"
    exit
fi
run_id="$1"

run_dir="run_${run_id}/"
template_dir="./template/"


#setup directories
function setup_dirs {
    cp -r "$template_dir" "$run_dir"
    cp ../OutputBeliefs.txt "$run_dir"
    cd "$run_dir"
}

# Format DeSTIN beliefs to be used by waffles
function format_beliefs {
    # convers OutputBeliefs.txt from destin BeliefExporter into a format that waffles can use
    waffles_transform import OutputBeliefs.txt -whitespace > OutputBeliefs.txt.arff 

    # generate a comma sperated list of class labels
    # from the first column
    classes=`cat OutputBeliefs.txt.arff | awk -F ',' '{print $1}' | sort -u  | grep -v '@' | grep '[0-9]' | paste -s -d,`

    # replaces
    # @ATTRIBUTE attr0 real
    # With
    # @ATTRIBUTE attr0 {0,4}
    sed -i "s/\(@ATTRIBUTE attr0.*\)real/\1{$classes}/" OutputBeliefs.txt.arff
}

function train {
    # Train the model.
    echo "Training..."
    waffles_cmd="waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 400 -learningrate 0.005 -momentum 0.3 -windowepochs 100 -minwindowimprovement 0.002"
    echo "$waffles_cmd > nn.model" > train_nn.sh
    chmod +x nn_train.sh
    time $waffles_cmd > nn.model
}

# Report the Results
function report_results {
    echo "Predicting..."
    
    #run the prediction
    waffles_learn predict nn.model OutputBeliefs.txt.arff -labels 0 > nn_predict.arff

    # convert to csv
    waffles_transform export nn_predict.arff
    
    # get the original labels
    cat OutputBeliefs.txt | cut -f1 > labels.txt
    
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
    format_beliefs
    train
    report_results
}

main
