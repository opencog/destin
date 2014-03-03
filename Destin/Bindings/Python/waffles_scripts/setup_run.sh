if [ "$#" = "0" ] ; then
    echo "Usage: $0 <run_id>"
    exit
fi


run_id="$1"
run_dir="run_${run_id}/"

test_set="OutputTestBeliefs.txt"
train_set="OutputTrainBeliefs.txt"
template_dir="./template/"

#setup directories
function setup_dirs {
    cp -r "$template_dir" "$run_dir"
	cp ../"$test_set" "$run_dir"
	cp ../"$train_set" "$run_dir"
    cd "$run_dir"
}


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


setup_dirs
	
format_beliefs "$test_set"
format_beliefs "$train_set"
	
