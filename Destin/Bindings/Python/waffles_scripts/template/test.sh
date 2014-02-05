if [ "$#" = "0" ]  
then
    echo "usage: $0 <model>"
    exit
fi
waffles_learn test -seed 1234 "$1" OutputBeliefs.txt.arff -labels 0
