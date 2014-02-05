runcmd="waffles_learn train -seed 0 OutputTrainBeliefs.txt.arff -labels 0 neuralnet $@"
runcmd2="$0 $@"


touch 0.run
num=`ls *.run | sort -n | tail -1 | cut -d. -f1`
num=$(( num + 1 ))
rm 0.run

(
start_time=`date +%s`
echo "Start: " `date`
echo

echo $runcmd
echo $runcmd2

$runcmd > ${num}.nn.model

echo waffles_learn test ${num}.nn.model OutputTestBeliefs.txt.arff -labels 0
waffles_learn test ${num}.nn.model OutputTestBeliefs.txt.arff -labels 0

echo "End: " `date`
end_time=`date +%s`

echo "Time: $(( end_time - start_time )) seconds"
) > ${num}.run
