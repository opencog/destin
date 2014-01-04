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
