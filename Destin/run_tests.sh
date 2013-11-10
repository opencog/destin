#!/bin/sh

# This file runs unit tests.
# Check the bottom line for failures.
# A test executable is marked for failure if its return code is not 0.

# These are the tests to run:
tests="
./TreeMining/testTreeMiner
./DavisDestin/run_test.sh
./Bindings/Python/test.sh
"

all_pass="yes"

# Run all the test
for test in $tests
do
    echo "################################"
    echo "###### Test Suite: $test #######"
	# if the return code is not 0, then there is a failure
	if ! "$test" ; then
		echo "The test '$test' FAILED!"
		all_pass="no"
	fi
done

# check if there were failures
if [ "$all_pass" = "yes" ] ; then
	echo "ALL PASS"
else
	echo "TEST FAILED"
fi
