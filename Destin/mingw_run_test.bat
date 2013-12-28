@ECHO OFF

REM This should be run from the install/bin directory of a windows mingw build

set run_test=test.exe testTreeMiner.exe

set has_errors=no
FOR %%F IN ( %run_test% ) DO (
	echo **************** Test Suite %%F
	%%F || set has_errors=yes
)

if "%has_errors%" == "yes" (
	echo SOME TEST FAILED!
) else (
	echo ALL PASS
)

pause