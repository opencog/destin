To see these instructions formatted, view this file at https://github.com/opencog/destin/blob/master/WindowsBuild.md

You will need around 10GB of free disk space to build and install the prerequisites. This has been tested on Window 7 64 bit.

This windows build is still a work in progress. Unit tests need to be fixed and Python and Java bindings need to be tested.

Install Qt Creator IDE (using MinGW)
----------------------------------------------
Go to http://qt-project.org/downloads

Look for, download and install using the "Qt 5.1.1 for Windows 32-bit (MinGW 4.8, OpenGL, 666 MB)" link. When running the installer it may take a minute or two before anything shows up due to the large file size. 

(Optional) Install Java
------------------------------
You only need this if you want to play with the Java language bindings. These have not been tested on windows yet.

Download and install a Java JDK using the defaults.

Install Python
---------------
Download and install python using http://www.python.org/ftp/python/2.7.5/python-2.7.5.msi
Use the defaults. Installs to C:\Python27\

Install SWIG
---------------
* Download from http://www.swig.org/download.html, look for the windows prebuild executable http://prdownloads.sourceforge.net/swig/swigwin-2.0.11.zip
* Extract everything to C:\swigwin-2.0.11 so you can find it's README and other files at C:\swigwin-2.0.11\README. You will probably need to move up the nested 
C:\swigwin-2.0.11\swigwin-2.0.11 directory to C:\


Install CMake
-----------------
Download and install using the defaults: http://www.cmake.org/files/v2.8/cmake-2.8.12-win32-x86.exe

Have it added to the windows path.	

( Link was found from this page: http://www.cmake.org/cmake/resources/software.html )

Install prebuilt OpenCV for Mingw
----------------------------------
* Download from Ted Sanders' dropbox: 
https://dl.dropboxusercontent.com/u/49968823/destin/opencv-2.4.6-mingw-x86.zip Google Chrome may complain that it's not commonly downloaded. Use the dropdown button to keep it.

* Move the zip to c:\ and extract all. Move the inner "opencv-2.4.6" folder out to the root to c:\, then delete the left over empty opencv-2.4.6-mingw-x86 folder. When done
properly you should able to find the file OpenCVConfig.cmake at C:\opencv-2.4.6\mingwbuild\install\OpenCVConfig.cmake and the rest of the files there too. 

Build OpenCV From source ( If not using prebuilt from previous step):
----------------------------------------------------------------------
( Note: if you have an opencv installation already at c:\ then you may want to rename it to c:\dontuse-opencv so that cmake does not automatically try to use that one. )

* Download the source from https://github.com/Itseez/opencv/archive/2.4.6.zip

* Unzip to c:\ so that you can find the README file at c:\opencv-2.4.6\README

* Open the QT Creator IDE.
File -> Open File or Project -> Select the C:\opencv-2.4.6\CMakeLists.txt file
.
* Set the build location to C:\opencv-2.4.6\mingwbuild -> Next

* First time setup, find your cmake executable, probably located at C:\Program Files\CMake 2.8\bin\cmake.exe 

* Next
* Generator: MinGW Generator

* Click "Run CMake". 

The log window should say at the end:

	-- Configuring done
	-- Generating done
	-- Build files have been written to: C:/opencv-2.4.6/mingwbuild

* Click Finish.

* You should see that OpenCV is opened. Click the "Projects" tab button on the left.

* Under Build Steps section, click the "Details" drop down button on the right. Check the "install" target. Uncheck the "all" target. 

* In "Additional Arguments" section use -j to speed up compiling. For example, use -j8 for if your CPU has a quad core with hyper threading.

* These settings save automatically. Now press the hammer button in the bottom left to begin building. Press the "Compile Output" button on the bottom bar to see build progress. This may take several minutes.

##### Now to build the Debug libs:
* In the top menu bar click Build -> Run CMake. In the "Arguments:" section put:

`-DCMAKE_BUILD_TYPE=Debug`
 
* Set the Generator to "MingGW Generatator". Click "Run CMake". You can ignore the "ImportError: No module named numpy.distutils" error. 

* You should again see `-- Build files have been written to: C:/opencv-2.4.6/mingwbuild` at the end of the log window. Click Finish. Build the same as before using the hammer button.

* In windows explorer inspect the C:\opencv-2.4.6\mingwbuild\install\lib directory. You should see about 18 pairs of libopencv dlls. In the pairs, one is the release build dll, and the other is the debug build dll postfixed with d.

* Close the OpenCV project from QT Creator using File -> Close Project "OpenCV".

Recommended step: Install Cygwin Git:
-------------------------------------------------

If you have trouble with the following instructions also see http://x.cygwin.com/docs/ug/setup.html#setup-cygwin-x-installing for alternate instructions.

* Visit http://www.cygwin.com/install.html to download the setup ( direct link: http://www.cygwin.com/setup-x86.exe ). I've only tested with x86 setup version on windows 7 64bit.

* Launch the setup file.

*  Next -> Install From Internet

* Root Directory C:\cygwin 

* Install for all Users.

* Next.

* Choose a Local Package Directory, anywhere but C:\cygwin. I chose C:\Users\Ted\Documents\Downloads\cygwin-install

* Next.

* Direct connection should work fine. 

* Next.

* Choose A Download site. If you happen to pick a bad one the download may be slow.

* Next.

* In the package selection window, click the "View" button in the top right to put it to "Full" view. This lists all packages without any categorization. Select packages to be installed by clicking the icon in the "New" column next to each package to be installed.

* Use the search box, type in "git". 

* Choose the packages: git, git-completion, git-gui, gitk

* Search for and select packages xinit and xorg-server. These will let you use git gui and gitk using the x server.

* After selecting packages, click next. Click next again to let it automatically resolve dependencies. 

* Click Next again, it will start installing everything. 

* Select the option to create the shortcut on your desktop.

* To run git gui and gitk, from the cygwin command prompt, type:
 
   $ startxwin

* This will open an additional window. You will also see a X icon in your quick start icons area.

In that window you can then run:

    $ git gui &

or 

    $ gitk --all &

Git gui is useful for making commits. Gitk is very useful to see commit history. 

If you want to be able to run these commands from your current terminal without using the additional window, then after running startxwin then run this command:

    $ export DISPLAY=":0"

You can append that command to the end of the file c:\cygwin\home\<Your user>\.bashrc to have that command automatically run in each new cygwin terminal.

Using Cygwin Git to get the Destin Source
--------------------------------------------------------
Open the Cygwin terminal.
Type the commands:

    $ cd
    $ git clone http://github.com/opencog/destin.git
    $ cd destin
    $ git submodule init
    $ git submodule update

This command will download the source to C:\cygwin\home\<your user>\destin


Building DeSTIN with Qt Creator IDE
----------------------------------------------

* Open Qt Creator IDE. 

* File -> Open File Or Project. Locate your local destin repository and open the destin/Destin/CMakeLists.txt file

* For Build Location, use the default destin\Destin-build directory. Next.

* You may need to "Choose CMake Executable". Your cmake.exe is probably located at C:\Program Files (x86)\CMake 2.8\bin\cmake.exe. Next.

* Run CMake: 
	* Arguments: none
	* Generator: MinGW Generator
	* Click Run CMake.

* You can ignore the warning "Couldn't find jni. Is a jdk installed? Not building java interfaces"

The log window should say something like this at the bottom

    -- Build files have been written to: C:/cygwin/home/<your home/destin/Destin-build

If you get the error:

    Cannot find source file:
    cluster/src/cluster
	
Then you forgot to run the git submodule command from the "Using Cygwin Git to get the Destin Source" section. You may need to learn how to use the submodule command if you use a different git tool.

* Click Finish.

* Click the "Projects" tab button on the left.
    * Under "Build Steps" section, click the "Details" drop down button to the very right
    * Check the "install" target. Uncheck the "all" target. 
    * In "Additional Arguments" section use -j to speed up compiling. For example, use -j8 for if your CPU has a quad core with hyper threading.

These settings save automatically. 

* Now press the hammer button in the bottom left to begin building. 

* Press the "Compile Output" button on the bottom bar to see build progress. In newer version of Qt Creator you have to click the updown/arrow button to bring up a menu to select "Compile Output".

* You can ignore the compile warning "Returning a pointer or reference in a director method is not recommended." 

Running DeSTIN unit tests:
----------------------------
These tests should be ran before checking in code.

* Navigate to c:\<your destin home>\Destin-build\install\bin in the command line or Windows Explorer. 

* Execute mingw_run_test.bat

* The tests should finish running in less than 10 seconds or so. The last line of the output should say:

	FINISHED TESTING: PASS
	ALL PASS


Running DeSTIN compiled executables:
-----------------------------------------------------

#### To run the destin.exe executable in Cygwin, 

Type the commands:

    cd ~/destin/Destin-build/install/bin
    ./destin.exe

#### To run it from Qt Creator IDE:

* With the DeSTIN project open, click the "Projects" tab button on the left. 

* Open the Run settings by clicking the pill shaped "Run" button near the top ( to the right of the "Build" button ).

* In the Run section, use the "Run configuration" dropdown to chose the executable to run. Unfortunately, each time you change the executable, you have to re enter the settings below.

* Set the Working Directory to C:\<your destin home>\Destin-build\install\bin

* In the Run Environment settings:
	* Use Build Environment

* Click the green Play (triangle) button on the left pane to run it. Press the "Application Output" button on the bottom to see the output. Sometimes Build -> "Run Without Deployment" is convenient to use.

Using Python Bindings:
-----------------------------------------------------

Create (or append to) the windows environment variable PYTHONPATH the value:

	C:\<your destin home>\Destin-build\install\bin
	
Then in your python scripts:

	import pydestin
