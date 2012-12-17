
cd ../../
cmake . 
cd Bindings/Java

rm gen_src/javadestin/*
make clean
make

ant -f build.xml clean
ant -f build.xml
