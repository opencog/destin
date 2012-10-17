
cd ../
cmake . 
cd JavaDestin

rm gen_src/javadestin/*
make clean
make

ant -f build.xml
