#!/bin/sh
destin_alt=`cd ../../DavisDestin ; pwd`
java_destin=`pwd`
the_path="${destin_alt}:${java_destin}"

echo java -cp build/classes/ -Djava.library.path="$the_path" javadestin.Dashboard
java -cp build/classes/ -Djava.library.path="$the_path" javadestin.Dashboard
