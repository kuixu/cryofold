wget https://github.com/soedinglab/MMseqs2/archive/71dd32ec43e3ac4dabf111bbc4b124f1c66a85f1.zip -O mmseqs2.zip
cd MMseqs2
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. ..
make -j8
make install 