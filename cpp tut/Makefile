CXX=g++
CXXFLAGS=-std=c++11 -g -Wall
SRC1=densematrix.cpp
SRC2=memorymgmt.cpp
SRC3=inherit.cpp
SRC4=numeric.cpp

default: densematrix memorymgmt inherit numeric 
densematrix: $(SRC1)
	$(CXX) $(CXXFLAGS) $< -o $@
memorymgmt: $(SRC2) $(HDR) 
	$(CXX) $(CXXFLAGS) $< -o $@
inherit: $(SRC3) 
	$(CXX) $(CXXFLAGS) $< -o $@
numeric: $(SRC4) 
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *~ *~ densematrix memorymgmt inherit numeric
	rm -rf *.dSYM

