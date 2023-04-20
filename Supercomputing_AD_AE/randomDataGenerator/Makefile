# Define the C++ compiler to use
CXX = g++

# Define the compiler flags
CXXFLAGS = -Wall -std=c++11

# Define the target executables
TARGETS = normalRand uniformRand

# Default target (all)
all: $(TARGETS)

# Rule to build normalRand
normalRand: normalRand.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Rule to build uniformRand
uniformRand: uniformRand.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<


# Rules to generate .pts files
Normal100M2D.pts: normalRand
	./normalRand 100000000 2  $@

Normal300M2D.pts: normalRand
	./normalRand 300000000 2  $@

Normal100M3D.pts: normalRand
	./normalRand 100000000 3  $@

Normal300M3D.pts: normalRand
	./normalRand 300000000 3  $@

Uniform100M2D.pts: uniformRand
	./uniformRand 100000000 2  $@

Uniform100M3D.pts: uniformRand
	./uniformRand 100000000 3  $@
# Rule to clean up the build
clean:
	rm -f $(TARGETS) *.pts
