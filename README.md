# Code to accompany "Efficient and Precise Representation of Pure Fluid Phase Equilibria with Chebyshev Expansions"

## Requirements

* Cmake
* Modern compiler supporting C++17 standard

## Build&run the expansion builder

1. Open a shell in the root of the code
2. ``mkdir build`` then ``cd build``
3. ``cmake .. -DCMAKE_BUILD_TYPE=Release``
4. ``cmake --build . --target fastcheb``
5. On windows with visual studio, ``Release\fastcheb.exe``, or on linux, ``./fastcheb``

## Questions/issues

Please email ian.bell@nist.gov, or find me online and I'll do my best to help.