# Code to accompany "Superancillary Equations for the Multiparameter Equations of State in REFPROP 10.0"

## Requirements

* Cmake
* Modern compiler supporting C++17 standard

## Build & run the expansion builder

1. Open a shell in the root of the code
2. ``mkdir build`` then ``cd build``
3. ``cmake .. -DCMAKE_BUILD_TYPE=Release``
4. ``cmake --build . --target fitcheb --config Release``
5. On windows with visual studio, ``Release\fitcheb.exe``, or on linux, ``./fitcheb``

After running, the output should be in the ``output`` folder. If files already exist in that destination they will not be over-written.

## Figures

Once the expansions have been generated, the figures can be created by running the script ``make_figures.py``

Notes:

* The code expects that the working directory when running is a subfolder of this code in order to properly find the fluid files
* The number of cores is set to 6 in the runner code. You might want to modify that.

## Questions/issues

Please email ian.bell@nist.gov, or find me online and I'll do my best to help.
