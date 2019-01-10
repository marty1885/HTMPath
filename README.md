# HTM Path
A HTM Application learning to detct anomalies of a object's path.

## How to  build
```
mkdir build
cd build
cmake ..
make -j4
```

## Dependency
 * xtensor
 * SFML
 * GLM
 * C++14 capable compiler

## How to use

`HTMPath` is a GUI application with visualizations of the HTM model and other info.
These are the controls.

* w - move orbit up 5 pixels
* a - move orbit left 5 pixels
* s - move orbit down 5 pixels
* d - move orbit right 5 pixels
* up - move orbit up 50 pixels
* down - move orbit left 50 pixels
* left - move orbit down 50 pixels
* right - move orbit right 50 pixels
* o - move orbit center to (250, 250)
* l - set orbit radius from 100 to 105
* f - fast forward
* p - slow motion (limit to 10 FPS)
* c - lock the circle at (50, 50)
* Left Shift - Force learning (Learning is disabled when orbit is altered)
* n - Force disable learning

`bench` is a CLI tool for generating test results as fast as possible. Change `GridCellEncoder2D` to `LocEncoder2D` in bench.cpp to switch between Grid Cells and Scalar Encoders.

## Licsence
AGPL v3
