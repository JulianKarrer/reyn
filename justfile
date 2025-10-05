cores := `nproc`

# default recipe: build and run
default: build run

setup:
    mkdir -p build
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

build:
    cmake --build build --parallel {{cores}}

run:
    ./build/reyn

clean:
    rm -rf build