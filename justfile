# the default recipe is build
default: build run

build:
    mkdir -p build
    cd build && cmake .. && cmake --build .

run:
    ./build/reyn

clean:
    rm -rf build