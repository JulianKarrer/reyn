cores := `nproc`

# default recipe: build and run
default: build run-tests run
test: build run-tests
clean-build: clean setup build run-tests run

setup:
    mkdir -p build
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

build:
    cmake --build build --parallel {{cores}}

run:
    ./build/reyn

run-tests:
    ./build/tests

clean:
    rm -rf build