cores := `nproc`

# default recipe: build and run
default: format build run-tests run
test: format build run-tests
clean-build: clean setup format build run-tests run

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

format:
    find src -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) -exec clang-format -i -style=file -- {} +