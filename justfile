CORES := `nproc`

DOCS_DIR := "builddocs"
DOXYFILE := "builddocs/Doxyfile"
DOXYGEN_OUTPUT := "builddocs/doxygen-output"

PYTHON := `cd builddocs && pyenv which python`

# default recipe: build and run
default: format build run-tests run
test: format build run-tests
clean-build: clean setup format build run-tests run

setup:
    mkdir -p build
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

build:
    cmake --build build --parallel {{CORES}}

run:
    ./build/reyn

run-tests:
    ./build/tests

clean:
    rm -rf build

format:
    find src -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) -exec clang-format -i -style=file -- {} +

docs:
    mkdir -p {{DOCS_DIR}}
    rm -rf docs
    mkdir -p docs
    touch docs/.nojekyll
    rm -rf {{DOCS_DIR}}/api
    rm -rf {{DOCS_DIR}}/_build
    rm -rf {{DOCS_DIR}}/_templates
    rm -rf {{DOCS_DIR}}/doxygen-output
    doxygen {{DOXYFILE}}
    cd {{DOCS_DIR}} && {{PYTHON}} -m sphinx -b html . _build/html
    # replace relativee URLs in markdown
    sed -i 's@<img src="./res/icon.png" width=300 height = 300/>@ @g' ./builddocs/_build/html/index.html
    mv builddocs/_build/html/* docs
