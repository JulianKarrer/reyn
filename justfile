CORES := `nproc`

DOCS_DIR := "builddocs"
DOXYFILE := "builddocs/Doxyfile"
DOXYGEN_OUTPUT := "builddocs/doxygen-output"

PYTHON := `cd builddocs && pyenv which python`

# default recipe: build, test and run
default: format build run-tests run
# rebuild and run tests
test: format build run-tests
# clean up and then run
clean-build: clean setup format build run-tests run


# setup the cmake build
setup:
    mkdir -p build
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# build the program using cmake
build:
    cmake --build build --parallel {{CORES}}

# run the built program
run:
    ./build/reyn

# run all tests that were previously built
run-tests:
    ./build/tests

# remove the build folder
clean:
    rm -rf build

# Format all code using Clang
format:
    find src -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) -exec clang-format -i -style=file -- {} +


run-benches:
    cmake -B build -DBENCH=ON &&  cmake --build build --parallel {{CORES}} && ./build/tests
    cmake -B build -DBENCH=OFF


# Generate documentation
docs:
    # delete and recreate folders to purge previous docs
    mkdir -p {{DOCS_DIR}}
    rm -rf docs
    mkdir -p docs
    touch docs/.nojekyll
    rm -rf {{DOCS_DIR}}/api
    rm -rf {{DOCS_DIR}}/_build
    rm -rf {{DOCS_DIR}}/_templates
    rm -rf {{DOCS_DIR}}/doxygen-output

    # run benchmarks (cmake setup must be re-run with DBENCH)
    just run-benches

    # generate markdown from benchmark results
    mkdir -p {{DOCS_DIR}}/_staticc/benchmarks
    python3 builddocs/scripts/generate_benchmarks.py \
        --json-dir builddocs/_staticc/benchmarks \
        --html-dir builddocs/_staticc/benchmarks \
        --out-file builddocs/benchmarks/index.md
    
    # run doxygen
    doxygen {{DOXYFILE}}

    # run sphinx build
    cd {{DOCS_DIR}} && {{PYTHON}} -m sphinx -b html . _build/html

    # replace relative URLs in markdown
    sed -i 's@<img src="./res/icon.png" width=300 height = 300/>@ @g' ./builddocs/_build/html/index.html

    # move the final html output to the docs directory for display on github pages
    mv builddocs/_build/html/* docs
