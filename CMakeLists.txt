cmake_minimum_required(VERSION 3.10)

# Project name and programming language
project(Tacc LANGUAGES CXX CUDA)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set GPU Computing Capability
set(CMAKE_CUDA_ARCHITECTURES 86)
# Specify CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# add this to succeed compile and link libtorch
set(CMAKE_PREFIX_PATH ~/.local/lib/python3.10/site-packages/torch/share/cmake/Torch)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Include Python and Torch
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${Python3_INCLUDE_DIRS}
                    ${TORCH_INCLUDE_DIRS}
)

# Add the executable with cpp or cu file
add_executable(Tacc 
    lib/util.cu
    src/main.cpp)

target_link_libraries(Tacc PRIVATE cudart cublas ${TORCH_LIBRARIES})

enable_testing()
add_subdirectory(test)
add_subdirectory(googletest)

# add custom target
# make customize cuda function to be compatible with pytorch
add_custom_target(
    Pysetup ALL
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/lib/setup.py build --build-lib ${CMAKE_BINARY_DIR}/lib.linux-x86_64-3.10
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    COMMENT "Running setup.py for self-customized CUDA function"
)

# Ensure the target directory exists
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/app")
# install the execute python srcipt into app
add_custom_target(
    PyInstall ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Copying Python files to runtime output directory..."
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/app ${CMAKE_BINARY_DIR}/app
    COMMENT "Copy Python files to the runtime output directory"
)