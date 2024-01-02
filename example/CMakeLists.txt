cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(example)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(
    example
    main.cpp
)

target_link_libraries(
    example
    "${TORCH_LIBRARIES}"
)

set_property(
    TARGET
    example
    PROPERTY CXX_STANDARD 17
)
