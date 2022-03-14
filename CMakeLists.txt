cmake_minimum_required(VERSION 3.1)

project(windmill-detect)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(windmill_detect OBJECT src/WindmillDetect.cpp)
target_include_directories(windmill_detect PUBLIC src)
target_link_libraries(windmill_detect ${OpenCV_LIBS})
