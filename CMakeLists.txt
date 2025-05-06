cmake_minimum_required(VERSION 3.14)
project(Phase2___Implementation)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Phase2___Implementation
        main.cpp
        src/Phase2.cpp
        src/Phase2.h
)

target_link_libraries(Phase2___Implementation ${OpenCV_LIBS})

set_target_properties(Phase2___Implementation PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Phase2___Implementation"
)
