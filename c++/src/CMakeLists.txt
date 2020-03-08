cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER C:/dev/mingw/bin/gcc.exe)
set(CMAKE_CXX_COMPILER C:/dev/mingw/bin/g++.exe)
# set(CMAKE_MAKE_PROGRAM C:/dev/mingw/bin/mingw32-make.exe)
set(CMAKE_BUILD_TYPE "Debug")
message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message("CMAKE_MAKE_PROGRAM = ${CMAKE_MAKE_PROGRAM}")


project(tmlr)
# set(SOURCE main.cxx)

#Bring the headers, such as Student.h into the project
include_directories(include)
 
#Can manually add the sources using the set command as follows:
#set(SOURCES src/mainapp.cpp src/Student.cpp)
 
#However, the file(GLOB...) allows for wildcard additions:
# file(GLOB SOURCES "src/*.cpp")
file(GLOB SOURCES "*.cpp")
 
add_executable(tmlr ${SOURCES})

# add the executable
# add_executable(TMLR main.cxx)
# configure_file(TutorialConfig.h.in TutorialConfig.h)

# target_include_directories(Tutorial PUBLIC
#                            "${PROJECT_BINARY_DIR}"
#                            )
                           
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@