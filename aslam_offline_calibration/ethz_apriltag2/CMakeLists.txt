cmake_minimum_required(VERSION 2.8)

project(ethz_apriltag2)

find_package(catkin REQUIRED COMPONENTS cmake_modules)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
    DEPENDS 
    INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
    LIBRARIES ${PROJECT_NAME}
)

add_definitions(-fPIC -O3)
include_directories(include ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

#library
file(GLOB SOURCE_FILES "src/*.cc")
add_library(${PROJECT_NAME}  ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${catkin_LIBRARIES})

#demo
if(NOT APPLE)
#  add_executable(apriltags_demo src/example/apriltags_demo.cpp src/example/Serial.cpp)
#  target_link_libraries(apriltags_demo ${PROJECT_NAME} v4l2)
endif()


