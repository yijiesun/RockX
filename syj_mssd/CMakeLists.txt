cmake_minimum_required(VERSION 3.10)

project(syj_mssd)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

set(OUT_PATH ${PROJECT_NAME})

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB IMGS ${CMAKE_CURRENT_SOURCE_DIR}/*.jpg)

add_executable(${PROJECT_NAME}
        ${SRCS} ../src/v4l2/v4l2.cpp ../src/screen/screen.cpp ../src/config.cpp)

target_link_libraries(${PROJECT_NAME}
        ${RockX_LIBS} ${OpenCV_LIBS} pthread)

install(TARGETS ${PROJECT_NAME} DESTINATION ${OUT_PATH})
install(FILES ${IMGS} config.txt DESTINATION ${OUT_PATH})