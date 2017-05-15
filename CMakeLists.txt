cmake_minimum_required(VERSION 2.8.3)
project(blueflow)
project(px4flow)
project(fastflow)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)



FILE(GLOB SOURCE "./*.cpp")
FILE(GLOB SOURCE "./bluefox/*.cpp")
 

ADD_EXECUTABLE(blueflow ${SOURCE} blueflow.cpp)

ADD_EXECUTABLE(px4flow ${SOURCE} px4flow.cpp)

ADD_EXECUTABLE(fastflow ${SOURCE} FastFlow.cpp)


find_package(OpenCV REQUIRED)
find_package(Threads)



TARGET_LINK_LIBRARIES(blueflow ${OpenCV_LIBS})


TARGET_LINK_LIBRARIES(fastflow ${OpenCV_LIBS})


TARGET_LINK_LIBRARIES(px4flow ${OpenCV_LIBS})




# dynamic reconfigure
# generate_dynamic_reconfigure_options(cfg/Bluefox2Dyn.cfg)

# services
#add_service_files(FILES SetExposeSrv.srv)
#generate_messages(DEPENDENCIES std_msgs)


include_directories("./bluefox")
MESSAGE(STATUS 'OpenCV_LIBS: ${OpenCV_LIBS}')

#add_subdirectory(src)

# dependencies
