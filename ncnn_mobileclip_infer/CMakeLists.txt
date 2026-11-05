cmake_minimum_required(VERSION 3.10)
project(ncnn_mobileclip_vulkan)
set(CMAKE_CXX_STANDARD 17)

find_package(ncnn REQUIRED)
find_package(Vulkan REQUIRED)
find_package(OpenCV REQUIRED)

find_program(GLSLANG_VALIDATOR glslangValidator)
if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found!")
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/preprocess.spv
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.0 ${CMAKE_CURRENT_SOURCE_DIR}/preprocess.comp -o ${CMAKE_CURRENT_BINARY_DIR}/preprocess.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/preprocess.comp
)
add_custom_target(shaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/preprocess.spv)

add_executable(ncnn_mobileclip_vulkan preprocess.cpp)
add_dependencies(ncnn_mobileclip_vulkan shaders)
target_link_libraries(ncnn_mobileclip_vulkan ncnn Vulkan::Vulkan ${OpenCV_LIBS})
