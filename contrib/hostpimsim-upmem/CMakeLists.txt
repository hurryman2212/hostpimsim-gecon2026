# Enlist source from current directory recursively
file(GLOB_RECURSE PROJECT_SOURCES *.c *.cc *.cpp *.cxx)

# Add external library to link
find_library(liboverlaysys overlaysys REQUIRED)
set(PROJECT_LIBRARIES libhostpimsim.so ${liboverlaysys})

# DO NOT EDIT THE BELOW!

# Declare project for this scope
project(lib${ROOT_PROJECT_NAME}-upmem-preload)

# Declare object output(s) to build with source
add_library(${PROJECT_NAME} OBJECT ${PROJECT_SOURCES})

# Add project-specific compiler options
target_compile_options(${PROJECT_NAME}
                       PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_FLAGS_PROJECT}>)
target_compile_options(${PROJECT_NAME}
                       PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS_PROJECT}>)

# Include directory for headers
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add `-fpic` compiler option for compiling objects
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C>:-fpic>)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fpic>)

# Declare project static and shared libraries to link with the compiled objects
add_library(${PROJECT_NAME}.so SHARED)
target_link_libraries(${PROJECT_NAME}.so PRIVATE ${PROJECT_NAME})

# Link external libraries we need
if(PROJECT_LIBRARIES)
  target_link_libraries(${PROJECT_NAME}.so PRIVATE ${PROJECT_LIBRARIES})
endif()

# Enforce output name format
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" OUTPUT_NAME
                                                           ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}.so PROPERTIES PREFIX "" OUTPUT_NAME
                                                              ${PROJECT_NAME})

# Set custom directory for "install"
install(TARGETS ${PROJECT_NAME}.so LIBRARY)
