# Angle of Repose test case

add_executable(angle_of_repose angle_of_repose.cpp)

target_link_libraries(angle_of_repose
    emsdem
)

target_include_directories(angle_of_repose PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Create results directory at build time
add_custom_command(TARGET angle_of_repose POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_BINARY_DIR}/results
)
