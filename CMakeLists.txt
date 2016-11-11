
add_executable(load load.cpp)
add_executable(main main.cpp)
target_link_libraries(main opennn)
target_link_libraries(load opennn)
install(DIRECTORY data/ DESTINATION data/)

install(TARGETS load RUNTIME DESTINATION bin)
install(TARGETS main RUNTIME DESTINATION bin)
