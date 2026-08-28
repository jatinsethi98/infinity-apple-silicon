FILE(GLOB_RECURSE jma_SRC "*.cpp")

# include modules
# usage: CHECK_INCLUDE_FILES (<header> <RESULT_VARIABLE> )
INCLUDE (CheckIncludeFiles)

# check libmecab configurations
INCLUDE(libmecab/ConfigureChecks.cmake)

add_library(jma STATIC ${jma_SRC})
target_link_libraries(jma ZLIB::ZLIB)
if(APPLE)
    target_link_libraries(jma ${Iconv_LIBRARIES})
else()
    target_link_libraries(jma Iconv::Iconv)
endif()
