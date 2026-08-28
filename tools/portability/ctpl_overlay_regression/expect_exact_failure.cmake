if(NOT DEFINED TEST_PROGRAM OR NOT EXISTS "${TEST_PROGRAM}")
    message(FATAL_ERROR "TEST_PROGRAM must name an existing executable")
endif()
if(NOT DEFINED EXPECTED_DIAGNOSTIC)
    message(FATAL_ERROR "EXPECTED_DIAGNOSTIC is required")
endif()

if(DEFINED TEST_ARGUMENT)
    execute_process(
        COMMAND "${TEST_PROGRAM}" "${TEST_ARGUMENT}"
        RESULT_VARIABLE TEST_RESULT
        OUTPUT_VARIABLE TEST_STDOUT
        ERROR_VARIABLE TEST_STDERR
        TIMEOUT 10
    )
else()
    execute_process(
        COMMAND "${TEST_PROGRAM}"
        RESULT_VARIABLE TEST_RESULT
        OUTPUT_VARIABLE TEST_STDOUT
        ERROR_VARIABLE TEST_STDERR
        TIMEOUT 10
    )
endif()

if(TEST_RESULT EQUAL 0)
    message(FATAL_ERROR "Mutation probe unexpectedly passed")
endif()

set(ACTUAL_OUTPUT "${TEST_STDOUT}${TEST_STDERR}")
set(EXPECTED_OUTPUT "${EXPECTED_DIAGNOSTIC}\n")
if(NOT ACTUAL_OUTPUT STREQUAL EXPECTED_OUTPUT)
    message(FATAL_ERROR "Unexpected mutation diagnostic.\nExpected: ${EXPECTED_OUTPUT}Actual: ${ACTUAL_OUTPUT}")
endif()
