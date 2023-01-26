CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/examples) 

SET(TESTFUNCTIONS ${PROJECT_SOURCE_DIR}/examples/testfunctions.cpp)

#Test for random number generator
ADD_EXECUTABLE(test_rand ./test_rand.cpp)

#Test for parsers
ADD_EXECUTABLE(test_parser ../utils/parser.cpp ./test_parser.cpp)

#Test for grid
ADD_EXECUTABLE(test_grid ./test_grid.cpp)

#Test for MCMC
ADD_EXECUTABLE(test_mcmc ../src/mcmc_sampler.cpp ./test_mcmc.cpp)

#Test for Cholesky decomposition
ADD_EXECUTABLE(test_chol ./test_chol.cpp)

#Test Restore
ADD_EXECUTABLE(test_restore ./test_restore.cpp  ${TESTFUNCTIONS})
add_dependencies(test_restore bayesopt)
TARGET_LINK_LIBRARIES(test_restore bayesopt)

#Test Save
ADD_EXECUTABLE(test_save ./test_save.cpp  ${TESTFUNCTIONS})
add_dependencies(test_save bayesopt)
TARGET_LINK_LIBRARIES(test_save bayesopt)

#Test File Parser
ADD_EXECUTABLE(test_fileparser ./test_fileparser.cpp)
add_dependencies(test_fileparser bayesopt)
TARGET_LINK_LIBRARIES(test_fileparser bayesopt)

#Test Initial Samples
ADD_EXECUTABLE(test_initial_samples ./test_initial_samples.cpp  ${TESTFUNCTIONS})
add_dependencies(test_initial_samples bayesopt)
TARGET_LINK_LIBRARIES(test_initial_samples bayesopt)
