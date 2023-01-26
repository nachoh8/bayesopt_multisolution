CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#Test functions
ADD_LIBRARY(bofunctions STATIC testfunctions.cpp)
add_dependencies(bofunctions bayesopt)
TARGET_LINK_LIBRARIES( bofunctions bayesopt)

#Optimization with system calls using CLI parameters
ADD_EXECUTABLE(branin_system_calls ./branin_system_calls.cpp)
add_dependencies(branin_system_calls bayesopt)
TARGET_LINK_LIBRARIES(branin_system_calls bayesopt)

#Optimization with system calls with XML interface
ADD_EXECUTABLE(branin_xml ./branin_xml.cpp)
add_dependencies(branin_xml bayesopt)
TARGET_LINK_LIBRARIES(branin_xml bayesopt)

#Batch Bayesian optimization via LP demo
ADD_EXECUTABLE(bo_bbo_lp ./bo_bbo_lp.cpp )
add_dependencies(bo_bbo_lp bayesopt)
TARGET_LINK_LIBRARIES(bo_bbo_lp bayesopt)

#Continuous demo
ADD_EXECUTABLE(bo_cont ./bo_cont.cpp )
add_dependencies(bo_cont bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_cont bayesopt bofunctions)

#Discrete demo
ADD_EXECUTABLE(bo_disc ./bo_disc.cpp )
add_dependencies(bo_disc bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_disc bayesopt bofunctions)

#Simple 1D test
ADD_EXECUTABLE(bo_oned ./bo_oned.cpp )
add_dependencies(bo_oned bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_oned bayesopt bofunctions)

#Branin
ADD_EXECUTABLE(bo_branin ./bo_branin.cpp  )
add_dependencies(bo_branin bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_branin bayesopt bofunctions)

ADD_EXECUTABLE(bo_branin_noise ./bo_branin_noise.cpp  )
add_dependencies(bo_branin_noise bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_branin_noise bayesopt bofunctions)

ADD_EXECUTABLE(bo_branin_mcmc ./bo_branin_mcmc.cpp  )
add_dependencies(bo_branin_mcmc bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_branin_mcmc bayesopt bofunctions)

#Rembo
ADD_EXECUTABLE(bo_rembo ./bo_rembo.cpp  )
add_dependencies(bo_rembo bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_rembo bayesopt bofunctions)

#Hartmann
ADD_EXECUTABLE(bo_hartmann ./bo_hartmann.cpp  )
add_dependencies(bo_hartmann bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_hartmann bayesopt bofunctions)

#Camelback
ADD_EXECUTABLE(bo_camelback ./bo_camelback.cpp  )
add_dependencies(bo_camelback bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_camelback bayesopt bofunctions)

#MCMC comparison
ADD_EXECUTABLE(bo_compare ./bo_compare.cpp  )
add_dependencies(bo_compare bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_compare bayesopt bofunctions)

#Test Polish Optimization
ADD_EXECUTABLE(bo_polish_opt ./bo_polish_opt)
add_dependencies(bo_polish_opt bayesopt bofunctions)
TARGET_LINK_LIBRARIES(bo_polish_opt bayesopt bofunctions)


IF(Boost_PROGRAM_OPTIONS_FOUND)
  IF(BAYESOPT_PARALLEL)
    find_package(OpenMP)
    if (OPENMP_FOUND)
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    ENDIF()
  ENDIF()

    ADD_EXECUTABLE(bo_cli ./bo_cli.cpp  )
    add_dependencies(bo_cli bayesopt bofunctions)
    TARGET_LINK_LIBRARIES(bo_cli bayesopt bofunctions ${Boost_LIBRARIES})

    ADD_EXECUTABLE(bo_cli_ubo ./bo_cli_ubo.cpp  )
    add_dependencies(bo_cli_ubo bayesopt bofunctions)
    TARGET_LINK_LIBRARIES(bo_cli_ubo bayesopt bofunctions ${Boost_LIBRARIES})
    
    ADD_EXECUTABLE(bo_cli_crit_sampling ./bo_cli_crit_sampling.cpp  )
    add_dependencies(bo_cli_crit_sampling bayesopt bofunctions)
    TARGET_LINK_LIBRARIES(bo_cli_crit_sampling bayesopt bofunctions ${Boost_LIBRARIES})

    ADD_EXECUTABLE(bo_cli_batch_sampling ./bo_cli_batch_sampling.cpp  )
    add_dependencies(bo_cli_batch_sampling bayesopt bofunctions)
    TARGET_LINK_LIBRARIES(bo_cli_batch_sampling bayesopt bofunctions ${Boost_LIBRARIES})

    ADD_EXECUTABLE(bo_cli_random_sampling ./bo_cli_random_sampling.cpp  )
    add_dependencies(bo_cli_random_sampling bayesopt bofunctions)
    TARGET_LINK_LIBRARIES(bo_cli_random_sampling bayesopt bofunctions ${Boost_LIBRARIES})
    
    ADD_EXECUTABLE(bo_cli_posterior_sampling ./bo_cli_posterior_sampling.cpp  )
    add_dependencies(bo_cli_posterior_sampling bayesopt bofunctions)
    TARGET_LINK_LIBRARIES(bo_cli_posterior_sampling bayesopt bofunctions ${Boost_LIBRARIES})
    
    ADD_EXECUTABLE(bo_cli_boltzmann_crit_sampling ./bo_cli_boltzmann_crit_sampling.cpp  )
    add_dependencies(bo_cli_boltzmann_crit_sampling bayesopt bofunctions)
    TARGET_LINK_LIBRARIES(bo_cli_boltzmann_crit_sampling bayesopt bofunctions ${Boost_LIBRARIES})

    ADD_EXECUTABLE(bo_cli_entropy_search ./bo_cli_entropy_search.cpp  )
    add_dependencies(bo_cli_entropy_search bayesopt bofunctions)
    TARGET_LINK_LIBRARIES(bo_cli_entropy_search bayesopt bofunctions ${Boost_LIBRARIES})
ENDIF()

enable_language(C)  #For OpenGL and bo_display demo

#Display test
find_package(GLUT)
find_package(OpenGL)
if(OPENGL_FOUND AND GLUT_FOUND)

  SET(DISPLAY_SRC     
    ../utils/displaygp.cpp
    ../utils/displaycrit.cpp
    )

  if(GLUT_LIBRARY MATCHES freeglut_static.lib)
      add_definitions(-DFREEGLUT_STATIC)
  endif()


  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/ext/matplotpp
		      ${GLUT_INCLUDE_DIRS} 
		      ${GLUT_INCLUDE_DIR} 
		      ${OpenGL_INCLUDE_DIRS})			  

  link_directories(${GLUT_LIBRARY_DIRS}
		   ${OpenGL_LIBRARY_DIRS})

  add_definitions(${GLUT_DEFINITIONS}
		  ${OpenGL_DEFINITIONS})

  ADD_LIBRARY(matplotpp STATIC 
    ../ext/matplotpp/matplotpp.cc 
    ../ext/matplotpp/gl2ps.c
    )

  TARGET_LINK_LIBRARIES(matplotpp
    ${GLUT_LIBRARY} ${OPENGL_LIBRARY})

  ADD_EXECUTABLE(bo_display 
    ./bo_display.cpp  
    ${DISPLAY_SRC})
  add_dependencies(bo_display bayesopt bofunctions matplotpp)
  TARGET_LINK_LIBRARIES(bo_display bayesopt bofunctions matplotpp
    ${GLUT_LIBRARY} ${OPENGL_LIBRARY})

  ADD_EXECUTABLE(bo_nonstationary_display 
    ./bo_nonstationary_display.cpp
    ${DISPLAY_SRC})
  add_dependencies(bo_nonstationary_display bayesopt bofunctions matplotpp)
  TARGET_LINK_LIBRARIES(bo_nonstationary_display bayesopt bofunctions matplotpp
    ${GLUT_LIBRARY} ${OPENGL_LIBRARY})

  ADD_EXECUTABLE(bo_hard_display 
    ./bo_hard_display.cpp 
    ${DISPLAY_SRC})
  add_dependencies(bo_hard_display bayesopt bofunctions matplotpp)
  TARGET_LINK_LIBRARIES(bo_hard_display bayesopt bofunctions matplotpp
    ${GLUT_LIBRARY} ${OPENGL_LIBRARY})


  ADD_EXECUTABLE(bo_branin_display
    ./bo_branin_display.cpp  
    ${DISPLAY_SRC})
  add_dependencies(bo_branin_display bayesopt bofunctions matplotpp)
  TARGET_LINK_LIBRARIES(bo_branin_display bayesopt bofunctions matplotpp
    ${GLUT_LIBRARY} ${OPENGL_LIBRARY})

  ADD_EXECUTABLE(bo_crit_sampling_display
    ./bo_crit_sampling_display.cpp
    ${DISPLAY_SRC})
  add_dependencies(bo_crit_sampling_display bayesopt bofunctions matplotpp)
  TARGET_LINK_LIBRARIES(bo_crit_sampling_display bayesopt bofunctions matplotpp
    ${GLUT_LIBRARY} ${OPENGL_LIBRARY})

endif()
