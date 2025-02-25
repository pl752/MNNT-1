project('MNNT-1', 'cpp', default_options : ['cpp_std=c++14'])
cmake = import('cmake')
cmake_opts = cmake.subproject_options()

if build_machine.system() != 'windows'
  cmake_opts.add_cmake_defines({'CMAKE_CXX_VISIBILITY_PRESET': 'hidden'})
endif

add_global_arguments('-I../boost', language: 'cpp')
add_global_arguments('-DBOOST_ALL_NO_LIB', language: 'cpp')
add_global_arguments('-DBOOST_PROGRAM_OPTIONS_STATIC_LINK', language: 'cpp')

boost_opt = cmake.subproject('program_options', options: cmake_opts).dependency('boost_program_options')
eigen = subproject('eigen').get_variable('eigen_dep')

executable('MNNT-1', 'main.cpp', dependencies: [boost_opt, eigen])
