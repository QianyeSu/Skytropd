project(
  'skytropd-zero-crossing',
  ['c', 'fortran'],
  version: '0.1.0',
  default_options: ['warning_level=2', 'buildtype=release', 'b_lto=true'],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy_cmd = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
)
inc_np = include_directories(incdir_numpy_cmd.stdout().strip())

host_system = host_machine.system()
cc = meson.get_compiler('c')
fc = meson.get_compiler('fortran')

c_args = ['-DNPY_NO_DEPRECATED_API=NPY_1_19_API_VERSION']
fortran_args = []

if host_system == 'windows' and cc.get_id() == 'msvc'
  c_args += ['/O2']
else
  c_args += ['-O3']
endif

if fc.get_id() == 'gcc'
  fortran_args += ['-O3']
endif

py.extension_module(
  '_zero_crossing_backend',
  ['_zero_crossing_module.c', 'Fortran/zero_crossing.f90'],
  include_directories: inc_np,
  dependencies: [py_dep],
  c_args: c_args,
  fortran_args: fortran_args,
  install: true,
  subdir: 'skytropd',
)
