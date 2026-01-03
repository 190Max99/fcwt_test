#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "fCWT" for configuration "Release"
set_property(TARGET fCWT APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(fCWT PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libfCWT.so.2.0"
  IMPORTED_SONAME_RELEASE "libfCWT.so.2.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS fCWT )
list(APPEND _IMPORT_CHECK_FILES_FOR_fCWT "${_IMPORT_PREFIX}/lib/libfCWT.so.2.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
