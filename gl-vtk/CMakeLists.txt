cmake_minimum_required(VERSION 3.8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(gl-vtk)
set(SRC main.cpp MainWindow.cpp shader.cpp vtk_gl.cpp)
set(HDR MainWindow.h shader.h vtk_gl.h)

find_package(VTK COMPONENTS 
  vtkCommonColor
  vtkCommonCore
  vtkFiltersSources
  vtkInteractionStyle
  vtkInteractionWidgets
  vtkRenderingContextOpenGL2
  vtkRenderingCore
  vtkRenderingFreeType
  vtkRenderingGL2PSOpenGL2
  vtkRenderingOpenGL2
  vtkGUISupportQt
  vtkRenderingQt
  QUIET
)

find_package(OpenGL)

find_package(Qt5 QUIET
  REQUIRED COMPONENTS Core Gui Widgets
)

message (STATUS "VTK_VERSION: ${VTK_VERSION}, Qt Version: ${Qt${VTK_QT_VERSION}Widgets_VERSION}")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_executable(${PROJECT_NAME} ${SRC} ${HDR})
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets)
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OPENGL_gl_LIBRARY})
vtk_module_autoinit(TARGETS ${PROJECT_NAME} MODULES ${VTK_LIBRARIES})