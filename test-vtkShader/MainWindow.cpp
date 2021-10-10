#include "MainWindow.h"

static const float test_points[] = {
    0, 0, 0,
    -0.5, 0, 0,
    0.5, 0, 0,
    0, -0.5, 0,
    0, 0.5, 0
};

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    vtkw = new QVTKOpenGLNativeWidget(parent);
    this->setCentralWidget(vtkw);
    vtkw->renderWindow()->AddRenderer(ren);
}

void MainWindow::SetupScene() {
    static bool init = true;
    if (init) {
        init = false;
    } else {
        return;
    }
    vtkw->renderWindow()->MakeCurrent();
    vtkNew<GLPointsProp> points;
    points->Reserve(256);
    points->Upload(test_points, sizeof(test_points) / sizeof(float));
    ren->AddViewProp(points);
    vtkw->renderWindow()->Render();
}