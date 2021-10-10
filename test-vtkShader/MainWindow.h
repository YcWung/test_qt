#include "vtk_gl.h"
#include <QMainWindow>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr);

    void SetupScene();

protected:
    void showEvent(QShowEvent* e) override {
        QMainWindow::showEvent(e);
        SetupScene();
        QMainWindow::showEvent(e);
    }
    QVTKOpenGLNativeWidget* vtkw;
    vtkNew<vtkRenderer> ren;
};