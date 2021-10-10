#include <QApplication>
#include <QSurfaceFormat>
#include "MainWindow.h"

int main(int argc, char** argv) {
    QApplication qapp(argc, argv);
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    MainWindow w;
    w.show();

    return qapp.exec();
}