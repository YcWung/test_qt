#include <QApplication>
#include "OpenGLWidget.h"

int main(int argc, char** argv) {
	QApplication qapp(argc, argv);
	OpenGLWidget glw;
	glw.show();

	return qapp.exec();
}