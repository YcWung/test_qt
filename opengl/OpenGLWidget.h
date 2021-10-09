#pragma once
#include "cu_gl.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions>

class OpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT
public:
	OpenGLWidget(QWidget* parent = nullptr);
	~OpenGLWidget();
protected:
	virtual void initializeGL() override;
	virtual void resizeGL(int w, int h) override;
	virtual void paintGL() override;
private:
    CuGLTex tex;

    Shader shader;

    //PointsGL points;

    PointsCuGL points_cugl;

    // Plane Vertices
    static constexpr const GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    static constexpr const char* vs_fs_code[] = {
        // Vertex Shader Code
        "#version 330 core\n"
        "layout(location = 0) in vec3 position;"
        "out vec2 UV;"
        "void main() {"
        "    gl_Position = vec4(position, 1.0);"
        "    UV = vec2(position) * 0.5 + 0.5;"
        "}",

        // Fragment Shader Code
        "#version 330 core\n"
        "in vec2 UV;"
        "out vec3 color;"
        "uniform sampler2D textureSampler;"
        "void main() {"
        "    color = texture(textureSampler, UV).rgb;"
        "}"
    };
};

