#include "OpenGLWidget.h"
#include "kernel.h"
#include <thread>
#include <chrono>

OpenGLWidget::OpenGLWidget(QWidget* parent) : QOpenGLWidget(parent)
{
}

OpenGLWidget::~OpenGLWidget()
{
    tex.Delete();
    //points.Delete();
    points_cugl.Delete();
    shader.Delete();
}

void OpenGLWidget::initializeGL()
{
	this->initializeOpenGLFunctions();
    InitGLEW();

    glDisable(GL_DEPTH_TEST);
    points_cugl.Create(sizeof(vertices) / sizeof(GLfloat) / 3);
    cudaMemcpy(points_cugl.cu_ptr, vertices, points_cugl.buffer_size, cudaMemcpyHostToDevice);
    tex.Create(this->width(), this->height());
    shader.Compile(vs_fs_code);

    // CUDA setup
    SetupCUDA();
}

void OpenGLWidget::resizeGL(int w, int h)
{
    glClearColor(0.0, 0.0, 0.0, 1.0); // Black background
    glViewport(0, 0, w, h);  // GL Screen size
}

void OpenGLWidget::paintGL()
{
    FillVBO(points_cugl.cu_ptr);
    tex.Map();
    // Kernel Launch
    kernelCall(this->width(), this->height(), tex.graphics_array);
    tex.Unmap();

    // OpenGL Loop
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader.Use();
    tex.Bind();

    // Vertex buffer at location=0
    glEnableVertexAttribArray(0);
    //glBindBuffer(GL_ARRAY_BUFFER, points.vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, points_cugl.vertex_buffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glDrawArrays(GL_TRIANGLES, 0, 3 * 2);

    glErrorCheck();

    //glBindTexture(GL_TEXTURE_2D, 0);
    tex.Unbind();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
}
