// Simple OpenGL cube test in C to verify GLFW/GLEW/OpenGL setup
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <stdio.h>
#include <math.h>

void draw_cube(float rotation) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    
    glTranslatef(0.0f, 0.0f, -5.0f);
    glRotatef(rotation, 0.0f, 1.0f, 0.0f);
    glRotatef(rotation * 0.5f, 1.0f, 0.0f, 0.0f);
    
    glBegin(GL_QUADS);
    
    // Front face (red)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.0f, -1.0f, 1.0f);
    glVertex3f(1.0f, -1.0f, 1.0f);
    glVertex3f(1.0f, 1.0f, 1.0f);
    glVertex3f(-1.0f, 1.0f, 1.0f);
    
    // Back face (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-1.0f, -1.0f, -1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);
    glVertex3f(1.0f, 1.0f, -1.0f);
    glVertex3f(1.0f, -1.0f, -1.0f);
    
    glEnd();
}

int main() {
    if (!glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "C Test - Cube", NULL, NULL);
    if (!window) {
        printf("Failed to create window\n");
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    
    if (glewInit() != GLEW_OK) {
        printf("Failed to initialize GLEW\n");
        return 1;
    }
    
    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    printf("GLEW Version: %s\n", glewGetString(GLEW_VERSION));
    
    // Setup OpenGL
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2.0, 2.0, -2.0, 2.0, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    
    float rotation = 0.0f;
    
    while (!glfwWindowShouldClose(window)) {
        draw_cube(rotation);
        rotation += 0.8f;
        if (rotation >= 360.0f) rotation -= 360.0f;
        
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, 1);
        }
    }
    
    glfwTerminate();
    return 0;
}

