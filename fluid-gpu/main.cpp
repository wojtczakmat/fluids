#define GLEW_STATIC
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "GLM/glm.hpp"
#include "GLM/vec3.hpp"
#include "GLM/vec4.hpp"
#include "GLM/mat4x4.hpp"
#include "GLM/gtc/matrix_transform.hpp"
#include "GLM/gtc/quaternion.hpp"
#include "GLM/gtx/quaternion.hpp"

#include <stdlib.h>
#include <stdio.h>
#include "particles.h"
#include <time.h>

#define GLSL(src) "#version 150 core\n" #src

int width = 800;
int height = 800;

glm::quat rotation;
glm::vec4 translation = glm::vec4(0, -1, 6, 1);
glm::mat4x4 ViewMatrix;

static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

bool isQDown, isWDown, isEDown, isZDown;

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		
	}
	else if (action == GLFW_RELEASE)
	{
	
	}
}

double mouseX, mouseY;
bool isLeftMouseDown = false;

static void mouseMoveCallback(GLFWwindow *window, double x, double y)
{
	mouseX = x;
	mouseY = y;
}

static void mousePressCallback(GLFWwindow *window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		isLeftMouseDown = true;
	}
	else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
	{
	}
	else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	{
	}
	else if (action == GLFW_RELEASE)
	{
		isLeftMouseDown = false;
	}
}

// Vertex shader
const GLchar* vertexShaderSrc = GLSL(
	in vec2 pos;
	//out vec2 vDirection;

	void main()
	{
		//gl_Position = vec4(-pos.x * 2, pos.y * 2, 0.0, 1.0);
		gl_Position = vec4(pos.x / 400 - 1, pos.y / 400 - 1, 0.0, 1.0);
		//vDirection = direction;
	}
);

// Geometry shader
const GLchar* geometryShaderSrc = GLSL(
	layout(points) in;
	layout(line_strip, max_vertices = 11) out;

	const float PI = 3.1415926;

	void main()
	{
		for (int i = 0; i <= 10; i++) {
			// Angle between each side in radians
			float ang = PI * 2.0 / 10.0 * i;

			// Offset from center of point (0.3 to accomodate for aspect ratio)
			vec4 offset = vec4(cos(ang) * 0.01, -sin(ang) * 0.01, 0.0, 0.0);
			gl_Position = gl_in[0].gl_Position + offset;

			EmitVertex();

//			if (i % 2 == 1) {
//				gl_Position = gl_in[0].glPosition;
//				EmitVertex();
//				EndPrimitive();
//			}
		}

//		gl_Position = gl_in[0].gl_Position + vec4(-0.02, -0.02, 0, 0);
//		EmitVertex();
//		gl_Position = gl_in[0].gl_Position + vec4(-0.02, 0.02, 0, 0);
//		EmitVertex();
//		gl_Position = gl_in[0].gl_Position + vec4(0.02, 0.02, 0, 0);
//		EmitVertex();
//		gl_Position = gl_in[0].gl_Position + vec4(0.02, -0.02, 0, 0);
//		EmitVertex();
//		gl_Position = gl_in[0].gl_Position + vec4(-0.02, -0.02, 0, 0);
//		EmitVertex();

		//EndPrimitive();
	}
);

// Fragment shader
const GLchar* fragmentShaderSrc = GLSL(
	out vec4 outColor;

	void main()
	{
		outColor = vec4(0,0,1,1); // black
	}
);

GLuint createShader(GLenum type, const GLchar* src) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);
	return shader;
}

time_t seconds;
static int fpstracker = 0;
static float fps = 0;

void updateFps(GLFWwindow* window)
{
	time_t seconds2 = time(NULL);

	fpstracker++;
	if (seconds2 - seconds >= 1)
	{
		fps = (float)fpstracker / (seconds2 - seconds);
		fpstracker = 0;
		seconds = seconds2;

		char currentFps[128];
		sprintf(currentFps, "Current fps: %f", fps);
		glfwSetWindowTitle(window, currentFps);
	}
}

int main(int argc, char **argv)
{
	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* window = glfwCreateWindow(width, height, "SPH fluids GPU", NULL, NULL);

	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouseMoveCallback);
	glfwSetMouseButtonCallback(window, mousePressCallback);
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return 0;
	}

	glfwSwapInterval(0);

	seconds = time(NULL);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	GLuint vertexShader = createShader(GL_VERTEX_SHADER, vertexShaderSrc);
	GLuint geometryShader = createShader(GL_GEOMETRY_SHADER, geometryShaderSrc);
	GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);

	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, geometryShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glUseProgram(shaderProgram);

	int numOfParticles = 1000;
    
    if (argc >= 2) {
        numOfParticles = strtol(argv[1], NULL, 10);
    }

	// Create VBO with point coordinates
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, numOfParticles * sizeof(Particle), NULL, GL_DYNAMIC_DRAW);

	initCuda(vbo, numOfParticles);

	// Specify the layout of the vertex data
	GLint posAttrib = glGetAttribLocation(shaderProgram, "pos");
	glEnableVertexAttribArray(posAttrib);
	glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);

	while (!glfwWindowShouldClose(window))
	{
		updateFps(window);
		simulate();

		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Render frame
		glDrawArrays(GL_POINTS, 0, numOfParticles);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteProgram(shaderProgram);
	glDeleteShader(fragmentShader);
	glDeleteShader(geometryShader);
	glDeleteShader(vertexShader);

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
