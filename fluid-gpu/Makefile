#######################################################################################################

# Mac OS X
#INCLUDE_PATH      = -I/usr/local/include/ -I/usr/local/include/eigen3/
#LIBRARY_PATH      = -L/usr/local/lib/
#OPENGL_LIBS       = -framework OpenGL -framework GLUT

# # Linux
INCLUDE_PATH      = -I/opt/cuda/include/
LIBRARY_PATH      = -L/opt/cuda/lib64/
OPENGL_LIBS       = -lglut -lGL -lX11 -lGLEW -lglfw
CUDA_LIBS         = -lcuda -lcudart

# # Windows / Cygwin
# INCLUDE_PATH      = -I/usr/include/opengl
# LIBRARY_PATH      = -L/usr/lib/w32api
# OPENGL_LIBS       = -lglut32 -lopengl32

#######################################################################################################

TARGET = particles
CC = g++
LD = g++
CFLAGS = -std=c++11 -O3 -Wall -Wno-deprecated -pedantic -Wno-vla-extension $(INCLUDE_PATH) -I./GL -I./GLM -I./GLFW -I. -DNDEBUG
LFLAGS = -std=c++11 -O3 -Wall -Wno-deprecated -Werror -pedantic $(LIBRARY_PATH) -DNDEBUG
LIBS = $(OPENGL_LIBS) $(SUITESPARSE_LIBS) $(BLAS_LIBS) $(CUDA_LIBS)

OBJS = obj/main.o obj/particles.o

default: $(TARGET)

all: clean $(TARGET)

$(TARGET): $(OBJS)
	$(LD) $(LFLAGS) $(OBJS) $(LIBS) -o $(TARGET)
	
obj/particles.o:
	nvcc -std=c++11 -c -arch=compute_50 -I. particles.cu -o obj/particles.o

obj/main.o: obj/particles.o
	$(CC) $(CFLAGS) -c main.cpp -o obj/main.o obj/particles.o

clean:
	rm -f $(OBJS)
	rm -f $(TARGET)
	rm -f $(TARGET).exe

