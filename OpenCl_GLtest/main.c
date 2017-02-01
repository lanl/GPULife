/* LA-CC-16080

 Copyright © 2016 Priscilla Kelly and Los Alamos National Laboratory. All Rights Reserved.
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. The name of the author may not be used to endorse or promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY Priscilla Kelly and Los Alamos National Laboratory "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Priscilla Kelly <priscilla.noreen@gmail.com>
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <GL/glew.h>

#ifdef __APPLE_CC__
#include <OpenCL/OpenCL.h>
#include <OpenCL/cl_gl.h>
#else
#include <CL/cl.h>
//#include <CL/cl_gl.h>
#endif

#if defined(__APPLE_CC__)
#include <OpenGL/gl3.h>
//#include <OpenGL/OpenGL.h>
//#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
//#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include <time.h>
#define ndims = 2
//GL related variables
int window_width = 800;
int window_height = 600;
int glutWindowHandle = 0;
float translate_z = -1.f;

// Function Declarations
int **alloc2D(int row, int col);
void dealloc(int **array);
void init_gl(int argc, char** argv);

//MAIN
int main(int argc, char *argv[]) {

	int ierr;
	// Command line stuff
	int myRows = 5;
	int myCols = 5;
	int i,j;	
	int **subMatrix;
	subMatrix = alloc2D(myRows,myCols);
	printf("%s: SubMat in Main\n",__FILE__);
	for(i=0;i<myRows;i++) {
		printf("[");
		for(j=0;j<myCols;j++){
			printf(" %2d ",subMatrix[i][j]);
		}
		printf("]\n");
	}	
	int *sub1D = (int *)&(subMatrix[0][0]);	
	size_t dims = myRows*myCols;
	// this up would be given in a command line

printf("DEBUG file %s line %d\n",__FILE__,__LINE__);

        init_gl(argc, argv);
/*
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowSize(300,300);
        glutCreateWindow("OpenCL_GLtest Interoperability");
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);

        glewInit();

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);

   glDepthFunc(GL_LESS);
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);
   glShadeModel(GL_SMOOTH);
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glMatrixMode(GL_PROJECTION);
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);
   glLoadIdentity();
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);
*/

	// malloc buffer space
	GLuint devptr;
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);
	glGenBuffers(1,&devptr);
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);

	cl_platform_id platform;
	cl_device_id deviceID;
	cl_context cxt;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel_applyRulesCL;

	ierr = clGetPlatformIDs(1,&platform,NULL);
	if (ierr != CL_SUCCESS) {
		printf("ERROR: Failed to create platform!\n");
		return(-1);
	}

	char ext_string[1024];
	ierr = clGetPlatformInfo(platform,CL_PLATFORM_EXTENSIONS,sizeof(ext_string),ext_string,NULL);
	
	char *extStringStart = NULL;
        #if defined (__APPLE__) || defined(MACOSX)
	  extStringStart = strstr(ext_string,"cl_APPLE_gl_sharing");
        #else
	  extStringStart = strstr(ext_string,"cl_khr_gl_sharing");
        #endif
	if (extStringStart != 0) {
		printf("Platform supports cl_khr_gl_sharing\n");
        } else {
		printf("Platform does not support cl_khr_gl_sharing\n");
	}

	ierr = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&deviceID,NULL);
	if (ierr != CL_SUCCESS) {
		printf("ERROR: Failed to create a device group!\n");
		return(-1);
	}

        #if defined (__APPLE__) || defined(MACOSX)
          static const char* CL_GL_SHARING_EXT="cl_APPLE_gl_sharing";
        #else
          static const char* CL_GL_SHARING_EXT="cl_khr_gl_sharing"; 
        #endif

        // Get string containing supported device extensions
        size_t ext_size = 1024;
        char* dev_ext_string = (char*)malloc(ext_size);
        int err = clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, ext_size, dev_ext_string, &ext_size);
        // Search for GL support in extension string (space delimited)
        extStringStart = strstr(dev_ext_string, CL_GL_SHARING_EXT);
        if( extStringStart ) {
          // Device supports context sharing with OpenGL
          printf("Found GL Sharing Support!\n");
        } else {
          printf("No GL Sharing Support!\n");
        }
        free(dev_ext_string);


        #ifdef __APPLE__
	   // Get the properties from the current system
           CGLContextObj     kCGLContext     = CGLGetCurrentContext();
	   CGLShareGroupObj  kCGLShareGroup  = CGLGetShareGroup(kCGLContext);

	   cl_context_properties props[] =
	   {
	     CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
	     (cl_context_properties) kCGLShareGroup, 0
	   };

	#elif __linux__
	   // Get the properties from the current system
	   cl_context_properties props[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
		CL_WGL_HDC_KHR, (cl_context_properties)glXGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
	   };

	#elif _WIN32
	   // Get the properties from the current system
	   cl_context_properties props[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
	   };

	#endif

printf("DEBUG file %s line %d\n",__FILE__,__LINE__);
  	// create the OCL context using the new properties
	cxt = clCreateContext(props, 1, &deviceID, NULL, NULL, &ierr);
	if (ierr != CL_SUCCESS) {
 	   printf("ERROR: failed to create context: %d\n",ierr);
	}
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);

	commands = clCreateCommandQueue(cxt,deviceID,0,&ierr);
	if (ierr != CL_SUCCESS) {
		printf("ERROR: failed to create command queue %d\n",ierr);
	}
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);

	// malloc buffer space
	GLuint gldevptr;
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);
	glGenBuffers(1,&gldevptr);
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);
	glNamedBufferData(gldevptr,dims*sizeof(int),NULL,GL_STATIC_DRAW);
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);

printf("DEBUG file %s line %d\n",__FILE__,__LINE__);
        cl_mem clmemBuff = clCreateFromGLBuffer(cxt, CL_MEM_WRITE_ONLY,
          gldevptr, &err); 
	if (ierr != CL_SUCCESS) {
		printf("ERROR: failed to create CL buffer: %d\n",ierr);
	}
printf("DEBUG file %s line %d\n",__FILE__,__LINE__);

#ifdef XXX
	ierr = clEnqueueAcquireGLObjects(commands,1,&devptr,0,0,0);
	if (ierr != CL_SUCCESS) {
		printf("ERROR: failed to aquire: %d\n",ierr);
	}

	ierr = clEnqueueWriteBuffer(commands,devptr,CL_TRUE,0,dims*sizeof(cl_int),(void *)&sub1D[0],0,NULL,NULL);

	if (ierr != CL_SUCCESS) {
		printf("ERROR: failed to allocate: %d\n",ierr);
	}

	/*
	// Set up kernel to run
	kernel_applyRulesCL = ezcl_create_kernel_wprogram(prog,"applyRulesOCL");
	ezcl_program_release(prog);
	// set arguments
	ezcl_set_kernel_arg(kernel_applyRulesCL,0,sizeof(cl_int), (void *)&myRows);
	ezcl_set_kernel_arg(kernel_applyRulesCL,1,sizeof(cl_int), (void *)&myCols);
	ezcl_set_kernel_arg(kernel_applyRulesCL,2,sizeof(cl_mem), update);

	int mod = dims%32;
	size_t globalSize, localSize;
	if (mod == 0) { // check if the total work elements is a factor of 32 
		globalSize = dims;
	} else {
		globalSize = dims + (32-mod);
	}
	localSize = 32;


	// loops
	
	// run everything
	ezcl_enqueue_ndrange_kernel(com_queue,kernel_applyRulesCL,1,NULL,&globalSize,&localSize,NULL);
	*/
	// read data back
	/*cl_command_queue com_queue = ezcl_get_command_queue();
	int recvSub[dims];
	clEnqueueReadBuffer(com_queue,update,CL_TRUE,0,dims*sizeof(cl_int),(void *)&recvSub[0],NULL,NULL,NULL);

	for (i=0;i<dims;i++) {
	printf(" %d ",recvSub[i]);
	}
	printf("\n");
	*/
	glBindBuffer(GL_ARRAY_BUFFER,0);
	//ezcl_device_memory_remove(update);
	dealloc(subMatrix);
#endif

    exit(0);
} // end of main

/***************************************/
/* Function: alloc 2D                  */
/***************************************/

int **alloc2D(int row, int col) {
	int i, j;
	size_t mem_size;
	int **arr;

	mem_size = col*sizeof(int *);
	arr = (int **)malloc(mem_size);

	mem_size = row*col*sizeof(int);
	arr[0] = (int *)malloc(mem_size);

	for(i = 1; i< col; i++) {
		arr[i] = arr[i-1]+row;
	}
	for(i=0;i<row;i++) {
		for(j=0;j<col;j++){
			arr[i][j] = i*10+j;
		}
	}	
	return arr;
}

/***************************************/
/* Function: dealloc 2D array          */
/***************************************/
void dealloc(int **array) {
	free(array[0]);
	free(array);
}

void init_gl(int argc, char** argv)
{

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);

    
    //std::stringstream ss;
    //ss << "Adventures in OpenCL: Part 2, " << NUM_PARTICLES << " particles" << std::ends;
    glutWindowHandle = glutCreateWindow("OpenCL_GLtest Interoperability");

    //glutDisplayFunc(appRender); //main rendering function
    //glutTimerFunc(30, timerCB, 30); //determin a minimum time between frames
    //glutKeyboardFunc(appKeyboard);
    //glutMouseFunc(appMouse);
    //glutMotionFunc(appMotion);

    glewInit();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 1000.0);

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);

}

