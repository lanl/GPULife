#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/gl.h>

int isExtensionSupported(const char* support_str,const char* ext_string, 
						 size_t ext_buffer_size); 
void main() {
	static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
	// get the string
	int ext_size = 1024;
	char* ext_string = (char*)malloc(ext_size);
	int err;
	err = clGetDeviceInfo(0,CL_DEVICE_EXTENSIONS,ext_size,ext_string,&ext_size);
	// search for GL support
	int supported = isExtensionSupported(CL_GL_SHARING_EXT,ext_string,ext_size);
	if (supported) {
		printf("Found GL Sharing Support!!\n");
	}

}

int isExtensionSupported(const char* support_str,const char* ext_string, 
						 size_t ext_buffer_size) {

	size_t offset = 0;
	const char* space_substr = strnstr(ext_string+offset," ",ext_buffer_size-offset);
	size_t space_pos = space_substr ? space_substr - ext_string :0;
	while (space_pos < ext_buffer_size)
	{
		if(strncmp(support_str, ext_string + offset, space_pos) == 0)
		{ 
			// Device supports request extension!
			printf("Info: Found extension support ‘%s’!\n", support_str);
			return 1;
		}
		// keep searching to next token string
        offset = space_pos + 1;
		space_substr = strnstr(ext_string + offset, " ", ext_buffer_size - offset);
		space_pos = space_substr ? space_substr - ext_string : 0;
	}
	printf("Warning: Extension is not supported '%s'!\n",support_str);
	return 0;
}
