#include "ofMain.h"
#include "cvEyeTracker.h"
#include "ofAppGlutWindow.h"

//========================================================================
int main( ){
    
    ofAppGlutWindow window;
	ofSetupOpenGL(&window, 1680, 1000, OF_WINDOW);
	ofRunApp( new cvEyeTracker() );

}