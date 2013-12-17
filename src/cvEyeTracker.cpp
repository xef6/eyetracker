#include "cvEyeTracker.h"

#include <opencv/cv.h>
#include <stdio.h>

#include <math.h>
#include "cv.h"
#include "highgui.h"
#include "ofxOpenCv.h"
#include <assert.h>

#define FPS_HI 60
#define FPS_LO 15
#define DBUG 0

/** 
 * @file cvEyeTracker.cpp
 * @brief this file contains the implementation of the opencv eyetracker class
 * @author John H De Witt
 *
 * @date 2012/05/12
 */

/**
 * This method is the frame_counter constructor
 * @date 2012/04/26
 */
frame_counter frame_counter_new()
{
    frame_counter a;
    a.cur = 0;
    a.tot = 0;
    return a;
}

/**
 * This method is the video_context constructor
 * @date 2012/04/26
 */
video_context video_context_new(int mode)
{
	assert( mode == CV_USE_VIDEO || mode == CV_USE_LIVE_VIDEO );
	video_context a;
	a.ready = false;
	a.mode = mode;
	return a;
}

/**
 *	This method prepares the video source and initializes the application.
 *	@date 2012/04/26
 */
void cvEyeTracker::setup(){
	
	UIMODE = CV_MODE_DEV;
	
    ofSetFrameRate(FPS_HI);
	
	cap_context = video_context_new( CV_USE_VIDEO );
//	cap_context = video_context_new( CV_USE_LIVE_VIDEO );

	cap_count = frame_counter_new();
    cap_size = cvSize(640, 480);
//    cap_size = cvSize(320, 240);
    
    bVideoReady = initVideo( &cap_context, &cap_count, &cap_size );
    assert( bVideoReady );
    
    initFrameBuffer( cap_size );
    initEnvironment( cap_size );

}

/**
 *	This method initializes the video capture structure.
 *	@date 2012/04/26
 *	@param mode		Either CV_USE_VIDEO or CV_USE_LIVE_VIDEO
 *	@param size		Receives the resulting capture resolution.
 *	@param want_w	(CV_USE_LIVE_VIDEO only) desired capture width
 *	@param want_h	(CV_USE_LIVE_VIDEO only) desired capture height
 */
bool cvEyeTracker::initVideo( video_context *vid_context,
							 frame_counter *counter,
							 CvSize *size
							 ){
	
	if(vid_context->mode==CV_USE_VIDEO){
        vid_context->vp.setSpeed(1.0);
        vid_context->vp.setUseTexture(true);
		
		// video file goes here
		//vid_context->vp.loadMovie("");

        vid_context->vp.play();
		counter->tot = vid_context->vp.getTotalNumFrames();
        if (vid_context->vp.isLoaded()) {
            *size = cvSize( vid_context->vp.width, vid_context->vp.height );
        }
		vid_context->ready = vid_context->vp.isLoaded();
        return vid_context->vp.isLoaded();
    }

	
    if(vid_context->mode==CV_USE_LIVE_VIDEO){
        vid_context->vg.setVerbose(true);
		vid_context->vg.setDeviceID(1);
		vid_context->vg.setDesiredFrameRate(60);
        vid_context->vg.setUseTexture(true);
        vid_context->vg.initGrabber( size->width, size->height );
		vid_context->ready = true;
        *size = cvSize( vid_context->vg.width, vid_context->vg.height );
        return true;
	}
	
    return false;
}

fLUT cvEyeTracker::newfLUT(int n){
//	fLUT *ret = (fLUT*) malloc( sizeof(fLUT) );
	fLUT ret;
	ret.n = MAX(n,1);
	ret.v = (CvPoint2D32f*) malloc( sizeof(CvPoint2D32f) * ret.n );
	return ret;
}

float cvEyeTracker::queryLUT( fLUT *qlut, float qv ){
	bool hit;
	// iterate over lookup values
	float x0,x1,y0,y1;
	for(int i=0;i<qlut->n-1;i++){
		hit = false;
		x0 = qlut->v[i].x;
		x1 = qlut->v[i+1].x;
		y0 = qlut->v[i].y;
		y1 = qlut->v[i+1].y;
		if( i==0         && qv < x0 ){ return qlut->v[0].y; } // before
		if( i==qlut->n-2 && qv > x1 ){ return qlut->v[qlut->n-1].y; } // after
		if( x0 < qv && qv < x1      ){ hit=true; } // inside
		if( hit ){
			return min(max( y0 + ((qv-x0)*y1-(qv-x0)*y0)/(x1-x0), 0.0f ),1.0f);
		}
	}
	return 0.0f;
}

/**
 *	This method sets the default values of general variables.
 *	@date 2012/05/07
 */
void cvEyeTracker::initEnvironment( CvSize the_size ){
	
	
	myfont0.loadFont("Inconsolata.otf", 10, true);

	last_match = cvPoint(-1, -1);
	
	pupil_bbox = cvRect(0, 0, cap_size.width, cap_size.height);
	
	// boolean toggles
	bUseEyeMouse		= false;
	bToggleRec			= false;
	bToggleRecRaw		= true;
	bNewFrame			= false;
	bNeedRefresh		= false;
	bUseStep			= false;
	bFlipOrientation	= false;
	bMapReady			= false;
	bUseLUT				= false;
	bLUTLoaded			= false;
	bPrintDev			= false;
	
	bDrawEyeImage		= true;
	bDrawDevScreen		= true;
	bDrawCalibrationTargetPoints = true;
	bDrawCalibrationSourcePoints = false;
	bDrawScreenGuessPoint		= true;
	bDrawScreenGazePoint		= true;
	bDrawPerkinjePoints			= true;
	bDrawPupilEllipse			= true;
	
	// histogram stuff
	int bins = 256;
	float range[] = {0,255};
	float *ranges[] = {range};
	
	cur_roi = cvRect(0, 0, the_size.width, the_size.height);
	
	hist_cur_sum = 0;
	hist_pupil_sum = 0;
	hist_iris_sum = 0;
	
	hist_cur_src = cvCreateHist(1, &bins, CV_HIST_ARRAY);
	hist_roi_src = cvCreateHist(1, &bins, CV_HIST_ARRAY);
	hist_thresh_src = cvCreateHist(1, &bins, CV_HIST_ARRAY);
	
	for(int i=0;i<256;i++){
		hist_cur[i] = -1;
		hist_avg[i] = -1;
		hist_roi[i] = -1;
		hist_roi_avg[i] = -1;
		hist_pupil[i] = -1;
		hist_pupil_avg[i] = -1;
		hist_iris[i] = -1;
		hist_iris_avg[i] = -1;
	}
		
	// caluclated values
	h_cur_peak_val = 0;
	h_cur_peak_idx = 15;
	h_avg_peak_val = 0;
	h_avg_peak_idx = 15;
	h_cur_min_val = 0;
	h_cur_min_idx = 5;
	h_avg_min_val = 0;
	h_avg_min_idx = 5;
	
	// interpolation/smoothing modulation stuff
	
	for(int i=0;i<I_INTERP_N;i++){
		int new_size = 1;
		if(i==I_INTERP_SCREEN) new_size = 5;
		if(i==I_INTERP_HIST  ) new_size = 6;
		interp_arr[i] = newfLUT(new_size);
	}
	
	fLUT *tmp;
	tmp = &interp_arr[I_INTERP_SCREEN];
	if(0){
		tmp->v[0] = cvPoint2D32f(  0, 0.80);
		tmp->v[1] = cvPoint2D32f( 25, 0.70);
		tmp->v[2] = cvPoint2D32f( 50, 0.55);
		tmp->v[3] = cvPoint2D32f(150, 0.15);
		tmp->v[4] = cvPoint2D32f(300, 0.00);
	} else {
		
		tmp->v[0] = cvPoint2D32f(  0, 0.95);
		tmp->v[1] = cvPoint2D32f( 25, 0.90);
		tmp->v[2] = cvPoint2D32f( 50, 0.85);
		tmp->v[3] = cvPoint2D32f(150, 0.35);
		tmp->v[4] = cvPoint2D32f(300, 0.00);
	}
	tmp = &interp_arr[I_INTERP_HIST];
	tmp->v[0] = cvPoint2D32f(  0, 0.96);
	tmp->v[1] = cvPoint2D32f(  2, 0.80);
	tmp->v[2] = cvPoint2D32f(  5, 0.30);
	tmp->v[3] = cvPoint2D32f( 10, 0.80);//cutoff after
	tmp->v[4] = cvPoint2D32f( 20, 0.98);
	tmp->v[5] = cvPoint2D32f( 30, 0.999);
	
	box_ellipse_avg = CvBox2D();
	
	for(int i=0;i<G_STOR_MAX;i++){
		g_storage[i] = cvCreateMemStorage(0);
	}
	
	// persistent
	perkinje_src_pts = cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32FC2, sizeof(CvContour), sizeof(CvPoint2D32f), g_storage[0]);
	perkinje_dst_pts = cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32FC2, sizeof(CvContour), sizeof(CvPoint2D32f), g_storage[2]);
	
	cvClearSeq(perkinje_src_pts);
	cvClearSeq(perkinje_dst_pts);
	
	// generate beginning perkinje points
	for (int i=0; i<4; i++) {
		CvPoint2D32f ins_src_pt = cvPoint2D32f(0, 0);
		CvPoint2D32f ins_dst_pt = cvPoint2D32f(0, 0);
		int xdiff = 0;
		int bord = 0;
		if(i==0) ins_dst_pt = cvPoint2D32f(ofGetScreenWidth()-bord,	bord);
		if(i==1) ins_dst_pt = cvPoint2D32f(bord,				    bord);
		if(i==2) ins_dst_pt = cvPoint2D32f(bord,				    ofGetScreenHeight()-bord);
		if(i==3) ins_dst_pt = cvPoint2D32f(ofGetScreenWidth()-bord,	ofGetScreenHeight()-bord);
		cvSeqPush(perkinje_src_pts, &ins_src_pt);
		cvSeqPush(perkinje_dst_pts, &ins_dst_pt);
	}
	
	// temporary
	pk_tmp		= cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32FC2,
							  sizeof(CvContour), sizeof(CvPoint2D32f), g_storage[3]);
	fin_contour	= cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2,
							  sizeof(CvContour), sizeof(CvPoint), g_storage[4]);
	seq_hull	= cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2,
							  sizeof(CvContour), sizeof(CvPoint), g_storage[5]);	
	
	cvClearSeq(pk_tmp);
	cvClearSeq(fin_contour);
//	cvClearSeq(seq_hull);
	
	step_rdy  = 0;
	threshold = 80;
	
	// initialize ellipse model
	box_ellipse_avg.center.x = 0;
	box_ellipse_avg.center.y = 0;
	box_ellipse_avg.size.width	= 1;
	box_ellipse_avg.size.height	= 1;
	box_ellipse_avg.angle		= 0;
	
	pupil_fit_cur.center = cvPoint2D32f(cap_size.width*0.5, cap_size.height*0.5);
	pupil_fit_cur.size   = cvSize2D32f(35, 35);
	pupil_fit_avg.center = cvPoint2D32f(-1, -1);
	pupil_fit_avg.size   = cvSize2D32f(35, 35);
	
	// mapping stuff
	
	guess_pt = cvPoint2D32f(0, 0);
	guess_pt_avg = cvPoint2D32f(0, 0);
	screen_pt = cvPoint2D32f(0, 0); 
	screen_pt_avg = cvPoint2D32f(0, 0); 
	
	
	persp_mat		= cvCreateMat(3, 3, CV_32FC1);
	
	persp_src_pt	= cvCreateMat(3, 1, CV_32FC1);
	persp_dst_pt	= cvCreateMat(3, 1, CV_32FC1);
	
	persp_src_3ch	= cvCreateMat(1, 1, CV_32FC3 );
	persp_dst_3ch	= cvCreateMat(1, 1, CV_32FC3 );

	
	h_manual_offset    = 10;
	h_manual_threshold = 10;
	derp_pt = cvPoint2D32f(the_size.width*0.5,the_size.height*0.5);
	center_pt = cvPoint2D32f(the_size.width*0.5,the_size.height*0.5);
	pupil_current = cvPoint2D32f(the_size.width*0.5,the_size.height*0.5);
	
	initCalibrationMap(&calibration_map, 3,3);
	
	
}

/**
 *	This method initializes framebuffer related variables.
 *	@date 2012/05/07
 */
void cvEyeTracker::initFrameBuffer( CvSize the_size ){
	
	// allocate OpenCV IplImage "registers" for various tasks
	// default to grayscale
	for( int i=0; i<I_ARR_N; i++ ){
		int arr_w = the_size.width;
		int arr_h = the_size.height;
		int arr_d = IPL_DEPTH_8U;
		int arr_c = 1;
		
		if( i==I_GFILLMASK){ arr_w+=2; arr_h+=2; }
		if( i==I_CHIST_0 ) { arr_w = 256; arr_c = 3; }
		if( i==I_CHIST_1 ) { arr_w = 256; arr_c = 3; }
		if( i==I_CHIST_2 ) { arr_w = 256; arr_c = 3; }
		if( i==I_GDEBUG  ) { arr_w = ofGetWidth(); arr_h = ofGetHeight(); arr_c = 1; }
		if( i==I_CDEBUG  ) { arr_w = ofGetWidth(); arr_h = ofGetHeight(); arr_c = 3; }
		if( i==I_GMASK   ) { arr_w += 2; arr_h += 2; }
		
		if( i==I_CTMP_0  ) { arr_c = 3; }
		if( i==I_CTMP_1  ) { arr_c = 3; }
		if( i==I_CSRC	 ) { arr_c = 3; }
		if( i==I_CREC	 ) { arr_c = 3; }
//		if( i==I_GSOBEL  ) { arr_d = IPL_DEPTH_16S; }
		if( i==I_GACC    ) { arr_d = IPL_DEPTH_32F; }
        
		ipl_arr[i] = cvCreateImage(cvSize(arr_w,arr_h), arr_d, arr_c);
		cvSetZero(ipl_arr[i]);
	}
	
	// of_ and ofx_ prefix : openframeworks display use
	// these are for displaying images through openframeworks
	
	of_img0.allocate(cap_size.width,cap_size.height, OF_IMAGE_COLOR);
	ofx_hist0.allocate(ipl_arr[I_CHIST_0]->width, ipl_arr[I_CHIST_0]->height);
	ofx_hist1.allocate(ipl_arr[I_CHIST_1]->width, ipl_arr[I_CHIST_1]->height);
	ofx_hist2.allocate(ipl_arr[I_CHIST_2]->width, ipl_arr[I_CHIST_2]->height);
	ofx_color_src.allocate(cap_size.width, cap_size.height);
	
    // ------
    of_gray_src.allocate(cap_size.width, cap_size.height, OF_IMAGE_GRAYSCALE);
    of_gray_out0.allocate(cap_size.width, cap_size.height, OF_IMAGE_GRAYSCALE);
    of_gray_out1.allocate(cap_size.width, cap_size.height, OF_IMAGE_GRAYSCALE);
    of_gray_out2.allocate(cap_size.width, cap_size.height, OF_IMAGE_GRAYSCALE);
    // ------
    
	ofx_gray_src.allocate(cap_size.width, cap_size.height);
	
	ofx_gray_out0.allocate(cap_size.width, cap_size.height);
	ofx_gray_out1.allocate(cap_size.width, cap_size.height);
	ofx_gray_out2.allocate(cap_size.width, cap_size.height);
	
	ofx_color_out0.allocate(cap_size.width, cap_size.height);
	ofx_color_out1.allocate(cap_size.width, cap_size.height);
	ofx_color_out2.allocate(cap_size.width, cap_size.height);
}

void cvEyeTracker::calcCalibrationMap( calibration_datum *new_map ){
	//	if(bMapReady){
	CvPoint2D32f src_pts[4];
	CvPoint2D32f dst_pts[4];
	int w = new_map->cols;
	int h = new_map->rows;
	int src_step = new_map->src_mat->step;
	int dst_step = new_map->dst_mat->step;
	float* src_ptr0 = NULL;
	float* dst_ptr0 = NULL;
	float* src_ptr1 = NULL;
	float* dst_ptr1 = NULL;
	for( int r=0; r<h-1; r++ ){
		src_ptr0 = (float*)(new_map->src_mat->data.ptr +  r    * src_step);
		src_ptr1 = (float*)(new_map->src_mat->data.ptr + (r+1) * src_step);
		dst_ptr0 = (float*)(new_map->dst_mat->data.ptr +  r    * src_step);
		dst_ptr1 = (float*)(new_map->dst_mat->data.ptr + (r+1) * src_step);
		
		for( int c=0; c<w-1; c++ ){
			
			src_pts[0] = cvPoint2D32f(src_ptr0[2*(c+0)], src_ptr0[2*(c+0)+1]);
			src_pts[1] = cvPoint2D32f(src_ptr0[2*(c+1)], src_ptr0[2*(c+1)+1]);
			src_pts[2] = cvPoint2D32f(src_ptr1[2*(c+1)], src_ptr1[2*(c+1)+1]);
			src_pts[3] = cvPoint2D32f(src_ptr1[2*(c+0)], src_ptr1[2*(c+0)+1]);
			
			dst_pts[0] = cvPoint2D32f(dst_ptr0[2*(c+0)], dst_ptr0[2*(c+0)+1]);
			dst_pts[1] = cvPoint2D32f(dst_ptr0[2*(c+1)], dst_ptr0[2*(c+1)+1]);
			dst_pts[2] = cvPoint2D32f(dst_ptr1[2*(c+1)], dst_ptr1[2*(c+1)+1]);
			dst_pts[3] = cvPoint2D32f(dst_ptr1[2*(c+0)], dst_ptr1[2*(c+0)+1]);
			
			cvGetPerspectiveTransform(src_pts, dst_pts, new_map->mat[xy2idx(c, r, w-1)]);
		}
	}
	printf("recalculated calibration map!\n");
	
}

void cvEyeTracker::initCalibrationMap(calibration_datum *new_map, int _r, int _c){
	
	new_map->selected = cvPoint(0, 0);
	
	new_map->rows = _r;
	new_map->cols = _c;
	
	new_map->mat = (CvMat**)malloc((new_map->rows-1)*(new_map->cols-1)*sizeof(CvMat));

	new_map->src_mat = cvCreateMat(new_map->rows, new_map->cols, CV_32FC2);
	new_map->dst_mat = cvCreateMat(new_map->rows, new_map->cols, CV_32FC2);
	
	cvSetZero(new_map->src_mat);
	cvSetZero(new_map->dst_mat);
	
	int mapw = new_map->cols-1;
	for(int row=0;row<new_map->rows-1;row++){
		for(int col=0;col<new_map->cols-1;col++){
			new_map->mat[row*mapw+col] = cvCreateMat(3, 3, CV_32FC1);
			cvSetZero(new_map->mat[row*mapw+col]);
		}
	}
	
	printf("using calibration grid: %d x %d\n", new_map->rows, new_map->cols);
	
	// begin filling values
	CvPoint src_bord = cvPoint(25, 25);
	CvPoint dst_bord = cvPoint(25, 25);
	CvPoint cal_src_origin = cvPoint(src_bord.x, src_bord.y);
	CvPoint cal_dst_origin = cvPoint(dst_bord.x, dst_bord.y);
	
	cal_src_origin.x += 150;
	cal_src_origin.y += 50;
	
	CvPoint2D32f cal_src_step;
	cal_src_step.x = (ofGetScreenWidth()-src_bord.x*2)/(new_map->cols-1);
	cal_src_step.y = (ofGetScreenHeight()-src_bord.y*2)/(new_map->rows-1);
		
	CvPoint2D32f cal_dst_step;
	cal_dst_step.x = (ofGetScreenWidth()-dst_bord.x*2)/(new_map->cols-1);
	cal_dst_step.y = (ofGetScreenHeight()-dst_bord.y*2)/(new_map->rows-1);
	
	// if(DBUG) printf("src: %f %f\n", cal_src_step.x, cal_src_step.y );
	// if(DBUG) printf("dst: %f %f\n", cal_dst_step.x, cal_dst_step.y );
	
	float* src_ptr = NULL;
	float* dst_ptr = NULL;
	for(int row=0;row<new_map->rows;row++){
		src_ptr = (float*)(new_map->src_mat->data.ptr + row * new_map->src_mat->step);
		dst_ptr = (float*)(new_map->dst_mat->data.ptr + row * new_map->dst_mat->step);
		
		for(int col=0;col<new_map->cols;col++){
			src_ptr[col*2+0] = cal_src_origin.x+col*cal_src_step.x;
			src_ptr[col*2+1] = cal_src_origin.y+row*cal_src_step.y;
			
			dst_ptr[col*2+0] = cal_dst_origin.x+col*cal_dst_step.x;
			dst_ptr[col*2+1] = cal_dst_origin.y+row*cal_dst_step.y;
		}
	}
	
//	cvSave("dst_pts.yml", new_map->dst_mat);
	
	bMapReady = true;
		
}

int cvEyeTracker::xy2idx( int x, int y, int w ){
	return y*w+x;
}

int cvEyeTracker::checkPoly(CvPoint2D32f P, CvPoint2D32f *V, int n){
	// cn_PnPoly(): crossing number test for a point in a polygon
	//      Input:   P = a point,
	//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
	//      Return:  0 = outside, 1 = inside
	// This code is patterned after [Franklin, 2000]
	int    cn = 0;    // the crossing number counter
	
	// loop through all edges of the polygon
	for (int i=0; i<n; i++) {    // edge from V[i] to V[i+1]
		if (((V[i].y <= P.y) && (V[i+1].y > P.y))    // an upward crossing
			|| ((V[i].y > P.y) && (V[i+1].y <= P.y))) { // a downward crossing
			// compute the actual edge-ray intersect x-coordinate
			float vt = (float)(P.y - V[i].y) / (V[i+1].y - V[i].y);
			if (P.x < V[i].x + vt * (V[i+1].x - V[i].x)) // P.x < intersect
				++cn;   // a valid crossing of y=P.y right of P.x
		}
	}
	return (cn&1);    // 0 if even (out), and 1 if odd (in)
}

void cvEyeTracker::whichMat( CvPoint2D32f query_pt, calibration_datum D, CvPoint* S ){
	CvPoint2D32f tmp_pts[5];
	float* src_ptr0 = NULL;
	float* src_ptr1 = NULL;
	for( int row=0; row<D.rows-1; row++ ){
		
		src_ptr0 = (float*)(D.src_mat->data.ptr +  row    * D.src_mat->step);
		src_ptr1 = (float*)(D.src_mat->data.ptr + (row+1) * D.src_mat->step);
		
		for( int col=0; col<D.cols-1; col++ ){
			
			tmp_pts[0] = cvPoint2D32f(src_ptr0[2*(col+0)], src_ptr0[2*(col+0)+1]);
			tmp_pts[1] = cvPoint2D32f(src_ptr0[2*(col+1)], src_ptr0[2*(col+1)+1]);
			tmp_pts[2] = cvPoint2D32f(src_ptr1[2*(col+1)], src_ptr1[2*(col+1)+1]);
			tmp_pts[3] = cvPoint2D32f(src_ptr1[2*(col+0)], src_ptr1[2*(col+0)+1]);
			
			tmp_pts[4] = cvPoint2D32f(src_ptr0[2*(col+0)], src_ptr0[2*(col+0)+1]);
			
			if( checkPoly(query_pt, tmp_pts, 5) ){
//				printf("found (%4.1f,%4.1f) @ %3d,%3d\n", query_pt.x, query_pt.y, col, row);
				S->x = col;
				S->y = row;
				return;
			}
		}
	}
}


/**
 *	This method takes "LED" space points and transforms them to screen space.
 *	The current calibration map is implicitly used.
 *	This is essentially a piecewise homographic mapping.
 */
void cvEyeTracker::queryCalibrationMap(CvPoint2D32f query_pt, CvPoint2D32f *res_pt){
	
	if( bMapReady ){
		
		CvPoint2D32f tmp_pts[5];
//		int w = calibration_map.cols;
//		int res;
//		query_pt = cvPoint2D32f(ofGetMouseX(),//+ofGetWindowPositionX(),
//								ofGetMouseY());//+ofGetWindowPositionY());
		
		whichMat(query_pt, calibration_map, &last_match);
		
		if(DBUG) printf("FOUND IT SIRE: %d, %d\n", last_match.x, last_match.y);

		if( last_match.x < 0 || last_match.y < 0 ){
			return;
		} else {
			cvSetZero(persp_src_3ch);
			cvSetZero(persp_dst_3ch);
			cvSetZero(persp_dst_pt);
			cvSetZero(persp_src_pt);
			cvmSet(persp_src_pt, 0, 0, query_pt.x);
			cvmSet(persp_src_pt, 1, 0, query_pt.y);
			cvmSet(persp_src_pt, 2, 0, 1);
			cvReshape(persp_src_pt, persp_src_3ch, 3, 1);
			
			int matidx = last_match.y * (calibration_map.cols-1) + last_match.x;
			
			cvTransform( persp_src_3ch, persp_dst_3ch,
						calibration_map.mat[matidx] );
			
			cvReshape(persp_dst_3ch, persp_dst_pt, 1, 3 );
			res_pt->x = cvmGet(persp_dst_pt, 0, 0)/cvmGet(persp_dst_pt, 2, 0);
			res_pt->y = cvmGet(persp_dst_pt, 1, 0)/cvmGet(persp_dst_pt, 2, 0);
		}
	}	
}


/**
 *	This method is OpenFrameworks' update tick function.
 *	@date 2012/05/07
 */
void cvEyeTracker::update(){
	
    bNewFrame = getFrame(ipl_arr[I_CSRC]);
	
	// execute if there's a new frame or a new change
	if( bVideoReady && ( bNewFrame || bNeedRefresh ) ) {	
		
		// copy image from I_CSRC to I_GSRC and rotate if needed
		uchar* img_color = (uchar*)(ipl_arr[I_CSRC]->imageData);
		uchar* img_gray  = (uchar*)(ipl_arr[I_GSRC]->imageData);
		uchar dst_v;
		for(int r=0;r<cap_size.height;r++){
			for(int c=0;c<cap_size.width;c++){
				if(bFlipOrientation){
					dst_v = (img_color[(cap_size.height-r-1)*ipl_arr[I_CSRC]->widthStep+(cap_size.width-c-1)*3]+
							 img_color[(cap_size.height-r-1)*ipl_arr[I_CSRC]->widthStep+(cap_size.width-c-1)*3+1]+
							 img_color[(cap_size.height-r-1)*ipl_arr[I_CSRC]->widthStep+(cap_size.width-c-1)*3+2])/3.0;					
				} else {
					dst_v = (img_color[r*ipl_arr[I_CSRC]->widthStep+c*3]+
							 img_color[r*ipl_arr[I_CSRC]->widthStep+c*3+1]+
							 img_color[r*ipl_arr[I_CSRC]->widthStep+c*3+2])/3.0;
				}
				img_gray[r*ipl_arr[I_GSRC]->widthStep+c] = MIN(MAX(dst_v,0),255);
			}
		}
		
		// calculate whole image histogram
		calcHistogramROI( ipl_arr[I_GSRC], hist_cur_src, cvRect(0, 0, 1000, 1000) );
		for(int i=0;i<256;i++){
			hist_cur[i] = cvQueryHistValue_1D(hist_cur_src, i);
		}
		
		// initializate pupil/iris histograms if needed
		for(int i=0;i<256;i++){
			if(hist_pupil[i]<0){
				hist_pupil[i] = hist_cur[i];
				hist_pupil_avg[i] = hist_pupil[i];
			}
			if(hist_iris[i]<0){
				hist_iris[i] = 0;
				hist_iris_avg[i] = hist_iris[i];
			}
		}
		
		// calculate threshold values
		calcHistPeak( hist_pupil_avg, hist_iris_avg, &h_cur_peak_idx, &h_cur_min_idx );
		
		// smooth based on distance to new peak value
		// negative when new value is less than
		float peak_diff = h_cur_peak_idx - h_avg_peak_idx;
		float peak_sm = queryLUT(&interp_arr[I_INTERP_HIST], peak_diff);
		
		if(DBUG) printf("hist sm: %1.5f from %3.1f\n", peak_sm, peak_diff );
		if( peak_diff < -35 ) peak_sm = 0.25;
		if( h_avg_peak_idx > 40 && h_cur_peak_idx < 20 ) peak_sm = 0.25;
		
		h_avg_peak_idx = h_avg_peak_idx*peak_sm + (1-peak_sm)*(h_cur_peak_idx);
		h_avg_min_idx  = h_avg_min_idx *peak_sm + (1-peak_sm)*(h_cur_min_idx);
		
		// BUSINESS HAPPENS HERE
		bool pfound = findPupil(ipl_arr[I_GSRC],
								(int)h_avg_peak_idx,
								(int)h_avg_min_idx,
								&pupil_fit_cur,
								&pupil_bbox);
		
		if( pfound == false ){
			float sm0 = 0.75;
			float sm1 = 0.75;
			for(int i=0;i<256;i++){
				if( i < 50 ){
					hist_pupil_avg[i] = hist_pupil_avg[i]*sm0 + (1-sm0)*hist_cur[i];
				} else {
					hist_pupil_avg[i] = hist_pupil_avg[i]*sm1 + (1-sm1)*hist_cur[i];
				}
//				hist_iris_avg[i]  = hist_iris_avg[i] *sm + (1-sm)*hist_iris[i];
			}
		}
		
		ofx_gray_out1 = ipl_arr[I_GTHRESH_0];
		//		ofx_gray_out2 = ipl_arr[I_GTHRESH_1];
		
		
		// calculate a histogram, masked to only include the pupil
		// used to more reliably find the pupil
		
		if( pfound == false ){
			cvSetZero(ipl_arr[I_GTHRESH_0]);
			cvCircle(ipl_arr[I_GTHRESH_0], cvPoint(cap_size.width*0.5, cap_size.height*0.5),
					 cap_size.height, cvScalarAll(255));
		} else {
			
			calcHistogramMask(ipl_arr[I_GSRC], hist_roi_src, ipl_arr[I_GTHRESH_1]);
			for(int i=0;i<256;i++){
				hist_pupil[i] = cvQueryHistValue_1D(hist_roi_src, i);
			}
			
			cvSetZero(ipl_arr[I_GTHRESH_1]);
			
			int bord = 25;
			CvRect iris_box = CvRect(pupil_bbox);
			iris_box.x = MIN(MAX( iris_box.x-bord, 0),cap_size.width-1);
			iris_box.y = MIN(MAX( iris_box.y-bord, 0),cap_size.height-1);
			iris_box.width = MIN(MAX( iris_box.width+bord*2, 0),cap_size.width-iris_box.x-1);
			iris_box.height = MIN(MAX( iris_box.height+bord*2, 0),cap_size.height-iris_box.y-1);
			cvRectangle(ipl_arr[I_GTHRESH_1],
						cvPoint(iris_box.x, iris_box.y),
						cvPoint(iris_box.x+iris_box.width,
								iris_box.y+iris_box.height),
						cvScalarAll(255), CV_FILLED, 8, 0);
			
			cvSet(ipl_arr[I_GTHRESH_1], cvScalarAll(0), ipl_arr[I_GTHRESH_0]);
			
			iris_box.x += 1;
			iris_box.y += 1;
			iris_box.width  -= 2;
			iris_box.height -= 2;
			
			cvSetImageROI(ipl_arr[I_GTHRESH_1], iris_box);
			cvErode(ipl_arr[I_GTHRESH_1], ipl_arr[I_GTHRESH_1]);
			cvResetImageROI(ipl_arr[I_GTHRESH_1]);
			
			// ofx_gray_out2 = ipl_arr[I_GTHRESH_1];
			
			calcHistogramMask(ipl_arr[I_GSRC], hist_roi_src, ipl_arr[I_GTHRESH_1]);
			for(int i=0;i<256;i++){
				hist_iris[i] = cvQueryHistValue_1D(hist_roi_src, i);
			}
			
			float ecc = pupil_fit_cur.size.width / pupil_fit_cur.size.height;
			if( ecc > 1.0 ) ecc = 1.0 / ecc;
			ecc = min(max( ecc, 0.0f),1.0f);
			int ell_x = MIN(MAX( pupil_fit_cur.center.x, 0),cap_size.width-1);
			int ell_y = MIN(MAX( pupil_fit_cur.center.y, 0),cap_size.height-1);
			
			ecc = pow( ecc, 4.0f );
			
			// cvCircle(ipl_arr[I_CTMP_1], cvPoint(ell_x, ell_y), 1, cvScalar((1-ecc)*255, ecc*255, (1-ecc)*255), -1, 8, 0);
			if(1){	
				uchar* img_data = (uchar*)ipl_arr[I_CTMP_1]->imageData;
				int img_idx = (ipl_arr[I_CTMP_1]->widthStep)*ell_y+ell_x*3;
				img_data[img_idx]   = ecc*255;
				img_data[img_idx+1] = ecc*255;
				img_data[img_idx+2] = ecc*255;
			}
			// ofx_color_out1.setFromPixels( (uchar*)ipl_arr[I_CTMP_1]->imageData, cap_size.width, cap_size.height);

		}
		
		float hist_sm_src = 0.8;
		float hist_bias = 0.05;
		float hist_sm;
		if(true){
			int smooth_w = 7;
			float smooth_v = 0;
			for(int i=0;i<smooth_w;i++){
				smooth_v += hist_pupil[i];
			}
			smooth_v /= smooth_w;
			for(int i=0;i<smooth_w;i++){
				float sm = pow( 1-(i/(float)smooth_w), 0.25f );
				if( pfound ){
					hist_pupil[i] = smooth_v*sm + (1-sm)*hist_pupil[i];
				} else {
					hist_pupil_avg[i] = smooth_v*sm + (1-sm)*hist_pupil_avg[i];
				}
			}
		}
		
		if( pfound ){
			float sm = 0.85;
			for(int i=0;i<256;i++){
				hist_pupil_avg[i] = hist_pupil_avg[i]*sm + (1-sm)*hist_pupil[i];
				hist_iris_avg[i]  = hist_iris_avg[i] *sm + (1-sm)*hist_iris[i];
			}
		}
		
//		printf("perkinje start ------\n");
		bool pkfound = findPerkinje(ipl_arr[I_GSRC], pupil_fit_cur.center, perkinje_src_pts);
//		printf("perkinje end   ------\n");
		
//		ofx_gray_out1 = ipl_arr[I_GTHRESH_0];

		
		if( pkfound ){
			if(DBUG) printf("calculating transformation\n");
			calcPerspectiveMap(perkinje_src_pts, perkinje_dst_pts, persp_mat);
			
			cvSetZero(persp_src_pt);
			cvmSet(persp_src_pt, 0, 0, pupil_fit_cur.center.x);
			cvmSet(persp_src_pt, 1, 0, pupil_fit_cur.center.y);
			cvmSet(persp_src_pt, 2, 0, 1);
			cvReshape(persp_src_pt, persp_src_3ch, 3, 1);
			cvTransform( persp_src_3ch, persp_dst_3ch, persp_mat );
			cvReshape(persp_dst_3ch, persp_dst_pt, 1, 3 );
			
			guess_pt.x = cvmGet(persp_dst_pt, 0, 0)/cvmGet(persp_dst_pt, 2, 0);
			guess_pt.y = cvmGet(persp_dst_pt, 1, 0)/cvmGet(persp_dst_pt, 2, 0);
			
			float dist_to_new = sqrt( pow(guess_pt_avg.x-guess_pt.x,2) + pow(guess_pt_avg.y-guess_pt.y,2) ); 
			if(DBUG) printf(" dst = %f\n", dist_to_new );
			float sm = queryLUT( &interp_arr[I_INTERP_SCREEN], dist_to_new);			
			if(DBUG) printf(" sm  = %f\n", sm );
			
			guess_pt_avg.x = guess_pt_avg.x*sm + (1-sm)*guess_pt.x;
			guess_pt_avg.y = guess_pt_avg.y*sm + (1-sm)*guess_pt.y;
						
			CvPoint2D32f scr_pt = cvPoint2D32f(0, 0);
			queryCalibrationMap(guess_pt_avg, &screen_pt);
			
			if( bUseEyeMouse ){
//				CGWarpMouseCursorPosition(CGPointMake(ofGetWindowPositionX()+screen_pt.x,ofGetWindowPositionY()+screen_pt.y));
				CGWarpMouseCursorPosition(CGPointMake(screen_pt.x,screen_pt.y));
			}
			
			
//			screen_pt.x = scr_pt.x;
//			screen_pt.y = scr_pt.y;
			
			
			if(DBUG) printf("   REALLY? : %4.1f, %4.1f\n", scr_pt.x, scr_pt.y );
			
//			printf("dest: %4.1f, %4.1f, %4.1f\n",
//				   cvmGet(persp_dst_pt, 0, 0),
//				   cvmGet(persp_dst_pt, 1, 0),
//				   cvmGet(persp_dst_pt, 2, 0));
			
//			cvCircle(ipl_arr[I_CTMP_0], cvPoint(guess_pt.x, guess_pt.y), 64, CV_RGB(0, 255, 255));
		}
		
		// draw fit ellipse to color image
		cvCvtColor(ipl_arr[I_GSRC], ipl_arr[I_CTMP_0], CV_GRAY2BGR);
//		cvSetZero(ipl_arr[I_CTMP_0]);
		
		if( bDrawPupilEllipse ){
			cvEllipse(ipl_arr[I_CTMP_0],
					  cvPoint(pupil_fit_cur.center.x, pupil_fit_cur.center.y),
					  cvSize(pupil_fit_cur.size.width*0.5, pupil_fit_cur.size.height*0.5),
					  pupil_fit_cur.angle, 0, 360.0, CV_RGB(255,255,255));
			
			cvCircle(ipl_arr[I_CTMP_0],
					 cvPoint(pupil_fit_cur.center.x, pupil_fit_cur.center.y),
					 4, CV_RGB(255,255,255));
			
			float p_ang = pupil_fit_cur.angle;
			
//			if(p_ang < 90) p_ang +=90;
			p_ang *= 3.14159/180.0;
			cvLine(ipl_arr[I_CTMP_0],
				   cvPoint(pupil_fit_cur.center.x, pupil_fit_cur.center.y),
				   cvPoint(pupil_fit_cur.center.x+pupil_fit_cur.size.width*0.5*cos(p_ang),
						   pupil_fit_cur.center.y-pupil_fit_cur.size.height*0.5*sin(p_ang)),
				   CV_RGB(255, 255, 255));
			
		}
		
		if( bDrawPerkinjePoints ){
			CvPoint2D32f *tmp_pt;
			for(int i=0;i<4;i++){
				tmp_pt = (CvPoint2D32f*)cvGetSeqElem(perkinje_src_pts, i);
				CvScalar the_color = CV_RGB(255, 255, 255);
				if(i==0) the_color = CV_RGB(255,   0,   0);
				if(i==1) the_color = CV_RGB(  0, 255,   0);
				if(i==2) the_color = CV_RGB(  0,   0, 255);
				if(i==3) the_color = CV_RGB(255, 255,   0);
				cvCircle(ipl_arr[I_CTMP_0], cvPoint(tmp_pt->x, tmp_pt->y), 8, the_color, 1, CV_AA, 0 );
			}
		}
		
		ofx_color_out0.setFromPixels( (uchar*)ipl_arr[I_CTMP_0]->imageData, cap_size.width, cap_size.height);
		
		
		
		// histogram stuff
		drawHistogram( hist_cur, ipl_arr[I_CHIST_0]);
		drawHistogram( hist_pupil_avg, ipl_arr[I_CHIST_1]);
		drawHistogram( hist_iris_avg, ipl_arr[I_CHIST_2]);
		
		cvLine(ipl_arr[I_CHIST_0],
			   cvPoint(h_avg_peak_idx, 0),
			   cvPoint(h_avg_peak_idx, cap_size.height), cvScalar(255,0,0));
		cvLine(ipl_arr[I_CHIST_0],
			   cvPoint(h_avg_peak_idx+h_avg_min_idx, 0),
			   cvPoint(h_avg_peak_idx+h_avg_min_idx, cap_size.height), cvScalar(0,255,0));
		
		cvLine(ipl_arr[I_CHIST_1],
			   cvPoint(h_avg_peak_idx, 0),
			   cvPoint(h_avg_peak_idx, cap_size.height), cvScalar(255,0,0));
		cvLine(ipl_arr[I_CHIST_1],
			   cvPoint(h_avg_peak_idx+h_avg_min_idx, 0),
			   cvPoint(h_avg_peak_idx+h_avg_min_idx, cap_size.height), cvScalar(0,255,0));

		cvLine(ipl_arr[I_CHIST_2],
			   cvPoint(h_avg_peak_idx, 0),
			   cvPoint(h_avg_peak_idx, cap_size.height), cvScalar(255,0,0));
		cvLine(ipl_arr[I_CHIST_2],
			   cvPoint(h_avg_peak_idx+h_avg_min_idx, 0),
			   cvPoint(h_avg_peak_idx+h_avg_min_idx, cap_size.height), cvScalar(0,255,0));

		
		ofx_hist0 = ipl_arr[I_CHIST_0];
		ofx_hist1 = ipl_arr[I_CHIST_1];
		ofx_hist2 = ipl_arr[I_CHIST_2];
		
		// record the frame if needed
		if( bToggleRec && bNewFrame ){
			
			of_gray_src.setFromPixels((uchar*)(ipl_arr[I_GSRC]->imageData),
									  cap_size.width, cap_size.height,
									  OF_IMAGE_GRAYSCALE);
			sprintf(reportStr, "/Users/jdewitt/doc/eye/capture/img_%04d.tiff",cap_count.cur);
			printf("save@ %s\n", reportStr);
			of_gray_src.saveImage( reportStr );
			rec_frame++;
		}
		
		// reset flags
		bNewFrame = false;
		bNeedRefresh = false;
	}
	
}

/**
 *	This method is OpenFrameworks' draw tick function.
 *	@date 2012/05/07
 */
void cvEyeTracker::draw(){
    
	ofEnableBlendMode(OF_BLENDMODE_ALPHA);
	// background
	ofSetColor(64,64,64,255);
	ofRect(0, 0, ofGetWidth(), ofGetHeight());
    
    ofSetColor(255,255,255,255);
	
	int bord = 15;
	int space_x = 10;
	int space_y = 10;
	int tmp_y = 0;
	int text_h = 12;
	
	// draw images, histograms, values
	if( bDrawEyeImage ){
		ofx_color_out0.draw(bord+space_x, space_y);
	}
	if( bDrawDevScreen ){
		
		// draw the images
//		ofx_color_out1.draw( bord+space_x, space_y+cap_size.height+bord );
		ofx_gray_out1.draw( bord+space_x, space_y+cap_size.height+bord );
		ofx_gray_out2.draw( bord+space_x+cap_size.width+bord, space_y+cap_size.height+bord );
		ofx_hist0.draw( bord+space_x+cap_size.width+bord*1+ofx_hist0.width*0, space_y);
		ofx_hist1.draw( bord+space_x+cap_size.width+bord*2+ofx_hist0.width*1, space_y);
		ofx_hist2.draw( bord+space_x+cap_size.width+bord*3+ofx_hist0.width*2, space_y);
		
		// draw text info
		ofSetColor(255, 255, 255,255);
		tmp_y = space_y+cap_size.height+bord*2;

		float idx0 = h_avg_peak_idx;
		float idx1 = h_avg_peak_idx + h_avg_min_idx;
		
		// threshold values
		sprintf(reportStr, "idx0  : %f", idx0 );
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);
		sprintf(reportStr, "idx1  : %f", idx1 ); tmp_y+=text_h;
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);
		sprintf(reportStr, "diff  : %f", idx1-idx0 ); tmp_y+=text_h;
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);
		
		// pupil
		sprintf(reportStr, "pupil hist val0 : %f", hist_pupil_avg[(int)idx0] ); tmp_y+=text_h;
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);
		sprintf(reportStr, "pupil hist val1 : %f", hist_pupil_avg[(int)idx1] ); tmp_y+=text_h;
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);

		// iris
		sprintf(reportStr, "iris hist val0 : %f", hist_iris_avg[(int)idx0] ); tmp_y+=text_h;
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);
		sprintf(reportStr, "iris hist val1 : %f", hist_iris_avg[(int)idx1] ); tmp_y+=text_h;
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);
		
		tmp_y+=text_h;
		
		sprintf(reportStr, "ell : %3.2f, %3.2f", pupil_fit_cur.size.width, pupil_fit_cur.size.height ); tmp_y+=text_h;
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);
		sprintf(reportStr, "asp : %3.2f", pupil_fit_cur.size.width/pupil_fit_cur.size.height ); tmp_y+=text_h;
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);
		
		float tmp_ang = pupil_fit_cur.angle;
		if(tmp_ang > 180){
			sprintf(reportStr, "ang*: %3.2f", tmp_ang-180 ); tmp_y+=text_h;
		} else {
			sprintf(reportStr, "ang : %3.2f", tmp_ang ); tmp_y+=text_h;
		}
		
		myfont0.drawString(reportStr, bord+space_x+cap_size.width*2+bord*2,tmp_y);
	}	
	
	// 
	if( false && bDrawPerkinjePoints ){
		CvPoint2D32f *tmp_pt;
		for(int i=0;i<4;i++){
			tmp_pt = (CvPoint2D32f*)cvGetSeqElem(perkinje_dst_pts, i);
			CvScalar the_color = CV_RGB(255, 255, 255);
			if(i==0) the_color = CV_RGB(255,   0,   0);
			if(i==1) the_color = CV_RGB(  0, 255,   0);
			if(i==2) the_color = CV_RGB(  0,   0, 255);
			if(i==3) the_color = CV_RGB(255, 255,   0);
			ofSetColor(the_color.val[0], the_color.val[1], the_color.val[2], 200);
			ofCircle(tmp_pt->x-ofGetWindowPositionX(),
					 tmp_pt->y-ofGetWindowPositionY(), 32);
			
		}
	}
	
	// draw the calibration map
	if( bDrawCalibrationSourcePoints || bDrawCalibrationTargetPoints ){
		
		CvPoint2D32f src_pt,dst_pt;
		int w = calibration_map.cols;
		float tmp_x, tmp_y, tmp_a, tmp_b;
		float* src_ptr = NULL;
		float* dst_ptr = NULL;
		CvScalar the_color;
		
		for(int r=0;r<calibration_map.rows;r++){
			src_ptr = (float*)(calibration_map.src_mat->data.ptr + r * calibration_map.src_mat->step);
			dst_ptr = (float*)(calibration_map.dst_mat->data.ptr + r * calibration_map.dst_mat->step);
			for(int c=0;c<calibration_map.cols;c++){
				
				// row and column color basis
				tmp_a = (r/(float)(calibration_map.rows-1));
				tmp_b = (c/(float)(calibration_map.cols-1));
				the_color = CV_RGB( tmp_a*255,
								    tmp_b*255,
								    (1-tmp_a)*(1-tmp_b)*255 );
				
				// destination points in screen space
				dst_pt.x = dst_ptr[2*c+0];
				dst_pt.y = dst_ptr[2*c+1];
				dst_pt.x -= ofGetWindowPositionX();
				dst_pt.y -= ofGetWindowPositionY();
				
				if( bDrawCalibrationTargetPoints ){
					
					ofSetColor(the_color.val[0],the_color.val[1],the_color.val[2], 220);
					ofCircle(dst_pt.x,
							 dst_pt.y, 16);
					
					if( c == calibration_map.selected.x &&
					   r == calibration_map.selected.y ){
						ofSetColor(255,255,255,255);
						ofCircle(dst_pt.x,
								 dst_pt.y, 10);
						ofSetColor(255,0,0,255);
						ofCircle(dst_pt.x,
								 dst_pt.y, 8);
						ofSetColor(255, 255, 255,200);
						int s = 10*0.70710678118655;
						ofSetLineWidth(2);
						ofLine(dst_pt.x-s, dst_pt.y-s, dst_pt.x+s, dst_pt.y+s);
						ofLine(dst_pt.x-s, dst_pt.y+s, dst_pt.x+s, dst_pt.y-s);
					}
				}
				
				// source points in projected space
				src_pt.x = src_ptr[2*c+0];
				src_pt.y = src_ptr[2*c+1];
				src_pt.x -= ofGetWindowPositionX();
				src_pt.y -= ofGetWindowPositionY();
				
				if( bDrawCalibrationSourcePoints ) {
					
					ofSetColor(the_color.val[0],the_color.val[1],the_color.val[2], 40);
					ofCircle(src_pt.x,
							 src_pt.y, 16);
					
					ofSetLineWidth(3);
					
					ofSetColor(the_color.val[0],the_color.val[1],the_color.val[2], 200);
					ofLine(src_pt.x, src_pt.y, dst_pt.x, dst_pt.y);
					
					if( c == calibration_map.selected.x &&
					   r == calibration_map.selected.y ){
						ofSetColor(255,255,255,150);
						ofCircle(src_pt.x,
								 src_pt.y, 10);
						ofSetColor(255,0,0,150);
						ofCircle(src_pt.x,
								 src_pt.y, 8);
					}
				}
				
			}
		}
	}
	
	// draw the guess point
	if( bDrawScreenGuessPoint ){
		ofEnableSmoothing();
		
		float guess_alpha = 1.0;
		if( true ){
			ofSetColor(0 , 128, 255, 15);
			ofCircle( guess_pt.x - ofGetWindowPositionX(), guess_pt.y - ofGetWindowPositionY(), 64);
			ofSetColor(  0, 255,   0, 80*guess_alpha);
			ofCircle( guess_pt_avg.x - ofGetWindowPositionX(), guess_pt_avg.y - ofGetWindowPositionY(), 9);
			ofSetColor(  0,   0,   0, 120*guess_alpha);
			ofCircle( guess_pt_avg.x - ofGetWindowPositionX(), guess_pt_avg.y - ofGetWindowPositionY(), 8);
			ofSetColor(255, 255, 255, 255*guess_alpha);
			ofCircle( guess_pt_avg.x - ofGetWindowPositionX(), guess_pt_avg.y - ofGetWindowPositionY(), 4);
		}
	}
	if( bDrawScreenGazePoint ){
		ofSetColor(0 , 128, 255, 15);
		ofCircle( screen_pt.x - ofGetWindowPositionX(), screen_pt.y - ofGetWindowPositionY(), 64);
		ofSetColor(  0,   0, 255, 120);
		ofCircle( screen_pt.x - ofGetWindowPositionX(), screen_pt.y - ofGetWindowPositionY(), 10);
		ofSetColor(  0,   0,   0, 120);
		ofCircle( screen_pt.x - ofGetWindowPositionX(), screen_pt.y - ofGetWindowPositionY(), 8);
		ofSetColor(255, 255, 255, 255);
		ofCircle( screen_pt.x - ofGetWindowPositionX(), screen_pt.y - ofGetWindowPositionY(), 4);
		ofDisableSmoothing();
		
	}
	
	if( bDrawScreenGazePoint ){
		screen_forreal.x = guess_pt_avg.x - offset_center.x - ofGetWindowPositionX() + ofGetScreenWidth()*0.5;
		screen_forreal.y = guess_pt_avg.y - offset_center.y - ofGetWindowPositionY() + ofGetScreenHeight()*0.5;
		
		screen_forreal.x = screen_pt.x - ofGetWindowPositionX();
		screen_forreal.y = screen_pt.y - ofGetWindowPositionY();
	}
	
	// getting screen capture
	if(false){
		if( cur_frame > max_frame ){
			sprintf(reportStr, "~/doc/eye/screencap/img_%04d.jpg", cur_frame );
			of_img0.grabScreen(0, 0, ofGetWindowWidth(), ofGetWindowHeight());
			//			-of_img0.saveImage(reportStr);
			max_frame = MAX( cur_frame, max_frame );
		}
	}

    
}

/**
 * This method calculates the location and size of the pupil as an ellipse.
 *	@param img_in		An image of an eye.
 *	@param img_thresh	Threshold value used for image.
 *	@param found_pt		Filled with the location of the center of the pupil.
 *	@date 2012/05/07
 */
bool cvEyeTracker::findPupil(IplImage *img_in,
							 int img_thresh,
							 int flood_stop,
							 CvBox2D *found_ellipse,
							 CvRect *bbox_out){
	bool ret = false;
    
	cvThreshold(ipl_arr[I_GSRC], ipl_arr[I_GTHRESH_0], img_thresh, 255, CV_THRESH_BINARY_INV);
	
	// find a point inside the pupil (to start)
	// threshold the image, and average the resulting "on" pixel locations
	float avg_x = 0;
	float avg_y = 0;
	int avg_n = 0;
	uchar *img_data = (uchar*)ipl_arr[I_GTHRESH_0]->imageData;
	for(int row=0;row<cap_size.height;row++){
		for(int col=0;col<cap_size.width;col++){
			if( img_data[col] > 0 ){
				avg_x += col;
				avg_y += row;
				avg_n += 1;
			}
		}
		img_data += ipl_arr[I_GTHRESH_0]->widthStep;
	}
	
	// if there aren't enough pixels, give up
	if( avg_n < 50 ){
		return false;
	}
	
	avg_x /= avg_n;
	avg_y /= avg_n;
	
	// clamp the point inside the image space
	avg_x = min(max(avg_x,0.0f),(float)cap_size.width-1);
	avg_y = min(max(avg_y,0.0f),(float)cap_size.height-1);
	
	cvCircle(ipl_arr[I_GTHRESH_0], cvPoint(avg_x, avg_y), 128, cvScalar(255));
	
	// floodfill to get pupil area
	int flood_r = 4;  //( box_ellipse_avg.size.width*0.5 + 0.5*box_ellipse_avg.size.height ) * 0.5;
	cvCopy(ipl_arr[I_GSRC], ipl_arr[I_GTMP_0]);
	cvCircle(ipl_arr[I_GTMP_0], cvPoint(avg_x, avg_y), flood_r, cvScalarAll(img_thresh), -1, 10, 0);
	cvSmooth(ipl_arr[I_GTMP_0], ipl_arr[I_GTMP_0], CV_MEDIAN, 3,0,0,0 );	
	cvFloodFill(ipl_arr[I_GTMP_0],
				cvPoint(avg_x,avg_y),
				cvScalarAll(0),
				cvScalarAll(img_thresh),
				cvScalarAll(abs(flood_stop)),
				&pupil_ccomp,
				CV_FLOODFILL_FIXED_RANGE,
				NULL);
	
	cvThreshold(ipl_arr[I_GTMP_0], ipl_arr[I_GTHRESH_1], img_thresh, 255, CV_THRESH_BINARY_INV);
//	cvSmooth(ipl_arr[I_GTHRESH_1], ipl_arr[I_GTHRESH_1], CV_MEDIAN, 3,0,0,0 );
	ofx_gray_out2 = ipl_arr[I_GTHRESH_1];
	
	// find contour that represents pupil
	cvClearSeq(fin_contour);
	CvSeq* cur_contour = 0;
	CvSeq* first_contour = 0;
	cvCopy(ipl_arr[I_GTHRESH_1], ipl_arr[I_GTMP_0]);
	cvClearMemStorage(g_storage[1]);
	cvFindContours( ipl_arr[I_GTMP_0], g_storage[1], &first_contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cvPoint(0,0) );
	
	cvSetZero(ipl_arr[I_GTMP_0]);
	cvDrawContours(ipl_arr[I_GTMP_0], first_contour, cvScalarAll(255), cvScalarAll(255), 5, 1, CV_AA);
	
	float avg_r = max( (found_ellipse->size.width*0.5+found_ellipse->size.height*0.5)*0.5, 35.0 );
	CvPoint *tmp_pt;
	float this_a,this_dist;
	float dist2last;
	float x_avg,y_avg;
	float x_min,x_max,y_min,y_max;
	int point_cnt;
	float max_a = 0;
	for(cur_contour = first_contour; cur_contour!=NULL; cur_contour = cur_contour->h_next) {
		this_a = cvContourArea(cur_contour);
		point_cnt = cur_contour->total;
		
		x_avg = 0;
		y_avg = 0;
		for(int i=0;i<point_cnt;i++) {
			tmp_pt = (CvPoint*)cvGetSeqElem(cur_contour, i);
			x_avg += tmp_pt->x;
			y_avg += tmp_pt->y;
			if((i==0)||(tmp_pt->x)<(x_min)){ x_min = tmp_pt->x; }
			if((i==0)||(tmp_pt->x)>(x_max)){ x_max = tmp_pt->x; }
			if((i==0)||(tmp_pt->y)<(y_min)){ y_min = tmp_pt->y; }
			if((i==0)||(tmp_pt->y)>(y_max)){ y_max = tmp_pt->y; }
		}
		x_avg /= point_cnt;
		y_avg /= point_cnt;
		
		this_dist = sqrt( pow(center_pt.x-x_avg,2) + pow(center_pt.y-y_avg,2) );
		dist2last = sqrt( pow(avg_x-x_avg,2.0f) + pow(avg_x-y_avg,2.0f) );
				
		if( this_a > 250 ){
			// cvCircle(ipl_arr[I_GTHRESH_0], cvPoint(x_avg, y_avg), 4, cvScalarAll(80));
			max_a = this_a;
			for(int i=0;i<point_cnt;i++) {
				tmp_pt = (CvPoint*)cvGetSeqElem(cur_contour, i);
				cvSeqPush(fin_contour, tmp_pt);
			}
		}
		
	}
	
	// calculate convex hull
//	cvClearSeq(seq_hull);
	seq_hull = cvConvexHull2( fin_contour, 0, CV_CLOCKWISE, 1);
	
	// fit ellipse and use if valid
	if(seq_hull != NULL){
		if(seq_hull->total >= 6){ // only 6 required
			CvRect pupil_bounds = cvBoundingRect(seq_hull,1);
			bbox_out->x = pupil_bounds.x;
			bbox_out->y = pupil_bounds.y;
			bbox_out->width = pupil_bounds.width;
			bbox_out->height = pupil_bounds.height;
			
			CvBox2D this_fit = cvFitEllipse2(seq_hull);
			
			float eccentricity = this_fit.size.width/this_fit.size.height;
			if(eccentricity>1.0) eccentricity = 1.0 / eccentricity;
			
			// make sure the fit makes sense
			if(eccentricity>0.35){
				found_ellipse->size   = cvSize2D32f(MIN(MAX(this_fit.size.width,1),cap_size.width),
													MIN(MAX(this_fit.size.height,1),cap_size.width));
				found_ellipse->center = cvPoint2D32f(MIN(MAX(this_fit.center.x,0),cap_size.width),
													 MIN(MAX(this_fit.center.y,0),cap_size.height));
				found_ellipse->angle  = this_fit.angle;
				ret = true;
			}
		}
	}
	
    return ret;
}

/**
 *	This method is used to sort points based on their distance from a fixed one.
 *	@date 2012/04/26
 */
static int dist_cmp_func( const void* _a, const void* _b, void* _center ){
	CvPoint2D32f* a = (CvPoint2D32f*)_a;
    CvPoint2D32f* b = (CvPoint2D32f*)_b;
	CvPoint2D32f* c = (CvPoint2D32f*)_center;
    float a_dist = (sqrt(pow(a->x - c->x,2.0f)+pow(a->y - c->y,2.0f))); // center <-> A dist
    float b_dist = (sqrt(pow(b->x - c->x,2.0f)+pow(b->y - c->y,2.0f))); // center <-> B dist
	int diff_dist = (int)(100*(a_dist-b_dist)); // order : ascending distance to center
    return diff_dist;
}

/**
 *	This method calculates the location of the reflected LEDs.
 *  Four surfaces in the eye cause reflections of decreasing brightness to be seen.
 *	The LED reflections are sorted in a consistent order.
 *	@param img_in		An image of the eye.
 *	@param seed_pt		The location of the pupil, used for sorting points.
 *	@param found_pt		The destination CvSeq pointer that holds the ordered points.
 *	@date 2012/05/07
 */
bool cvEyeTracker::findPerkinje( IplImage *img_in, CvPoint2D32f seed_pt, CvSeq* found_pt ){
	
	// threshold and smooth
	cvAdaptiveThreshold(img_in,    ipl_arr[I_GTHRESH_0], 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 23, -50);
	
	cvCopy(ipl_arr[I_GTHRESH_0], ipl_arr[I_GTMP_0]);
	if( bDrawPupilEllipse ){
		cvEllipse(ipl_arr[I_GTMP_0],
				  cvPoint(pupil_fit_cur.center.x, pupil_fit_cur.center.y),
				  cvSize(pupil_fit_cur.size.width*0.5, pupil_fit_cur.size.height*0.5),
				  pupil_fit_cur.angle, 0, 360.0, CV_RGB(255,255,255));
		cvCircle(ipl_arr[I_GTMP_0],
				 cvPoint(pupil_fit_cur.center.x, pupil_fit_cur.center.y),
				 4, CV_RGB(255,255,255));
	}
	ofx_gray_out1 = ipl_arr[I_GTMP_0];
//	cvThreshold(img_in, ipl_arr[I_GTHRESH_0], 255-30, 255, CV_THRESH_BINARY);
	
	// cvSmooth(ipl_arr[I_GTHRESH_0], ipl_arr[I_GTHRESH_0], CV_MEDIAN, 1);
	
	// find glint contours in threshold image
	CvSeq* first_contour = 0;
	cvClearMemStorage(g_storage[1]);
	cvCopy(ipl_arr[I_GTHRESH_0], ipl_arr[I_GTMP_0]);
	cvFindContours( ipl_arr[I_GTMP_0],
				   g_storage[1],
				   &first_contour,
				   sizeof(CvContour),
				   CV_RETR_EXTERNAL,
				   CV_CHAIN_APPROX_NONE,
				   cvPoint(0,0) );
	
	// use I_GRAY_1 as a mask
	cvSetZero(ipl_arr[I_GRAY_1]);
	cvClearSeq(pk_tmp);
	
	int max_perkinje_idx = 0;
	CvSeq* cur_contour = NULL;
	
	// calculate weighted center of gravity for all canditate points
	// place them in pk_tmp
	for(cur_contour = first_contour; cur_contour!=NULL; cur_contour = cur_contour->h_next) {
		int point_cnt = cur_contour->total;
		if( point_cnt ){
			
			CvPoint* tmp_pt = NULL;
			// calculate bounding box
			int _xmin,_xmax,_ymin,_ymax;
			for(int i=0;i<point_cnt;i++) {
				tmp_pt = (CvPoint*)cvGetSeqElem(cur_contour, i);
				if(i==0){
					_xmin = tmp_pt->x;
					_xmax = tmp_pt->x;
					_ymin = tmp_pt->y;
					_ymax = tmp_pt->y;
				}
				_xmin = MIN(_xmin,tmp_pt->x);
				_xmax = MAX(_xmax,tmp_pt->x);
				_ymin = MIN(_ymin,tmp_pt->y);
				_ymax = MAX(_ymax,tmp_pt->y);
			}
			int roi_bord = 4;
			_xmin = MAX(_xmin-roi_bord,0);
			_ymin = MAX(_ymin-roi_bord,0);
			_xmax = MIN(_xmax+roi_bord,cap_size.width);
			_ymax = MIN(_ymax+roi_bord,cap_size.height);
			
			cvRectangle(ipl_arr[I_GRAY_1],
						cvPoint(_xmin, _ymin),
						cvPoint(_xmax, _ymax), cvScalarAll(255),
						1, 8, 0);
			
			int this_w = _xmax-_xmin;
			int this_h = _ymax-_ymin;
			
			// calculate min/max pixel value inside box
			uchar *img_data = (uchar*)(ipl_arr[I_GSRC]->imageData);
			int img_widthstep = ipl_arr[I_GSRC]->widthStep;
			float vmin,vmax;
			for(int r=0;r<this_h;r++){
				for(int c=0;c<this_w;c++){
					uchar val = img_data[ (_ymin+r)*img_widthstep+(_xmin+c) ];
					if(r==0&&c==0){
						vmin = val;
						vmax = val;
					}
					vmin = MIN(vmin,val);
					vmax = MAX(vmax,val);
				}
			}
			
			if(DBUG) printf("min,max : %d,%d\n", (int)vmin,(int)vmax);
			
			// calculate weighted center of gravity
			float avg_x = 0, avg_y = 0;
			float this_weight, weight_tot = 0;
			for(int r=0;r<this_h;r++){
				for(int c=0;c<this_w;c++){
					uchar val = img_data[ (_ymin+r)*img_widthstep+(_xmin+c) ];
					this_weight = ( 1.0 - min(max( ( val - vmin ) / ( vmax - vmin ), 0.0f),1.0f) );
//					this_weight = (255-val);
					avg_x += this_weight * c;
					avg_y += this_weight * r;
					weight_tot += this_weight;
				}
			}
			if(weight_tot>0){
				avg_x /= weight_tot;
				avg_y /= weight_tot;
			}
			avg_x += _xmin;
			avg_y += _ymin;
			
			if(DBUG) printf("x,y : %d,%d\n", (int)avg_x,(int)avg_y);
			
			// calculate distance from seed point (pupil)
			float this_dist = sqrt(pow(avg_x-seed_pt.x,2.0f)+pow(avg_y-seed_pt.y,2.0f));
			float this_a = abs( cvContourArea(cur_contour) );
			if( this_a > 5 &&
			   this_a < 35 ){
//				&& this_dist < sqrt( pow(cap_size.width,2.0)+pow(cap_size.height,2.9) )*0.45 ){
				
				CvPoint2D32f ins_pt = cvPoint2D32f(avg_x, avg_y);
				cvSeqPush(pk_tmp, &ins_pt);
				max_perkinje_idx++;

				if(DBUG) printf("found pk: %d (%3.2f)\n", pk_tmp->total, this_a);
				
			} else if ( this_a <= 2.5) {
				//			printf(" **[%d]: %3.3f, %3.3f : %3.3f px\n", max_perkinje_idx, this_x, this_y, this_a );
			}
			
			
		}
	}
	
	// if enough points were found
	if( max_perkinje_idx >= 4 ){
		
		// sort pk_tmp based on distance from seed point
		cvSeqSort(pk_tmp, dist_cmp_func, &seed_pt);
		
		if(DBUG) printf("-------------\n");
		if(DBUG) printf("max_perkinje_idx = %d (%d)\n", max_perkinje_idx, pk_tmp->total );
		
		CvPoint2D32f* tmp_pt32;
		// calculate center of gravity of first four points
		float cog_x = 0, cog_y = 0;
		for(int i=0; i < 4; i++){
			tmp_pt32 = (CvPoint2D32f*)cvGetSeqElem(pk_tmp, i);
			// these points have been sorted by distance from pupil center
			cog_x += tmp_pt32->x;
			cog_y += tmp_pt32->y;
		}
		cog_x /= MIN(pk_tmp->total, 4);
		cog_y /= MIN(pk_tmp->total, 4);
		
		//		center_x = (center_x + found_pt->x)/2.0;
		//		center_y = (center_y + found_pt->y)/2.0;
		
		float tmp_min_x,tmp_max_x;
		float tmp_min_y,tmp_max_y;
		for(int i=0;i<max_perkinje_idx;i++){
			tmp_pt32 = (CvPoint2D32f*)cvGetSeqElem(pk_tmp, i);
			if(i==0){
				tmp_min_x = tmp_pt32->x;
				tmp_min_y = tmp_pt32->y;
				tmp_max_x = tmp_pt32->x;
				tmp_max_y = tmp_pt32->y;
			} else {
				tmp_min_x = min( tmp_min_x, tmp_pt32->x );
				tmp_max_x = max( tmp_max_x, tmp_pt32->x );
				tmp_min_y = min( tmp_min_y, tmp_pt32->y );
				tmp_max_y = max( tmp_max_y, tmp_pt32->y );
			}
		}
		
		//		center_x = (tmp_min_x+tmp_max_x)*0.5;
		//		center_y = (tmp_min_y+tmp_max_y)*0.5;
		
		if(DBUG) printf( "x:[%3.3f -> %3.3f], y:[%3.3f -> %3.3f]\n", tmp_min_x, tmp_max_x, tmp_min_y, tmp_max_y );
		if(DBUG) printf( "center: (%3.3f,%3.3f), (%3.3f,%3.3f)\n",
			   cog_x, cog_y,
			   (tmp_min_x+tmp_max_x)*0.5, (tmp_min_y+tmp_max_y)*0.5 );
		
		// I_GRAY_0 is perkinje info overlaid on source image
		cvCopy(ipl_arr[I_GSRC], ipl_arr[I_GRAY_0]);
		
		int set_idx[4];
		for(int i=0;i<4;i++){ // MIN(pk_seq->total,N_PERKINJE_SEEK)
			set_idx[i] = -1;
			tmp_pt32 = (CvPoint2D32f*)cvGetSeqElem(pk_tmp, i);
			
			if(bDrawPerkinjePoints) cvCircle(ipl_arr[I_GRAY_0], cvPoint(tmp_pt32->x, tmp_pt32->y), 3, cvScalar(255));
			//			if(bDrawPerkinjePoints) cvCircle(ipl_arr[I_GRAY_1], cvPoint(tmp_pt32->x, tmp_pt32->y), 3+i*2, cvScalar(255));
			//			cvCircle(ipl_arr[I_GRAY_0], cvPoint((tmp_min_x+tmp_max_x)*0.5,(tmp_min_y+tmp_max_y)*0.5), 3, cvScalar(128));
			//			cvCircle(ipl_arr[I_GRAY_0], cvPoint(tmp_pt32->x,tmp_pt32->y), 8, cvScalar(255));
			//			cvCircle(ipl_arr[I_GRAY_0], cvPoint(tmp_pt32->x,tmp_pt32->y), 8, cvScalar(255));
			cvLine(ipl_arr[I_GRAY_0], cvPoint(cog_x,cog_y), cvPoint(tmp_pt32->x, tmp_pt32->y), cvScalar(255), 1, CV_AA, 0);
			
			float this_ang = atan2( tmp_pt32->x - cog_x, tmp_pt32->y - cog_y );
			float check_ang;
			float dif = 0.7;
			
//			printf("pk: a=%2.3f\n", this_ang );
			
			check_ang = 0.78;
			if( check_ang - dif < this_ang && this_ang < check_ang + dif ){ set_idx[i] = 2; }
			
			check_ang = 2.35;
			if( check_ang - dif < this_ang && this_ang < check_ang + dif ){ set_idx[i] = 1; }
			
			check_ang = -2.35;
			if( check_ang - dif < this_ang && this_ang < check_ang + dif ){ set_idx[i] = 0; }
			
			check_ang = -0.78;
			if( check_ang - dif < this_ang && this_ang < check_ang + dif ){ set_idx[i] = 3; }
						
			if(DBUG) printf("pk[%d rly %d] : %+3.3f  @ (%3.3f,%3.3f) rly (%3.3f,%3.3f) = %3.3f\n",
				   i, set_idx[i], this_ang,
				   tmp_pt32->x, tmp_pt32->y,
				   tmp_pt32->x - cog_x, tmp_pt32->y - cog_y,
				   (tmp_pt32->x - tmp_min_x)+(tmp_pt32->y - tmp_min_y)/2.0f );
			
			CvPoint2D32f *src_pt = 0;
			CvPoint2D32f *dest_pt = 0;
			
			// write out the points to the src array
			float sm2 = 0;
			if( set_idx[i] > -1 ){
				src_pt  = (CvPoint2D32f*)cvGetSeqElem(pk_tmp, i);
				dest_pt = (CvPoint2D32f*)cvGetSeqElem(found_pt, set_idx[i]);
				dest_pt->x = src_pt->x;
				dest_pt->y = src_pt->y;
			}
		}
		return true;
	}
	
	return false;
}

/**
 *	This method captures a frame, and returns indicating whether it was new.
 *	@param ipl_csrc		The destination 3-channel 8-bit IplImage
 *	@date 2012/05/07
 */
bool cvEyeTracker::getFrame( IplImage *ipl_csrc ){
	
    if(cap_context.mode==CV_USE_LIVE_VIDEO){
        
        cap_context.vg.update();
        if( cap_context.vg.isFrameNew() ){
            memcpy(ipl_csrc->imageData, cap_context.vg.getPixels(), ipl_csrc->imageSize);
            cap_count.cur++;
            cap_count.tot++;
            return true;
        }
        return false;
    }
    
    if(cap_context.mode==CV_USE_VIDEO){
        
        // frame step mode
        if( bUseStep ){
            if (cap_context.vp.isPlaying()) {
                cap_context.vp.setPaused(true);
            }
            
            if(step_rdy<0){ cap_context.vp.previousFrame(); }
            if(step_rdy>0){ cap_context.vp.nextFrame(); }
            cap_context.vp.update();
        }
        
        // freerun mode
        else {
            //            ofSetFrameRate(FPS_HI);
            //            if (cap_context.vp.isPaused()) {
            //                cap_context.vp.play();
            //            }
            cap_context.vp.update();
        }
        
        // copy
        if( cap_context.vp.isFrameNew() ){
            if(bUseStep) {
                if(step_rdy<0) step_rdy++;
                if(step_rdy>0) step_rdy--;
            }
            cap_count.cur =  cap_context.vp.getCurrentFrame();
            memcpy(ipl_csrc->imageData, cap_context.vp.getPixels(), ipl_csrc->imageSize);
            return true;
        }
        return false;
    }
    
    return false;
}

/**
 *	This method calculates the histogram for a masked area of an image.
 *	@param img_in		The source IplImage.
 *	@param hist_out		The destination CvHistogram structure.
 *	@param mask_in		The source IplImage mask.
 *	@date 2012/05/07
 */
void cvEyeTracker::calcHistogramMask( IplImage *img_in, CvHistogram *hist_out, IplImage *mask_in ){
	cvClearHist(hist_out);
	cvCalcHist(&img_in, hist_out, 0, mask_in);

}

/**
 *	This method calculates the histogram for a Region of Interest.
 *	@param img_in		The source IplImage.
 *	@param hist_out		The destination CvHistogram structure.
 *	@param the_roi		The ROI to use.
 *	@date 2012/05/07
 */
void cvEyeTracker::calcHistogramROI( IplImage *img_in, CvHistogram *hist_out, CvRect the_roi ){
	
	cvClearHist(hist_out);
	
	the_roi.x = MIN(MAX(the_roi.x,0),img_in->width-1);
	the_roi.y = MIN(MAX(the_roi.y,0),img_in->height-1);
	
	the_roi.width  = MIN(MAX(img_in->width -the_roi.x,0),img_in->width);
	the_roi.height = MIN(MAX(img_in->height-the_roi.y,0),img_in->height);
	
	cvSetImageROI(img_in, the_roi);
	cvCalcHist(&img_in, hist_out, 0, NULL);
	cvResetImageROI(img_in);

}

/**
 *	This method draws a histogram into an image.
 *	@param hist_in		The source histogram; array of 256 floats (8-bit is assumed)
 *	@param img_out		The destination IplImage.
 *	@date 2012/05/07
 */
void cvEyeTracker::drawHistogram(float *hist_in, IplImage *img_out) {
	
	cvSetZero(img_out);
	
	int _x = 0;
	int _y = img_out->height;
	int _w = img_out->width;
	int _h = img_out->height;
	
	float hscale = _w/256.0;
	
	float histmax = 0;
	float tmp_v;
	for(int i=1;i<255;i++){
		tmp_v = hist_in[i];
		hist_maxv = max(tmp_v,hist_maxv);
	}
	
	histmax = _w*_h*0.035;
	
	for(int i=0;i<255;i++){
		float thisval = min(1.0f, hist_in[i]  /histmax);
		float nextval = min(1.0f, hist_in[i+1]/histmax);
		CvPoint pt1 = cvPoint(_x+(i  )*hscale, _h);
		CvPoint pt2 = cvPoint(_x+(i  )*hscale, _y-_h*thisval);
		CvPoint pt3 = cvPoint(_x+(i+1)*hscale, _y-_h*nextval);
		CvPoint pt4 = cvPoint(_x+(i+1)*hscale, _h);
		CvPoint pts[] = {pt1,pt2,pt3,pt4,pt1};
		cvFillConvexPoly(img_out, pts, 5, cvScalar(255,255,255));
	}
	/*
	float _u = h_manual_threshold / 255.0;
	
	cvLine(img_out,
		   cvPoint(_u*_w,  0),
		   cvPoint(_u*_w, _h),
		   cvScalar(255,0,0));
	
	*/
	
	/*
	 //h_peak_avg
	 cvLine(img_out,
	 cvPoint(0,                img_out->height*(1.0-h_avg_peak_val/hist_maxv)),
	 cvPoint(h_avg_peak_idx-3, img_out->height*(1.0-h_avg_peak_val/hist_maxv)),
	 cvScalar(255,0,0));
	 cvLine(img_out,
	 cvPoint(h_avg_peak_idx, 0),
	 cvPoint(h_avg_peak_idx, img_out->height*(1.0-hist_in[(int)(h_avg_peak_idx)]/hist_maxv)),
	 cvScalar(255,0,0));
	 
	 cvLine(img_out,
	 cvPoint(h_cur_min_idx, 0),
	 cvPoint(h_cur_min_idx, img_out->height*(1.0-(0.03+hist_in[(int)(h_cur_min_idx)]/hist_maxv))),
	 cvScalar(0,255,0));
	 
	 cvLine(img_out,
	 cvPoint(getAutoThresh(), 0),
	 cvPoint(getAutoThresh(), img_out->height*(1.0-(0.03+hist_in[getAutoThresh()]/hist_maxv))),
	 cvScalar(0,0,255));
	 */
	/*
	 // histogram: pupil peak
	 cvLine(dst, cvPoint(h_peakidx_avg_cur, 0),
	 cvPoint(h_peakidx_avg_cur, dst->height*(1.0-avg_hist1[h_peakidx_avg_cur]/histmax)-4 ),
	 cvScalar(255,0,0));
	 // histogram: pupil end
	 cvLine(dst, cvPoint(h_minidx_avg, 0),
	 cvPoint(h_minidx_avg, dst->height*(1.0-avg_hist1[(int)h_minidx_avg]/histmax)-10 ),
	 cvScalar(0,255,0));
	 // histogram: autothresh
	 cvLine(dst, cvPoint(getIntThresh(), 0),
	 cvPoint(getIntThresh(), dst->height*(1.0-avg_hist1[getIntThresh()]/histmax)-15  ),
	 cvScalar(0,0,255));
	 */
}

/**
 *	This method calculates the first peak based on two histograms.
 *	The first peak corresponds to the pupil, as it is the darkest collection of pixels.
 *  The first histogram is of the surrounding area, the second histogram is of the pupil.
 *	@param tot_hist		Source histogram; array of 256 floats (8-bit is assumed)
 *	@param roi_hist		Source histogram; array of 256 floats (8-bit is assumed)
 *  @returns			Calculated peak threshold value.
 *	@date 2012/05/07
 */
void cvEyeTracker::calcHistPeak( float *pupil_hist, float *iris_hist, int *peak, int *valley ){
	
	// analyze two histograms to find needed values
	// 1: pupil threshold value
	// 2: flood fill stop value
	
	// crawl the pupil histogram
	// stop when ahead values smaller than the max dominate
	int ahead_n = 10;
	int ahead_more;
	int ahead_less;
	float peak_max = 0;
	int   peak_idx = 0;
	for(int i=0;i<200;i++){
		// calculate maximum value
		if(pupil_hist[i] > peak_max ){
			peak_max = pupil_hist[i];
			peak_idx = i;
		}
		// count the number of values ahead that are less than the max
		ahead_more = 0;
		ahead_less = 0;
		for(int j=0;j<ahead_n;j++){
			if(pupil_hist[i+j] < peak_max){
				ahead_less++;
			} else {
				ahead_more++;
			}
		}
		// stop if
		// enough pixels contributed
		// greater than 10
		// majority of values ahead are less than max
		if( peak_max > 40 && i > 4 && ahead_less > ahead_more ){
			break;
		}
	}
	
	// search around the index to find the peak
	float new_max = pupil_hist[peak_idx];
	int new_idx = peak_idx;
	int new_n = 5;
	for(int i=-new_n;i<new_n;i++){
		if( pupil_hist[peak_idx+i] > new_max ){
			new_idx = peak_idx+i;
			new_max = pupil_hist[new_idx];
		}
	}	
	
	// calculate weighted center of gravity
	float vmin = pupil_hist[new_idx];
	float vmax = pupil_hist[new_idx];
	for(int i=-new_n;i<new_n;i++){
		int val = pupil_hist[MAX(new_idx+i,0)];
		vmin = MIN(val,vmin);
		vmax = MAX(val,vmax);
	}
	new_n = 10;
	float avg_idx = 0;
	float this_weight, weight_tot = 0;
	for(int i=-new_n;i<new_n;i++){
		int val = pupil_hist[MAX(new_idx+i,0)];
		this_weight = val;//( min(max( ( val - vmin ) / ( vmax - vmin ), 0.0f),1.0f) );
		avg_idx += MAX(new_idx+i,0) * this_weight;
		weight_tot += this_weight;
	}
	if(weight_tot>0){
		avg_idx /= weight_tot;
	}

	avg_idx = min(max(avg_idx,0.0f),255.0f);
	
	if(DBUG) printf("thresh = %3.2f\n", avg_idx);
	
	float tmp_v;
	int tmp_idx;
	int low_idx = 0;
	for(int i=0;i<15;i++){
		tmp_idx = MIN((int)(avg_idx)+i,255);
		tmp_v = pupil_hist[tmp_idx];
		
		// stop condition for upper flood fill value
		if( tmp_v < pupil_hist[(int)avg_idx]*0.2 ||
		    tmp_v < iris_hist[tmp_idx]*0.85 ){
			*valley = min(max(i-1,5),20);
			break;
		}
	}
	
	*peak = new_idx;
//	*valley = low_idx-new_idx;
	
	return;
}

/**
 *	This method calculates the pupil decentration factor given pupil size.
 *	@param _px		Radius of query pupil in pixels.
 *	@param _offx	Destination of x pixel offset component.
 *	@param _offy	Destination of y pixel offset component.
 *	@date 2012/05/07
 */
void cvEyeTracker::getPupilOffset( float _px, float *_offx, float*_offy ){
	if( bLUTLoaded ){
		//		printf("**************************************\n");
		float head_vx = perkinje_org_pts_avg[1].x-perkinje_org_pts_avg[0].x;
		float head_vy = perkinje_org_pts_avg[1].y-perkinje_org_pts_avg[0].y;
		float head_t = atan2( head_vy, head_vx );
		
		//		printf("head: %3.3f\n", head_t );
		
		float tmp_t = pupil_lut_angle[0];
		float tmp_mag = pupil_lut_mag[0];
		int ret_idx = 0;
		float _v = 0;
		for(int i=0;i<LUT_N-1;i++){
			if(pupil_lut_size[i] <= _px && _px <= pupil_lut_size[i+1]){
				ret_idx = i;
				_v = (_px - pupil_lut_size[i])/(pupil_lut_size[i+1]-pupil_lut_size[i]);
				tmp_t   = pupil_lut_angle[i] + _v*(pupil_lut_angle[i+1]-pupil_lut_angle[i]);
				tmp_mag = pupil_lut_mag[i] + _v*(pupil_lut_mag[i+1]-pupil_lut_mag[i]);
				break;
			}
		}
		if( pupil_lut_size[LUT_N-1] < _px ) {
			tmp_t = pupil_lut_angle[LUT_N-1];
			tmp_mag = pupil_lut_mag[LUT_N-1];
			ret_idx = LUT_N-1;
		}
		
		tmp_t += head_t;
		
		float add_x = cos( tmp_t ) * tmp_mag;
		float add_y = sin( tmp_t ) * tmp_mag;		
		*_offx = add_x;
		*_offy = add_y;
		//		printf(" test : %3.3f -> (%3.3f %3.3f) [%d]@%1.3f\n", pupil_radius_avg, add_x, add_y, ret_idx, _v );
	}
}

/**
 *	This method loads the pupil decentration Look Up Table.
 *	The table consists of offset data ordered by pupil size.
 *	Offset data is listed as: <pupil size:px> <direction:rad> <magnitude:px>
 *	@date 2012/04/26
 */
bool cvEyeTracker::loadLUT(){
	
	pupil_lut_size  = NULL;
	pupil_lut_angle = NULL;
	pupil_lut_mag   = NULL;
	
	pupil_lut_size  = (float*)malloc(LUT_N*sizeof(float));
	pupil_lut_angle = (float*)malloc(LUT_N*sizeof(float));
	pupil_lut_mag   = (float*)malloc(LUT_N*sizeof(float));
	
	FILE *fp;
	if ((fp = fopen("/Users/jdewitt/Desktop/of_preRelease_v007_osx/apps/addonsExamples/opencvExample/bin/data/pupil_axis_lut.txt","r"))==NULL) { 
		//	if ((fp = fopen("pupil_axis_lut.txt","r"))==NULL) {
		printf("File not found.\n");
		return 1;
	} else {
		bLUTLoaded = true;
		
		float tmp_pd, tmp_angle, tmp_mag;
		int i = 0;
		while( fscanf(fp, "%f : %f %f", &tmp_pd, &tmp_angle, &tmp_mag) == 3 ){
			pupil_lut_size[i]  = tmp_pd;
			pupil_lut_angle[i] = tmp_angle;
			pupil_lut_mag[i]   = tmp_mag;
			printf("%d: (%3.4f) = (%3.4f) (%3.4f)\n", i, pupil_lut_size[i], pupil_lut_angle[i], pupil_lut_mag[i] );
			i++;
		}
		printf("loaded LUT file\n");
		fclose(fp);
	}
	
	
}

/**
 *	This method updates the perspective map.
 *	The resulting map transforms from camera space to an intermediate one.
 *	The resulting coordinates are in "screen" space.
 *	@date 2012/04/26
 */
void cvEyeTracker::calcPerspectiveMap( CvSeq *src_seq, CvSeq *dst_seq, CvMat* map_mat ){
	CvPoint2D32f src_pts[4];
	CvPoint2D32f dst_pts[4];
	
	cvCvtSeqToArray(src_seq, src_pts);
	cvCvtSeqToArray(dst_seq, dst_pts);
	
	float minx,miny;
	for(int i=0;i<4;i++){
		if(i==0){
			minx = src_pts[i].x;
			miny = src_pts[i].y;
		}
		minx = min(minx,src_pts[i].x);
		miny = min(miny,src_pts[i].y);
	}
	
	for(int i=0;i<4;i++){
		if(0)
		printf("pt: (%4.1f,%4.1f) (%4.1f,%4.1f)\n",
			   src_pts[i].x-minx, src_pts[i].y-miny,
			   dst_pts[i].x, dst_pts[i].y );
	}
	
	if(0)
	printf("res:(%4.1f,%4.1f)\n",
		   guess_pt.x, guess_pt.y );
	
	cvSetZero(map_mat);
	cvGetPerspectiveTransform(src_pts, dst_pts, map_mat);
	
}

//--------------------------------------------------------------
void cvEyeTracker::keyPressed(int key){
	
	float* src_ptr = (float*)
	(calibration_map.src_mat->data.ptr + calibration_map.selected.y * calibration_map.src_mat->step);
	
	switch(key){
		// toggle console writing
		case '`':
			bPrintDev = !bPrintDev;
			break;
		
		// toggle drawing dev screen
		case '\t':
			bDrawDevScreen = !bDrawDevScreen;
			bNeedRefresh = true;
			break;
		
		// toggle drawing eye image
		case 'a':
			bDrawEyeImage = !bDrawEyeImage;
			bNeedRefresh = true;
			break;
			
		// toggle drawing calibration target points
		case 'q':
			bDrawCalibrationTargetPoints = !bDrawCalibrationTargetPoints;
			bNeedRefresh = true;
			break;
		
		// toggle drawing calibration source points
		case 'w':
			bDrawCalibrationSourcePoints = !bDrawCalibrationSourcePoints;
			bNeedRefresh = true;
			break;
			
		// toggle drawing pupil ellipse
		case 'e':
			bDrawPupilEllipse = !bDrawPupilEllipse;
			bNeedRefresh = true;
			break;
			
		// toggle drawing perkinje points
		case 'd':
			bDrawPerkinjePoints = !bDrawPerkinjePoints;
			bNeedRefresh = true;
			break;
		
		// toggle drawing guess point
		case 'g':
			bDrawScreenGuessPoint = !bDrawScreenGuessPoint;
			bNeedRefresh = true;
			break;
		
		// ------
		
		// toggle recording
		case 'r':
			bToggleRec = !bToggleRec;
			bNeedRefresh = true;
			break;
			
		// toggle raw or with overlays?
		case 't':
			bToggleRecRaw = !bToggleRecRaw;
			bNeedRefresh = true;
			break;
			
		// toggle fullscreen
		case 'f':
			ofToggleFullscreen();
			break;
			
		// toggle image flipping
		case 'o':
			bFlipOrientation = !bFlipOrientation;
			bNeedRefresh = true;
			break;
			
		// toggle gaze cursor control
		case 'm':
			bUseEyeMouse = !bUseEyeMouse;
			bNeedRefresh = true;
			break;
			
		// ------
		
		// moving calibration point selector
		case OF_KEY_UP:
			calibration_map.selected.y = MAX( calibration_map.selected.y - 1, 0 );
			break;
		case OF_KEY_DOWN:
			calibration_map.selected.y = MIN( calibration_map.selected.y + 1, calibration_map.rows - 1 );
			break;
		case OF_KEY_LEFT:
			calibration_map.selected.x = MAX( calibration_map.selected.x - 1, 0 );
			break;
		case OF_KEY_RIGHT:
			calibration_map.selected.x = MIN( calibration_map.selected.x + 1, calibration_map.cols - 1);
			break;
		
		// register a calibration point
		case OF_KEY_RETURN:
			src_ptr[2*calibration_map.selected.x+0] = guess_pt_avg.x;
			src_ptr[2*calibration_map.selected.x+1] = guess_pt_avg.y;

			// printf("saved after %d,%d\n", calibration_map.selected.x, calibration_map.selected.y );
			// cvSave("src_pts.yml", calibration_map.src_mat);
			
			calcCalibrationMap( &calibration_map );
			bNeedRefresh = true;
			break;
			
		// pause play
		case ' ':
			if(cap_context.mode==CV_USE_VIDEO){
				bUseStep = !bUseStep;
				if(bUseStep){
					ofSetFrameRate(FPS_LO);
				} else {
					ofSetFrameRate(FPS_HI);
				}
				bNeedRefresh = true;
				if(bUseStep){ cap_context.vp.setPaused(true); } else { cap_context.vp.play(); }
			}
			break;
			
		// backwards a frame
		case '[':
			if(cap_context.mode==CV_USE_VIDEO){
				bUseStep = true;
				step_rdy--;
				
				// printf("step: %d\n",step_rdy);
				bNeedRefresh = true;
			}
			break;
			
		// forwards a frame
		case ']':
			if(cap_context.mode==CV_USE_VIDEO){
				bUseStep = true;
				step_rdy++;
				
				// printf("step: %d\n",step_rdy);
				bNeedRefresh = true;
			}
			break;
					
		// open video settings for camera
		case 's':
			if(cap_context.mode==CV_USE_LIVE_VIDEO){
				cap_context.vg.videoSettings();
			}
			break;
		
		// ------
			
		// clear ctmp_1 buffer
		case 'c':
			cvSetZero(ipl_arr[I_CTMP_1]);
			bNeedRefresh = true;
			break;
	}
}

//--------------------------------------------------------------
void cvEyeTracker::keyReleased(int key){

}

//--------------------------------------------------------------
void cvEyeTracker::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void cvEyeTracker::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void cvEyeTracker::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void cvEyeTracker::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void cvEyeTracker::windowResized(int w, int h){

}

//--------------------------------------------------------------
void cvEyeTracker::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void cvEyeTracker::dragEvent(ofDragInfo dragInfo){ 

}

