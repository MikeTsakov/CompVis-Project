/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stddef.h>
#include <string>

#include "../utilities/General.h"
#include <iostream>

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
	Scene3DRenderer::Scene3DRenderer(
		Reconstructor &r, const vector<Camera*> &cs) :
		m_reconstructor(r),
		m_cameras(cs),
		m_num(4),
		m_sphere_radius(1850)
	{
		m_width = 640;
		m_height = 480;
		m_quit = false;
		m_paused = true;
		m_rotate = false;
		m_camera_view = true;
		m_show_volume = true;
		m_show_grd_flr = true;
		m_show_cam = true;
		m_show_org = true;
		m_show_arcball = false;
		m_show_info = true;
		m_fullscreen = false;

		// Read the checkerboard properties (XML)
		FileStorage fs;
		fs.open(m_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
		if (fs.isOpened())
		{
			fs["CheckerBoardWidth"] >> m_board_size.width;
			fs["CheckerBoardHeight"] >> m_board_size.height;
			fs["CheckerBoardSquareSize"] >> m_square_side_len;
		}
		fs.release();

		m_current_camera = 0;
		m_previous_camera = 0;

		m_number_of_frames = m_cameras.front()->getFramesAmount();
		m_current_frame = 0;
		m_previous_frame = -1;

		const int D = 0;
		m_d_threshold = D;
		m_h_threshold = HSLIDER;
		m_ph_threshold = HSLIDER;
		m_s_threshold = SSLIDER;
		m_ps_threshold = SSLIDER;
		m_v_threshold = VSLIDER;
		m_pv_threshold = VSLIDER;
		m_morph_open_size = 1;
		m_morph_size = 1;
		m_hole_min_area = 0;
		m_hole_min_max_area = 80;

		m_custom_threshold = MINCLUSTERSIZE;
		m_pcustom_threshold = MINCLUSTERSIZE;

		createTrackbar("Frame", VIDEO_WINDOW, &m_current_frame, m_number_of_frames - 2);
		createTrackbar("Centroid T", VIDEO_WINDOW, &m_custom_threshold, 5000);
#if IMPROVEDBGMODEL
		//createTrackbar("D", VIDEO_WINDOW, &m_d_threshold, 255);
		createTrackbar("Variance T", VIDEO_WINDOW, &m_h_threshold, 350);
		createTrackbar("Shadow T", VIDEO_WINDOW, &m_s_threshold, 350);
		//createTrackbar("Morph", VIDEO_WINDOW, &m_morph_size, 10);
		createTrackbar("Hole Ratio", VIDEO_WINDOW, &m_hole_min_area, m_hole_min_max_area);
#else
		createTrackbar("H", VIDEO_WINDOW, &m_h_threshold, 255);
		createTrackbar("S", VIDEO_WINDOW, &m_s_threshold, 255);
		createTrackbar("V", VIDEO_WINDOW, &m_v_threshold, 255);
#endif
		createFloorGrid();
		setTopView();

		std::stringstream ss;
		int curr_shadow_t = 0, curr_variance_t = 0, curr_hole_t = 0;
		int variance_t_max = 300, shadow_t_max = 100, hole_t_max = 80;
		int variance_step = 50, shadow_step = 50, hole_step = 50;

#pragma omp parallel for
	for (int i = 0; i < 4; i++)
	{
		ss.str("");
		ss << "data\\cam" << i + 1 << "\\" << i + 1 << "_gs.png";
		Mat cam_reference = imread(ss.str());
		cvtColor(cam_reference, cam_reference, CV_BGR2GRAY);

		Camera* cam = m_cameras[i];
		cam->setVideoFrame(0);
		cam->advanceVideoFrame();

		uint minval = INT_MAX;
		Mat cam_xor, cam_foreground;
		Scalar tmp;
		int best_v_t = 0, best_s_t = 0, best_h_t = 0;
		for (int x = 0; x < variance_t_max; x += variance_step)
		{
			printf("Cam %d | X: %d/%d\n", i, x, variance_t_max);
			for (int y = 0; y < shadow_t_max; y += shadow_step)
			{
				for (int z = 0; z < hole_t_max; z += hole_step)
				{
					cam->m_variance_threshold = x;
					cam->m_shadow_threshold = y;
					cam->m_hole_threshold = z;
					processForeground(cam);
					cam_foreground = cam->getForegroundImage();
					bitwise_xor(cam_foreground, cam_reference, cam_xor);
					tmp = sum(cam_xor);

					if (tmp.val[0] < minval)
					{
						m_h_threshold = x;
						m_s_threshold = y;
						m_hole_min_area = z;
						best_v_t = x;
						best_s_t = y;
						best_h_t = z;
						minval = (int)tmp.val[0];
					}
				}
			}
		}

		cam->m_variance_threshold = best_v_t;
		cam->m_shadow_threshold = best_s_t;
		cam->m_hole_threshold = best_h_t;

		printf("Cam %d params: %d, %d, %d\n",
			i, cam->m_variance_threshold, cam->m_shadow_threshold, cam->m_hole_threshold);
	}
}

/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < m_floor_grid.size(); ++f)
		for (size_t g = 0; g < m_floor_grid[f].size(); ++g)
			delete m_floor_grid[f][g];
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	m_reconstructor.m_current_frame = m_current_frame;
	m_reconstructor.m_min_cluster_size = m_custom_threshold;
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			m_cameras[c]->advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			m_cameras[c]->getVideoFrame(m_current_frame);
		}
		assert(m_cameras[c] != NULL);
		processForeground(m_cameras[c]);
	}
	return true;
}

/**
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(
		Camera* camera)
{
	assert(!camera->getFrame().empty());
#if IMPROVEDBGMODEL
	Mat morph_open_element =getStructuringElement(MORPH_ELLIPSE,
		Size(m_morph_open_size + 1, m_morph_open_size + 1), Point(m_morph_open_size, m_morph_open_size));
	Mat morph_element = getStructuringElement(MORPH_ELLIPSE,
		Size(m_morph_size + 1, m_morph_size + 1), Point(m_morph_size, m_morph_size));
	Mat pre_mask = camera->getFrame();
	Mat foreground = Mat::Mat(pre_mask.size[0], pre_mask.size[1], CV_8UC3, Scalar(0, 0, 0));
	camera->SetBSParameters(camera->m_shadow_threshold, camera->m_variance_threshold);
	camera->GetForegroundFromBS(pre_mask, pre_mask);

	// remove shadows from Gaussian-mixtured pre-mask
	threshold(pre_mask, pre_mask, 200, 255, THRESH_BINARY);

	// morph open to remove noise in mask
	erode(pre_mask, pre_mask, morph_open_element);
	dilate(pre_mask, pre_mask, morph_open_element);

	// dilate mask to enclose any cuts in the boundary
	dilate(pre_mask, pre_mask, morph_element);

	// detect contours in mask
	std::vector<std::vector<Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(pre_mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	// find largest-area contour
	float maxArea = -INFINITY;
	float maxIdx = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		float area = contourArea(contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			maxIdx = i;
		}
	}

	// draw (white-filled) largest-area contour on the mask
	drawContours(foreground, contours, maxIdx, Scalar(255, 255, 255), FILLED);

	// draw (black-filled) child contours above a certain area ratio on the mask
	int currcidx = hierarchy[maxIdx][2];
	while (true)
	{
		if (currcidx < 0) break;
		if ((contourArea(contours[currcidx]) / maxArea) * m_hole_min_max_area * 50.0 > camera->m_hole_threshold)
		{
			drawContours(foreground, contours, currcidx, Scalar(0, 0, 0), FILLED);
		}
		currcidx = hierarchy[currcidx][0];
	}

	// compensate for prior dilation by same amount
	erode(foreground, foreground, morph_element);

	// convert to GRAYSCALE and apply mask
	cvtColor(foreground, foreground, CV_BGR2GRAY);
	camera->setForegroundImage(foreground);
#else
	Mat hsv_image;
	cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	// Background subtraction H
	Mat tmp, foreground, background;
	absdiff(channels[0], camera->getBgHsvChannels().at(0), tmp);
	threshold(tmp, foreground, m_h_threshold, 255, CV_THRESH_BINARY);

	// Background subtraction S
	absdiff(channels[1], camera->getBgHsvChannels().at(1), tmp);
	threshold(tmp, background, m_s_threshold, 255, CV_THRESH_BINARY);
	bitwise_and(foreground, background, foreground);

	// Background subtraction V
	absdiff(channels[2], camera->getBgHsvChannels().at(2), tmp);
	threshold(tmp, background, m_v_threshold, 255, CV_THRESH_BINARY);
	bitwise_or(foreground, background, foreground);

	Mat morph_single_element = getStructuringElement(MORPH_ELLIPSE,
		Size(2, 2), Point(1, 1));
	Mat morph_double_element = getStructuringElement(MORPH_ELLIPSE,
		Size(4, 4), Point(2, 2));

	// fill holes
	dilate(foreground, foreground, morph_double_element);
	erode(foreground, foreground, morph_double_element);
	// delete singles
	erode(foreground, foreground, morph_double_element);
	dilate(foreground, foreground, morph_double_element);
	// shrink image a bit
	erode(foreground, foreground, morph_double_element);
	camera->setForegroundImage(foreground);
#endif
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera]->getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera]->getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera]->getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	m_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.push_back(new Point3i(-size * m_num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.push_back(new Point3i(x, size * m_num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.push_back(new Point3i(size * m_num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.push_back(new Point3i(x, -size * m_num, z_offset));

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}

} /* namespace nl_uu_science_gmt */
