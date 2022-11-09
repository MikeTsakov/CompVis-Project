/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <cassert>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
				m_cameras(cs),
				m_height(2048), //2304
				m_step(32),
				m_clustercount(MAINCLUSTERCOUNT)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 3 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	initialize();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{

	
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height * 1.5;
	const int xR = m_height * 1.5;
	const int yL = -m_height * 1.5;
	const int yR = m_height * 1.5;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	m_persons.push_back(new Reconstructor::Person(0, Scalar(1.0f, 0.0f, 0.0f), std::vector<Mat>()));
	m_persons.push_back(new Reconstructor::Person(1, Scalar(0.0f, 1.0f, 0.0f), std::vector<Mat>()));
	m_persons.push_back(new Reconstructor::Person(2, Scalar(0.0f, 0.0f, 1.0f), std::vector<Mat>()));
	m_persons.push_back(new Reconstructor::Person(3, Scalar(1.0f, 0.0f, 1.0f), std::vector<Mat>()));
	for (int p = 0; p < m_persons.size(); p++)
	{
		m_persons[p]->path = std::vector<Point2d>(m_cameras[0]->getFramesAmount());
		m_persons[p]->path_computed = std::vector<bool>(m_cameras[0]->getFramesAmount());
		for (int i = 0; i < m_persons[p]->path.size(); i++)
		{
			m_persons[p]->path[i].x = INFINITY;
			m_persons[p]->path[i].y = INFINITY;
			m_persons[p]->path_computed[i] = 0;
		}
	}

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels... \n";
	
	if (General::fexists("data/voxels.txt")) {
		std::ifstream in("data/voxels.txt");
		for (int i = 0; i < m_voxels_amount; i++) {
			Voxel* vox = new Voxel();
			in >> *vox;
			m_voxels.push_back(vox);
		}
		return;
	}

	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(auto) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	std::ofstream out("data/voxels.txt");
	for (int i = 0; i < m_voxels_amount; i++) {
		out << *m_voxels[i];
	}

	cout << "done!" << endl;
}

void Reconstructor::computeClusterHistogram(Cluster* cluster, int& ignored_samples, int cam_id)
{
	if (cam_id == -1)
	{
		cam_id = cluster->best_camid;
	}
	Mat frame, frame_hsv, histogram, histogram_data, frame_b_blurred;
	vector<Mat> frame_split;

	// histogram parameters
	const int h_bins[] = { BINCOUNT0, BINCOUNT1 };
	const int h_channels[] = { 0, 1 };
	float h_range0[] = { RANGEMIN0, RANGEMAX0 };
	float h_range1[] = { RANGEMIN1, RANGEMAX1 };
	const float* h_range_p[] = { h_range0, h_range1 };

	frame = m_cameras[cluster->best_camid]->getFrame();

	//std::stringstream ss;
	//ss << "C:/Users/tniko/Desktop/Uni/Computer Vision/assignment_3/VoxelReconstruction/data/c" << cam_id << ".png";
	//imwrite(ss.str(), frame);

	frame_b_blurred = frame.clone();
	bilateralFilter(frame, frame_b_blurred, BLURDIAMETER, BLURSIGMACOLOR, BLURSIGMASPACE);
	frame = frame_b_blurred;
	// split into HSV vectors
	cvtColor(frame, frame_hsv, CV_BGR2HSV);
	split(frame_hsv, frame_split);
	// fill the data array with pixel values for each visible voxel in camera 'c'
	histogram_data = Mat(cluster->data.size(), 1, CV_8UC2);
	for (int v = 0; v < cluster->data.size(); v++)
	{
		Reconstructor::Voxel* voxel = cluster->data[v];
		uchar s_val = frame_split[1].at<uchar>(voxel->camera_projection[cluster->best_camid]);
		uchar v_val = frame_split[2].at<uchar>(voxel->camera_projection[cluster->best_camid]);
		//uchar h_val = frame_split[0].at<uchar>(voxel->camera_projection[c]);
		histogram_data.at<Vec2b>(v) = Vec2b(s_val, v_val);
	}
	calcHist(&histogram_data, 1, h_channels, Mat(), histogram, 2, h_bins, h_range_p, true, false);
	ignored_samples = histogram_data.rows - (int) sum(histogram).val[0];
	if (true)
	{
		double hist_sum = sum(histogram).val[0];
		histogram = histogram / hist_sum;
	}
	else
	{
		normalize(histogram, histogram, 1.0, 0.0, NORM_L1);
	}
	cluster->histogram = histogram;
}

std::vector<Reconstructor::Person*> Reconstructor::matchHistogram(
	cv::Mat histogram, std::vector<double>& out_distances)
{
	// output variables
	vector<Reconstructor::Person*> matches;
	vector<double> match_distances;
	// method variables
	vector<double> histogram_distances;
	vector<int> histogram_indices;
	for (int p = 0; p < m_persons.size(); p++)
	{
		double dst_min_h = INFINITY;
		int h_idx = -1;
		for (int h = 0; h < m_persons[p]->histograms.size(); h++)
		{
			Mat p_histogram = m_persons[p]->histograms[h];
			double dst = compareHist(histogram, p_histogram, CV_COMP_CHISQR);
			if (dst < dst_min_h)
			{
				h_idx = h;
				dst_min_h = dst;
			}
		}
		histogram_distances.push_back(dst_min_h);
		histogram_indices.push_back(h_idx);
	}
	for (int p = 0; p < histogram_distances.size(); p++)
	{
		float dst_min = INFINITY;
		int d_idx = -1;
		for (int d = 0; d < histogram_distances.size(); d++)
		{
			if (histogram_distances[d] < dst_min)
			{
				dst_min = histogram_distances[d];
				d_idx = d;
			}
		}
		matches.push_back(m_persons[d_idx]);
		match_distances.push_back(histogram_distances[d_idx]);
		histogram_distances[d_idx] = INFINITY;
	}
	out_distances = match_distances;
	return matches;
}

// uniquely matches each cluster to one person
void Reconstructor::matchClusters(std::vector<Cluster*>& clusters)
{
	vector<vector<double>> all_cluster_distances;
	vector<vector<Person*>> all_cluster_matches;

	// reset person match flags
	for (int p = 0; p < m_persons.size(); p++) { m_persons[p]->matched = false; }
	for (int c = 0; c < clusters.size(); c++) { clusters[c]->matched = false; }

	// compute histogram match data for each cluster
	for (int c = 0; c < clusters.size(); c++)
	{
		vector<double> cluster_distances;
		Cluster* cluster = clusters[c];
		all_cluster_matches.push_back(matchHistogram(cluster->histogram, cluster_distances));
		all_cluster_distances.push_back(cluster_distances);
	}

	// find closest person for each cluster
	
	bool mismatch = true;
	while (mismatch)
	{
		mismatch = false;
		double min_dst = INFINITY;
		int c_idx = -1;
		Person* p_match = nullptr;
		for (int c = 0; c < all_cluster_matches.size(); c++)
		{
			if (!clusters[c]->matched)
			{
				for (int o = 0; o < all_cluster_matches[c].size(); o++)
				{
					if (!all_cluster_matches[c][o]->matched)
					{
						if (all_cluster_distances[c][o] < min_dst)
						{
							mismatch = true;
							min_dst = all_cluster_distances[c][o];
							p_match = all_cluster_matches[c][o];
							c_idx = c;
						}
					}
				}
			}
		}
		if (mismatch)
		{
			p_match->matched = true;
			clusters[c_idx]->matched = true;
			clusters[c_idx]->person = p_match;
		}
	}
}

// once matches are found, apply processing to the matches such as path smoothing and sharp label switching
void Reconstructor::processMatches()
{
#if POSITIONESTIMATION
	if (m_current_frame > PATHESTIMATIONPAST + 1)
	{
		for (int c = 0; c < m_clusters.size(); c++)
		{
			Cluster* cluster = m_clusters[c];
			Person* closest_person = nullptr;
			double min_dst = INFINITY;
			for (int p = 0; p < m_persons.size(); p++)
			{
				Person* person = m_persons[p];
				Point2f estimated_location = person->estimateLocation(m_current_frame);
				//if (person->mismatch_counter > MISMATCHTHRESHOLD)
				//{
				//	estimated_location = person->estimateLocation(
				//		m_current_frame - MISMATCHTHRESHOLD - person->grace_counter
				//	);
				//}
				//printf("%.2f, %.2f\n", estimated_location.x, estimated_location.y);
				double dst = norm(cluster->center - estimated_location);
				if (dst < min_dst)
				{
					min_dst = dst;
					closest_person = person;
				}
			}
			if (cluster->person != closest_person)
			{
				//if (closest_person->mismatch_counter < MISMATCHTHRESHOLD)
				//{
					closest_person->mismatch_counter++;
					cluster->person = closest_person;
				//}
				//else
				//{
				//	if (closest_person->grace_counter >= GRACEPERIOD)
				//	{
				//		closest_person->grace_counter = 0;
				//		closest_person->mismatch_counter = 0;
				//		printf("GRACE!");
				//	}
				//	else
				//	{
				//		closest_person->grace_counter++;
				//	}
				//}
			}
		}
	}
#endif

#if WRONGMATCHDETECTION
	// detect very long changes in new path point
	for (int c = 0; c < m_clusters.size(); c++)
	{
		m_clusters[c]->person->wrong_match = false;
		if (m_current_frame > 0 && m_clusters[c]->person->path_computed[m_current_frame - 1])
		{
			double diff = norm(Point2d(m_clusters[c]->center) - m_clusters[c]->person->path[m_current_frame - 1]);
			if ( diff > PATHDISTANCETHRESHOLD )
			{
				m_clusters[c]->person->wrong_match = true;
				m_clusters[c]->person->wrong_match_distance = true;
			}
		}
	}
#endif

#if MATCHSWAPPING
	// swap wrong matches
	for (int c1 = 0; c1 < m_clusters.size(); c1++)
	{
		for (int c2 = 0; c2 < m_clusters.size(); c2++)
		{
			if (c1 == c2) continue;
			Person* person1 = m_clusters[c1]->person;
			Person* person2 = m_clusters[c2]->person;
			if (person1->wrong_match && person2->wrong_match &&
				abs(person1->wrong_match_distance - person2->wrong_match_distance) < MISWAPDIFFDISTANCE)
			{
				printf("%d: Swapping people %d and %d!\n", m_current_frame, person1->id, person2->id);
				m_clusters[c1]->person = person2;
				m_clusters[c2]->person = person1;
				person1->wrong_match = false;
				person2->wrong_match = false;
			}
		}
	}
#endif

	// reset person match flags
	for (int p = 0; p < m_persons.size(); p++) { m_persons[p]->matched = false; }
	// finally add cluster center to "correct" person
	for (int c = 0; c < m_clusters.size(); c++)
	{
		Person* person = m_clusters[c]->person;
		person->path[m_current_frame] = m_clusters[c]->center;
		person->path_computed[m_current_frame] = true;
		person->matched = true;
	}
	// if person was not matched, estimate his position
	for (int p = 0; p < m_persons.size(); p++) {
		Person* person = m_persons[p];
		if (!person->matched) {
			person->path[m_current_frame] = person->estimateLocation(m_current_frame);
			person->path_computed[m_current_frame] = true;
			person->matched = true;
		}
	}
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	//m_voxel_labels.clear();
	m_visible_voxels.clear();
	m_surface_voxels.clear();
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for schedule(auto) private(v) shared(visible_voxels)
	for (v = 0; v < (int) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];
				//cout << point << endl;

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255)
				{
					++camera_counter;
				}
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size() && voxel->z > SHIRTABSOLUTELOWERZ)
		{
#pragma omp critical
			//push_back is critical
			visible_voxels.push_back(voxel);
		}
	}

	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

#if VOXELCOLORING
	// for each pixel store a pointer to the closest voxel to it
	std::vector<std::vector<Voxel*>> surface_voxel_pointers;
	// for each pixel store the distance to the closest voxel to it
	std::vector<Mat> surface_voxel_distances;
	int f_width, f_height;
	for (int i = 0; i < m_cameras.size(); i++)
	{
		f_width = m_cameras[i]->getFrame().cols;
		f_height = m_cameras[i]->getFrame().rows;
		int pixel_count =  f_width * f_height;
		surface_voxel_pointers.push_back(std::vector<Voxel*>(pixel_count, nullptr));
		surface_voxel_distances.push_back(Mat(m_cameras[i]->getFrame().rows, m_cameras[i]->getFrame().cols, CV_32F));
		surface_voxel_distances[surface_voxel_distances.size() - 1].setTo(INFINITY);
	}

	// for each visible voxel, check if it is the closest voxel to any camera
	for (int v = 0; v < m_visible_voxels.size(); v++)
	{
		Voxel* voxel = m_visible_voxels[v]; 
		// reset flags from previous frame
		voxel->closest_to_camera[0] = 0;
		voxel->closest_to_camera[1] = 0;
		voxel->closest_to_camera[2] = 0;
		voxel->closest_to_camera[3] = 0;
		// check if voxel is closest to each camera at projected camera position
		for (int c = 0; c < m_cameras.size(); c++)
		{
			const Point2i point = voxel->camera_projection[c];
			Point3f cam2voxel = m_cameras[c]->getCameraLocation() - Point3f((float)voxel->x, (float)voxel->y, (float)voxel->z);
			float distance2 = cam2voxel.dot(cam2voxel);
			if (distance2 < surface_voxel_distances[c].at<float>(point))
			{
				Voxel* oldVoxel = surface_voxel_pointers[c][point.y * f_width + point.x];
				if (oldVoxel != nullptr)
				{
					oldVoxel->closest_to_camera[c] = 0;
				}
				surface_voxel_pointers[c][point.y * f_width + point.x] = voxel;
				surface_voxel_distances[c].at<float>(point) = distance2;
				voxel->closest_to_camera[c] = 1;
			}
		}
	}
#endif

	if (VOXELCLUSTERING)
	{
		std::vector<Cluster*> clusters = computeClusters();
		processClusters(clusters);
		isolateShirts(clusters);
		computeSuitableClusterCameras(clusters); // find a suitable camera for each cluster
		// compute cluster histograms
		for (int c = 0; c < clusters.size(); c++)
		{
			int ignored_samples;
			computeClusterHistogram(clusters[c], ignored_samples);
			//printf("	%d: Ignored %d/%d values.\n", c, ignored_samples, clusters[c]->data.size());
		}
		m_clusters = clusters;
		if (m_initialized)
		{
			matchClusters(m_clusters);
			processMatches();
		}
	}
}

void Reconstructor::computeSuitableClusterCameras(std::vector<Reconstructor::Cluster*>& clusters)
{
	if (!m_initialized)
	{
		for (int c = 0; c < clusters.size(); c++)
		{
			clusters[c]->best_camid = m_initialize_step;
		}
		return;
	}

	// comment if necessary \/
	//std::vector<std::vector<Vec2f>> centroid_vectors; // for each camera get direction vector to each cluster
	//for (int cam = 0; cam < m_cameras.size(); cam++)
	//{
	//	Camera* camera = m_cameras[cam];
	//	Point2f cam_pos = Point2f(camera->getCameraLocation().x, camera->getCameraLocation().y);
	//	centroid_vectors.push_back(std::vector<Vec2f>());
	//	for (int c = 0; c < clusters.size(); c++)
	//	{
	//		double length = norm(clusters[c]->center - cam_pos);
	//		centroid_vectors[cam].push_back(Vec2f(clusters[c]->center - cam_pos) / length);
	//	}
	//}
	printf("--- %d ---\n", m_current_frame);
	for (int c = 0; c < clusters.size(); c++)
	{
		std::vector<double> smallest_angle_per_cam;
		for (int cam = 0; cam < m_cameras.size(); cam++)
		{
			double smallest_angle = -INFINITY;
			for (int c1 = 0; c1 < clusters.size(); c1++)
			{
				if (c == c1) continue;
				Point2f cameraXY = Point2f(m_cameras[cam]->getCameraLocation().x, m_cameras[cam]->getCameraLocation().y);
				Vec2f diffc0 = clusters[c]->center - cameraXY;
				Vec2f diffc1 = clusters[c1]->center - cameraXY;
				diffc0 = diffc0 / sqrtf(diffc0.dot(diffc0));
				diffc1 = diffc1 / sqrtf(diffc1.dot(diffc1));
				double angle = diffc0.dot(diffc1);
				if (angle > smallest_angle) // smallest angle has largest dot product result
				{
					smallest_angle = angle;
				}
			}
			smallest_angle_per_cam.push_back(smallest_angle);
		}
		double largest_angle = INFINITY;
		for (int i = 0; i < smallest_angle_per_cam.size(); i++)
		{
			if (smallest_angle_per_cam[i] < largest_angle) // largest angle has smallest dot product result
			{
				largest_angle = smallest_angle_per_cam[i];
				clusters[c]->best_camid = i;
			}
		}
		//printf("%d: Cam %d with angle %.4f\n", c, clusters[c]->best_camid + 1, largest_angle);
	}

	//for (int c = 0; c < clusters.size(); c++)
	//{
	//	std::vector<double> smallest_angle_per_cam;
	//	for (int cam = 0; cam < centroid_vectors.size(); cam++)
	//	{
	//		double smallest_angle = INFINITY;
	//		for (int c1 = 0; c1 < centroid_vectors[cam].size(); c1++)
	//		{
	//			if (c == c1) continue;
	//			double angle = centroid_vectors[cam][c].dot(centroid_vectors[cam][c1]);
	//			if (angle < smallest_angle)
	//			{
	//				smallest_angle = angle;
	//			}
	//		}
	//		smallest_angle_per_cam.push_back(smallest_angle);
	//	}
	//	double largest_angle = -INFINITY;
	//	printf("%d: Comparing angles...\n", c);
	//	for (int i = 0; i < smallest_angle_per_cam.size(); i++)
	//	{
	//		printf("%.2f ", smallest_angle_per_cam[i]);
	//		if (smallest_angle_per_cam[i] > largest_angle)
	//		{
	//			largest_angle = smallest_angle_per_cam[i];
	//			clusters[c]->best_camid = i;
	//		}
	//	}
	//	printf("\Angle: %.4f", largest_angle);
	//	printf("%d: Best cam is %d\n", c, clusters[c]->best_camid);
	//}
}


std::vector<Reconstructor::Cluster*> Reconstructor::computeClusters()
{
	Mat labels, centers;
	Mat data = Mat(m_visible_voxels.size(), 2, CV_32F);
	std::vector<Reconstructor::Cluster*> out_clusters;
	for (int i = 0; i < m_clustercount; i++)
	{
		out_clusters.push_back(new Cluster());
	}
	for (int i = 0; i < data.rows; i++) {
		Voxel* voxel = m_visible_voxels[i];
		data.at<float>(i, 0) = (float) voxel->x;
		data.at<float>(i, 1) = (float) voxel->y;
	}
	double compactness = kmeans(data, m_clustercount, labels, TermCriteria(CV_TERMCRIT_ITER & CV_TERMCRIT_EPS, 10, 1.0),
		3, KMEANS_PP_CENTERS, centers);
	// populate clusters with corresponding voxels
	for (int i = 0; i < data.rows; i++) {
		out_clusters[labels.at<int>(i)]->data.push_back(m_visible_voxels[i]);
	}
	for (int i = 0; i < centers.rows; i++)
	{
		out_clusters[i]->center = centers.at<cv::Point2f>(i);
	}
	return out_clusters;
}

void Reconstructor::processClusters(std::vector<Cluster*>& clusters)
{
	bool detected = false;
	int first_id = -1, second_id = -1;
	Cluster* first = nullptr;
	Cluster* second = nullptr;
	while (true)
	{
		detected = false;
		// detects a close enough pair
		for (int c0 = 0; c0 < clusters.size(); c0++)
		{
			for (int c1 = 0; c1 < clusters.size(); c1++)
			{
				if (c0 == c1) continue;
				if (norm(clusters[c0]->center - clusters[c1]->center) < m_centroid_threshold)
				{
					detected = true;
					first = clusters[c0];
					second = clusters[c1];
					first_id = c0;
					second_id = c1;
				}
			}
		}

		if (!detected) { break; }
		//printf("Norm: %f, f_id %d, s_id %d\n", norm(clusters[first_id]->center - clusters[second_id]->center),
		//	first_id, second_id);
		Cluster* merger = new Cluster();
		merger->center = (first->center + second->center) / 2.0;
		merger->data.insert(merger->data.begin(), first->data.begin(), first->data.end());
		merger->data.insert(merger->data.end(), second->data.begin(), second->data.end());
		if (first_id > second_id) { std::swap(first_id, second_id); }
		clusters.erase(clusters.begin() + second_id);
		clusters.erase(clusters.begin() + first_id);
		clusters.push_back(merger);
	}

	// clamp the number of clusters to 4, keeping the largest ones
	while (clusters.size() > 4)
	{
		int min_size = INT_MAX;
		int c_idx = -1; // smallest cluster index
		for (int c = 0; c < clusters.size(); c++)
		{
			if (clusters[c]->data.size() < min_size)
			{
				min_size = clusters[c]->data.size();
				c_idx = c;
			}
		}
		//printf("removed cluster of size: %d\n", clusters[c_idx]->data.size());
		clusters.erase(clusters.begin() + c_idx);
	}

	// filter out small clusters
	for (int c = clusters.size() - 1; c >= 0; c--)
	{
		if (clusters[c]->data.size() < m_min_cluster_size)
		{
			clusters.erase(clusters.begin() + c);
		}
	}

	assert(clusters.size() <= 4);
}

void Reconstructor::isolateShirts(std::vector<Reconstructor::Cluster*>& clusters)
{
	int lower_z[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
	int upper_z[4] = { INFINITY, INFINITY, INFINITY, INFINITY };
	int max_z[4] = { -1, -1, -1, -1 };
	for (int c = 0; c < clusters.size(); c++)
	{
		for (int v = clusters[c]->data.size() - 1; v >= 0; v--)
		{
			Voxel* voxel = clusters[c]->data[v];
			if (voxel->z > max_z[c])
			{
				max_z[c] = (int) voxel->z;
			}
		}
		lower_z[c] = max_z[c] * SHIRTLOWERZ;
		upper_z[c] = max_z[c] * SHIRTUPPERZ;
	}
	
	for (int c = 0; c < clusters.size(); c++)
	{
		for (int v = clusters[c]->data.size() - 1; v >= 0; v--)
		{
			Voxel* voxel = clusters[c]->data[v];
			if (voxel->z < lower_z[c] || voxel->z > upper_z[c])
			{
				clusters[c]->data.erase(clusters[c]->data.begin() + v);
			}
		}
	}
}

cv::Point2d Reconstructor::Person::estimateLocation(int frame)
{
	Point2d output = Point2d(0, 0);
	if (frame > PATHESTIMATIONPAST + 1 && frame < path.size() - 1)
	{
		std::vector<Point2d> diffs;
		for (int i = frame - 1; i > frame - PATHESTIMATIONPAST + 1; i--)
		{
			Point2d last_point = path[i];
			Point2d last_last_point = path[i - 1];
			diffs.push_back(last_point - last_last_point);
		}
		for (int i = 0; i < diffs.size(); i++)
		{
			output += diffs[i];
		}
		output /= (int)diffs.size();
		output = output + path[frame - 1];
	}
	return output;
}

} /* namespace nl_uu_science_gmt */
