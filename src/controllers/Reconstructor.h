/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>

#include "Camera.h"

namespace nl_uu_science_gmt
{

class Reconstructor
{
public:
	/*
	 * Voxel structure
	 * Represents a 3D pixel in the half space
	 */
	struct Voxel
	{
		int x, y, z;                               // Coordinates
		cv::Scalar color;                          // Color
		std::vector<cv::Point> camera_projection;  // Projection location for camera[c]'s FoV (2D)
		std::vector<int> valid_camera_projection;  // Flag if camera projection is in camera[c]'s FoV
		
		bool operator==(const Voxel &obj)
		{
			return (x == obj.x) && (y == obj.y) && (z == obj.z) &&
				(camera_projection[0].x == obj.camera_projection[0].x) &&
				(camera_projection[0].y == obj.camera_projection[0].y) &&
				(camera_projection[0].x == obj.camera_projection[1].x) &&
				(camera_projection[0].y == obj.camera_projection[1].y) &&
				(camera_projection[0].x == obj.camera_projection[2].x) &&
				(camera_projection[0].y == obj.camera_projection[2].y) &&
				(camera_projection[0].x == obj.camera_projection[3].x) &&
				(camera_projection[0].y == obj.camera_projection[4].y) &&
				(valid_camera_projection[0] == obj.valid_camera_projection[0]) &&
				(valid_camera_projection[1] == obj.valid_camera_projection[1]) &&
				(valid_camera_projection[2] == obj.valid_camera_projection[2]) &&
				(valid_camera_projection[3] == obj.valid_camera_projection[3]);
		}

		/*
		* Write the member variables to stream objects
		*/
		friend std::ostream & operator << (std::ostream &out, const Voxel &obj)
		{
			out << obj.x << "\n"
				<< obj.y << "\n"
				<< obj.z << "\n"
				<< obj.camera_projection[0].x << "\n" 
				<< obj.camera_projection[0].y << "\n" 
				<< obj.camera_projection[1].x << "\n" 
				<< obj.camera_projection[1].y << "\n" 
				<< obj.camera_projection[2].x << "\n" 
				<< obj.camera_projection[2].y << "\n" 
				<< obj.camera_projection[3].x << "\n" 
				<< obj.camera_projection[3].y << "\n" 
				<< obj.valid_camera_projection[0] << "\n" 
				<< obj.valid_camera_projection[1] << "\n" 
				<< obj.valid_camera_projection[2] << "\n"
				<< obj.valid_camera_projection[3] << std::endl;
			return out;
		}
		/*
		* Read data from stream object and fill it in member variables
		*/
		friend std::istream & operator >> (std::istream &in, Voxel &obj)
		{
			in >> obj.x;
			in >> obj.y;
			in >> obj.z;
			int x, y;
			in >> x;
			in >> y;
			obj.camera_projection.push_back(cv::Point(x, y));
			in >> x;
			in >> y;
			obj.camera_projection.push_back(cv::Point(x, y));
			in >> x;
			in >> y;
			obj.camera_projection.push_back(cv::Point(x, y));
			in >> x;
			in >> y;
			obj.camera_projection.push_back(cv::Point(x, y));
			in >> x;
			obj.valid_camera_projection.push_back(x);
			in >> x;
			obj.valid_camera_projection.push_back(x);
			in >> x;
			obj.valid_camera_projection.push_back(x);
			in >> x;
			obj.valid_camera_projection.push_back(x);
			return in;
		}
	};

private:
	const std::vector<Camera*> &m_cameras;  // vector of pointers to cameras
	const int m_height;                     // Cube half-space height from floor to ceiling
	const int m_step;                       // Step size (space between voxels)

	std::vector<cv::Point3f*> m_corners;    // Cube half-space corner locations

	size_t m_voxels_amount;                 // Voxel count
	cv::Size m_plane_size;                  // Camera FoV plane WxH

	std::vector<Voxel*> m_voxels;           // Pointer vector to all voxels in the half-space
	std::vector<Voxel*> m_visible_voxels;   // Pointer vector to all visible voxels

	void initialize();

public:
	Reconstructor(
			const std::vector<Camera*> &);
	virtual ~Reconstructor();

	void update();

	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return m_visible_voxels;
	}

	const std::vector<Voxel*>& getVoxels() const
	{
		return m_voxels;
	}

	void setVisibleVoxels(
			const std::vector<Voxel*>& visibleVoxels)
	{
		m_visible_voxels = visibleVoxels;
	}

	void setVoxels(
			const std::vector<Voxel*>& voxels)
	{
		m_voxels = voxels;
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return m_corners;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
