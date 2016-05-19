/**
 *	\file
 */


#pragma once


#include <Eigen/Dense>
#include <dynfu/depth_device.hpp>
#include <cstddef>
#include <vector>


namespace dynfu {
	
	
	/**
	 *	A mock \ref depth_device which simply returns
	 *	a predefined sequence of frames and throws
	 *	an exception if it is invoked without a frame
	 *	to return.
	 */
	class mock_depth_device final : public depth_device {
		
		
		private:
		
		
			std::vector<std::vector<float>> fs_;
			std::size_t h_;
			std::size_t w_;
			Eigen::Matrix3f k_;
			
			
			void check (const std::vector<float> &) const;
			
			
		public:
		
		
			/**
			 *	Creates a new mock_depth_device with no frames
			 *	and a certain width and height.
			 *
			 *	\param [in] k
			 *		The \f$K\f$ matrix to be returned by the mock_depth_device.
			 *	\param [in] width
			 *		The width of frames returned by the mock_depth_device.
			 *		Defaults to zero.
			 *	\param [in] height
			 *		The height of frames returned by the mock_depth_device.
			 *		Defaults to zero.
			 */
			explicit mock_depth_device (Eigen::Matrix3f k=Eigen::Matrix3f::Zero(), std::size_t width=0, std::size_t height=0);
			
			
			/**
			 *	Adds a frame to the queue of frames to return.
			 *
			 *	\param [in] f
			 *		The frame to return.  Defaults to an empty
			 *		vector.
			 */
			void add (std::vector<float> f=std::vector<float>{});
			
			
			virtual value_type operator () (value_type v=value_type{}) override;
			virtual std::size_t width () const noexcept override;
			virtual std::size_t height () const noexcept override;
			virtual Eigen::Matrix3f k () const noexcept override;
		
		
	};
	
	
}

