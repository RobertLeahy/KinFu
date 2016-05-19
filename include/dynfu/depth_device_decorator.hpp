/**
 *	\file
 */


#include <dynfu/depth_device.hpp>


namespace dynfu {
	
	
	/**
	 *	A convenience base class which may be used to
	 *	implement classes which decorate \ref depth_device.
	 *
	 *	The provided implementations of all methods of
	 *	\ref depth_device simply call the corresponding
	 *	method of the decorated object.
	 */
	class depth_device_decorator : public depth_device {
		
		
		protected:
		
		
			depth_device & dev_;
					
		
		public:
		
		
			depth_device_decorator () = delete;
			
			
			/**
			 *	Creates a new \ref depth_device_decorator.
			 *
			 *	\param [in] dev
			 *		The \ref depth_device to decorate.
			 */
			depth_device_decorator (depth_device & dev) noexcept;
			
			
			virtual value_type operator () (value_type v=value_type{}) override;
			virtual std::size_t width () const noexcept override;
			virtual std::size_t height () const noexcept override;
			virtual Eigen::Matrix3f k () const noexcept override;
		
		
	};
	
	
}
