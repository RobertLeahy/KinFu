/**
 *	\file
 */


#pragma once


#include <boost/compute/type_traits/type_name.hpp>
#include <Eigen/Dense>


//	This is evil black magic
namespace boost {
	
	
	namespace compute {
		
		
		namespace detail {
			
			
			template <>
			struct type_name_trait<Eigen::Vector3f> {
				
				
				static const char * value () noexcept {
					
					return "Eigen::Vector3f";
					
				}
				
				
			};
			
			
		}
		
		
	}
	
	
}
