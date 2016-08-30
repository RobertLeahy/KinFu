/**
 *	\file
 */


#pragma once


#include <boost/compute.hpp>
#include <kinfu/half.hpp>
#include <kinfu/pixel.hpp>
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


			template <>
			struct type_name_trait<kinfu::pixel> {
				
				
				static const char * value () noexcept {
					
					return "kinfu::pixel";
					
				}
				
				
			};

			template <>
			struct type_name_trait<kinfu::half> {
				
				
				static const char * value () noexcept {
					
					return "kinfu::half";
					
				}
				
				
			};
			
			
		}
		
		
	}
	
	
}
