/**
 *	\file
 */


#pragma once


#include <dynfu/optional.hpp>
#include <dynfu/pipeline_value.hpp>
#include <type_traits>
#include <utility>


namespace dynfu {
	
	
	/**
	 *	Represents a pipeline value which is stored
	 *	on the CPU (i.e. in main memory).
	 *
	 *	\tparam T
	 *		The type of value.
	 */
	template <typename T>
	class cpu_pipeline_value : public pipeline_value<T> {
		
		
		private:
		
		
			optional<T> obj_;
			
			
		public:
		
		
			virtual const T & get () override {
				
				return *obj_;
				
			}
			
			
			/**
			 *	Supplies a value which this object will then
			 *	represent until this method is invoked once
			 *	again.
			 *
			 *	\tparam Args
			 *		The types of arguments to forward through
			 *		to a constructor of \em T.
			 *
			 *	\param [in] args
			 *		Arguments of types \em Args which shall be
			 *		forwarded through to a construct of \em T.
			 *
			 *	\return
			 *		A reference to the newly created object.
			 */
			template <typename... Args>
			T & emplace (Args &&... args) noexcept(std::is_nothrow_constructible<T,Args...>::value) {
				
				obj_.emplace(std::forward<Args>(args)...);
				return *obj_;
				
			}
		
		
	};
	
	
}
