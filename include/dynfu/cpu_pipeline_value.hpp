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
			 *		forwarded through to a constructor of \em T.
			 *
			 *	\return
			 *		A reference to the newly created object.
			 */
			template <typename... Args>
			T & emplace (Args &&... args) noexcept(std::is_nothrow_constructible<T,Args...>::value) {
				
				obj_.emplace(std::forward<Args>(args)...);
				return *obj_;
				
			}
			
			
			/**
			 *	If this object contains a value of type \em T
			 *	retrieves that value.  Otherwise emplaces an object
			 *	of type \em T using arguments of type \em Args and
			 *	returns a reference thereto.
			 *
			 *	\tparam Args
			 *		The types of arguments to forward through to a
			 *		constructor of \em T if this object does not contain
			 *		such an object.
			 *
			 *	\param [in] args
			 *		Arguments to types \em Args which shall be forwarded
			 *		through to a constructor of \em T should it be
			 *		necessary to construct an object of type \em T.
			 *
			 *	\return
			 *		A reference to the object either created or already
			 *		contained by this object.
			 */
			template <typename... Args>
			T & get_or_emplace (Args &&... args) noexcept(std::is_nothrow_constructible<T,Args...>::value) {
				
				if (!obj_) return emplace(std::forward<Args>(args)...);
				
				return *obj_;
				
			}
		
		
	};
	
	
}
