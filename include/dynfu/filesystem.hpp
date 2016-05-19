/**
 *	\file
 */


#if __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#else
#include <boost/filesystem.hpp>
#endif


namespace dynfu {
	
	
	/**
	 *	Contains the contents of boost::filesystem,
	 *	std::experimental::filesystem, or std::filesystem,
	 *	depending on what is available in the current compiler
	 *	and standard library version.
	 */
	namespace filesystem {
		
		
		using namespace
		#if __has_include(<experimental/filesystem>)
		std::experimental::filesystem
		#else
		boost::filesystem
		#endif
		;
		
		
	}
	
	
}
