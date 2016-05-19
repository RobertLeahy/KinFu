/**
 *	\file
 */


#if __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#else
#include <boost/filesystem.hpp>
#endif


namespace dynfu {
	
	
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
