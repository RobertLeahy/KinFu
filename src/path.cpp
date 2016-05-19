#include <dynfu/path.hpp>
#include <whereami.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>


namespace dynfu {
	
	
	static std::string wai_wrapper (bool exec=true) {
		
		auto result=wai_getExecutablePath(nullptr,0,nullptr);
		if (result==-1) throw std::runtime_error("Error invoking wai_getExecutablePath to get length");
		
		auto ptr=reinterpret_cast<char *>(std::malloc(result+1));
		if (ptr==nullptr) throw std::bad_alloc{};
		std::unique_ptr<char,void (*) (void *)> p(ptr,&std::free);
		
		int dir;
		auto result_2=wai_getExecutablePath(ptr,result,exec ? nullptr : &dir);
		if (result_2==-1) throw std::runtime_error("Error invoking wai_getExecutablePath to get path");
		ptr[result]='\0';
		
		return std::string(ptr,ptr+(exec ? std::strlen(ptr) : dir));
		
	}
	
	
	std::string current_executable_path () {
		
		return wai_wrapper();
		
	}
	
	
	std::string current_executable_parent_path () {
		
		return wai_wrapper(false);
		
	}
	
	
}
