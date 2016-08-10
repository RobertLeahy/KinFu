/**
 *	\file
 */


#pragma once

#include <half.hpp>

namespace dynfu {

	using half=half_float::half;

	static_assert(sizeof(half)==2U,"dynfu::half is not 2 bytes");

}
