/**
 *	\file
 */


#pragma once

#include <half.hpp>

namespace kinfu {

	using half=half_float::half;

	static_assert(sizeof(half)==2U,"kinfu::half is not 2 bytes");

}
