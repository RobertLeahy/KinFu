/**
 *	\file
 */


#pragma once


#include <string>


namespace seng499 {
	
	
	/**
	 *	Retrieves the full path to the current executable.
	 *
	 *	\return
	 *		A string containing the full path to the current
	 *		executable.
	 */
	std::string current_executable_path ();
	/**
	 *	Retrieves the full path to the directory in which the
	 *	current executable resides.
	 *
	 *	\return
	 *		A string containing the full path to the directory
	 *		in which the current executable resides.
	 */
	std::string current_executable_parent_path ();
	
	
}
