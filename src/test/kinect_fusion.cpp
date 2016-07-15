#include <dynfu/kinect_fusion.hpp>


#include <stdexcept>
#include <catch.hpp>


SCENARIO("dynfu::kinect_fusion objects reject attempts to invoke them when the pipeline they manage is incomplete","[dynfu][kinect_fusion]") {

	GIVEN("A dynfu::kinect_fusion with an incomplete pipeline") {

		dynfu::kinect_fusion kf;

		THEN("Attempting to invoke it throws an exception") {

			CHECK_THROWS_AS(kf(),std::logic_error);

		}

	}

}
