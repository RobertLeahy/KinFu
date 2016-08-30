#include <kinfu/kinect_fusion.hpp>


#include <stdexcept>
#include <catch.hpp>


SCENARIO("kinfu::kinect_fusion objects reject attempts to invoke them when the pipeline they manage is incomplete","[kinfu][kinect_fusion]") {

	GIVEN("A kinfu::kinect_fusion with an incomplete pipeline") {

		kinfu::kinect_fusion kf;

		THEN("Attempting to invoke it throws an exception") {

			CHECK_THROWS_AS(kf(),std::logic_error);

		}

	}

}
