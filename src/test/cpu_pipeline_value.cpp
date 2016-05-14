#include <seng499/cpu_pipeline_value.hpp>


#include <catch.hpp>


SCENARIO("Values may be provided and retrieved through a seng499::cpu_pipeline_value<T>","[seng499][pipeline_value][cpu_pipeline_value]") {
	
	GIVEN("A seng499::cpu_pipeline_value<T>") {
		
		seng499::cpu_pipeline_value<int> pv;
		
		WHEN("A value is emplaced therein") {
			
			auto && v=pv.emplace(5);
			
			THEN("seng499::cpu_pipeline_value::get returns a reference to that same object") {
				
				CHECK(&v==&pv.get());
				
			}
			
			THEN("The correct value is created") {
				
				CHECK(v==5);
				
			}
			
		}
		
	}
	
}
