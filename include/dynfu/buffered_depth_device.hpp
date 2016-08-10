/**
 *	\file
 */


#pragma once


#include <dynfu/depth_device.hpp>
#include <dynfu/depth_device_decorator.hpp>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>


namespace dynfu {


	/**
	 *	A \ref depth_device which buffers images from an underlying
	 *	\ref depth_device by loading them using a background thread.
	 */
	class buffered_depth_device final : public depth_device_decorator {


		private:


			std::deque<value_type> q_;
			std::vector<value_type> pool_;
			mutable std::mutex m_;
			mutable std::condition_variable wait_;
			std::thread t_;
			std::exception_ptr ex_;
			bool stop_;
			std::size_t limit_;


			std::unique_lock<std::mutex> lock () const noexcept;
			void worker () noexcept;
			void worker_impl ();


		public:


			/**
			 *	Creates a new buffered_depth_device which buffers up to
			 *	\em limit frames from the underlying \ref depth_device.
			 *
			 *	\param [in] dev
			 *		The \ref depth_device from which frames shall be
			 *		buffered.  This reference must remain valid for the
			 *		lifetime of this object or the behaviour is undefined.
			 *	\param [in] limit
			 *		The maximum number of threads which may be buffered.
			 *		If this many threads are buffered the background thread
			 *		will stop loading threads until some are dequeued.
			 */
			buffered_depth_device (depth_device & dev, std::size_t limit);


			/**
			 *	Shuts down the background thread and releases all resources.
			 */
			~buffered_depth_device () noexcept;


			virtual value_type operator () (value_type v=value_type{}) override;


	};


}
