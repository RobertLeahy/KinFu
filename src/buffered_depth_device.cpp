#include <kinfu/buffered_depth_device.hpp>
#include <exception>
#include <stdexcept>
#include <thread>


namespace kinfu {


	std::unique_lock<std::mutex> buffered_depth_device::lock () const noexcept {

		return std::unique_lock<std::mutex>(m_);

	}


	void buffered_depth_device::worker () noexcept {

		try {

			worker_impl();

		} catch (...) {

			auto l=lock();
			ex_=std::current_exception();
			wait_.notify_all();

		}

	}


	void buffered_depth_device::worker_impl () {

		value_type v;
		for (;;) {

			//	Actually retrieve a depth frame
			v=depth_device_decorator::operator () (std::move(v));

			//	Enter critical section and enqueue depth frame
			auto l=lock();
			q_.push_back(std::move(v));
			wait_.notify_all();

			//	Wait if we have to
			wait_.wait(l,[&] () noexcept {

				if (stop_) return true;
				if (q_.size()<limit_) return true;
				return false;

			});

			//	Stop if this object is being destroyed
			if (stop_) break;

			//	Get a pre-allocated buffer if we have one
			if (pool_.empty()) {

				v=value_type{};

			} else {

				v=std::move(pool_.back());
				pool_.pop_back();

			}

		}

	}


	buffered_depth_device::buffered_depth_device (depth_device & dev, std::size_t limit) : depth_device_decorator(dev), stop_(false), limit_(limit) {

		if (limit_==0) throw std::logic_error("Must be able to buffer at least one frame");

		t_=std::thread([&] () noexcept {	worker();	});

	}


	buffered_depth_device::~buffered_depth_device () noexcept {

		auto l=lock();
		stop_=true;
		wait_.notify_all();
		l.unlock();

		t_.join();

	}


	buffered_depth_device::value_type buffered_depth_device::operator () (value_type v) {

		auto l=lock();
		if (v) pool_.push_back(std::move(v));
		wait_.wait(l,[&] () noexcept {

			if (!q_.empty()) return true;
			if (ex_) return true;
			return false;
		
		});

		//	We continue dequeuing until we run out of
		//	frames then we throw the stored exception
		if (!q_.empty()) {

			auto retr=std::move(q_.front());
			q_.pop_front();
			wait_.notify_all();
			return retr;

		}

		//	If we got here there's an exception and the
		//	queue is empty so throw the exception
		std::rethrow_exception(ex_);

	}


}
