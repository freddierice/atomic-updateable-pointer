#pragma once

#include <array>
#include <atomic>
#include <cinttypes>
#include <functional>
#include <memory>
#include <vector>

namespace atomicops {
#ifdef __x86_64__
	constexpr uintptr_t PTR_MASK = ~(1ULL << 48);
	template <typename T>
	class AtomicUpdateablePointer {
	public:
		AtomicUpdateablePointer(T *x) : m_packed(pack_ptr_data(x, 0)) {}
		~AtomicUpdateablePointer() {
			// spin on any leftover pointers
			while (unpack_data(m_packed.load())) {}
			
			// if a pointer is still there, delete it
			T *ptr = unpack_ptr(m_packed.load());
			if (ptr) delete ptr;
		}
		
		uint16_t get_data() {
			return unpack_data(m_packed.load());
		}
		
		template <typename F=std::function<void(T*)>>
		std::unique_ptr<T, F> get_ptr() {
			T* ptr;
			uint16_t data;
			uint64_t packed;

			packed = m_packed.load();
			ptr = unpack_ptr(packed);
			if (!ptr) 
				return std::unique_ptr<T,F>(nullptr, [](T*){});
			data = unpack_data(packed);

			data += 1;
			if (!m_packed.compare_exchange_strong(packed,
					pack_ptr_data(ptr, data)))
				return std::unique_ptr<T, F>(nullptr, [](T*){});
			return std::unique_ptr<T, F>(ptr, [&](T *t){
					for (;;) {
						uint64_t packed = m_packed.load();
						uint64_t new_packed;
						ptr = unpack_ptr(packed);
						data = unpack_data(packed);
						data -= 1;
						new_packed = pack_ptr_data(ptr, data);
						if (m_packed.compare_exchange_strong(packed, new_packed)) break;
					}
				});
		}

		std::unique_ptr<T> take_ptr() {
			T* old_ptr, *ptr;
			uint16_t data;
			uint64_t packed;

			packed = m_packed.load();
			old_ptr = unpack_ptr(packed);
			if (!old_ptr) 
				return std::unique_ptr<T>(nullptr);
			data = unpack_data(packed);

			data += 1;
			ptr = static_cast<T*>(nullptr);
			if (!m_packed.compare_exchange_strong(packed,
					pack_ptr_data(ptr, data)))
				return std::unique_ptr<T>(nullptr);
			
			// delete when done
			return std::unique_ptr<T>(old_ptr);
		}

		bool put_ptr(T *ptr) {
			uint16_t data;
			uint64_t packed;

			packed = m_packed.load();
			if (unpack_ptr(packed))
				return false;
			data = unpack_data(packed);
			if (data != 1)
				return false;

			data = 0;
			// nobody else could have changed the value, just do a store
			m_packed.store(pack_ptr_data(ptr, data));
			return true;
		}

		static T* unpack_ptr(uint64_t packed) {
			return reinterpret_cast<T*>((packed << 16) >> 16);
		}

		static uint16_t unpack_data(uint64_t packed) {
			return static_cast<uint16_t>(packed >> 48);
		}

		static uint64_t pack_ptr_data(T *ptr, uint16_t data) {
			return (reinterpret_cast<uint64_t>(ptr) & PTR_MASK) | 
				(static_cast<uint64_t>(data) << 48);
		}

	private:
		AtomicUpdateablePointer() = delete;
		AtomicUpdateablePointer(AtomicUpdateablePointer&) = delete;
		void get_ptr_data(T*& ptr, uint16_t& data) {
			uint64_t packed = m_packed.load();
			ptr = unpack_ptr(packed);
			data = unpack_data(packed);
		}

		// std::atomic<uint64_t> m_packed;
		std::atomic<uint64_t> m_packed;
	};

	class FlagArray {
	public:
		FlagArray() : FlagArray(0) {}
		FlagArray(uint64_t len) : m_flag_size(len),
			m_aup(new std::vector<std::atomic<uint8_t>>((m_flag_size + 7)/8)),
	   		m_count(0) {
			}
		~FlagArray() {
			// spin on m_count
			while (m_count.load()) {}
		}

		class Flag {
		public:
			Flag(FlagArray* arr, uint64_t idx) : m_parent(arr), m_idx(idx){
				arr->take_wait(m_idx);
			}
			~Flag() {
				m_parent->give_wait(m_idx);
			}
			Flag(Flag&& f) : m_parent(f.m_parent), m_idx(f.m_idx) {}
			
		private:
			Flag() = delete;
			Flag(Flag&) = delete;
			FlagArray *m_parent;
			uint64_t m_idx;
		};

		static void get_idx_mask(uint64_t &idx, uint8_t &mask) {
			mask = 1 << (idx % 8);
			idx /= 8;
		}

		Flag flag(uint64_t idx) {
			return Flag(this, idx);
		}

		bool give(uint64_t idx) {
			if (toggle(idx, false)) {
				m_count--;
				return true;
			}
			return false;
		}

		void give_wait(uint64_t idx) {
			while (!give(idx)) {}
		}

		bool resize(uint64_t size) {
			if (size < m_flag_size) throw std::runtime_error("shrinking not supported");
			if (size == m_flag_size) return true;
			
			auto ptr = m_aup.take_ptr();
			if (!ptr) return false;

			auto new_ptr = new std::vector<std::atomic<uint8_t>>((size + 7) / 8);
			
			// spin until count is 1
			while (m_aup.get_data() != 1) {}

			// we are free to copy
			for (uint64_t i = 0; i < (m_flag_size + 7)/8; i++)
				(*new_ptr)[i].store((*ptr)[i].load());

			// spin on putting the new ptr (should not actually spin)
			while (!m_aup.put_ptr(new_ptr)) {}
			
			// update size
			m_flag_size = size;

			return true;
		}
		
		void resize_wait(uint64_t idx) {
			while (!resize(idx)) {}
		}

		bool take(uint64_t idx) {
			m_count++;
			if (!toggle(idx, true)) {
				m_count--;
				return false;
			}
			return true;
		}

		void take_wait(uint64_t idx) {
			while (!take(idx)) {}
		}

	private:
		FlagArray(const FlagArray&) = delete;
		
		bool toggle(uint64_t idx, uint8_t on) {
			uint8_t mask, data;
			if (idx >= m_flag_size) 
				return false;
			auto ptr = m_aup.get_ptr();
			if (!ptr) return false;
			get_idx_mask(idx, mask);
			data = ptr->at(idx).load();
			if ((on && (data & mask)) || (!on && !(data & mask)))
				return false;
			return ptr->at(idx).compare_exchange_strong(data, data ^ mask);
		}
		
		uint64_t m_flag_size;
		AtomicUpdateablePointer<std::vector<std::atomic<uint8_t>>> m_aup;
		std::atomic<uint64_t> m_count;
	};
#else
	#error "ERROR: could not compile for non x86-64 bit computer"
#endif
}
