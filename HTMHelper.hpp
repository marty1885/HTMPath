#pragma once

#include <nupic/algorithms/Cells4.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/algorithms/SpatialPooler.hpp>

#include <vector>

#include <cstdint>
#include <exception>
#include <algorithm>
#include <memory>

// #define HTM_USE_SYS_XTENSOR will allow users to use a custom vertsion of xtensor instead of the system provided version
#ifndef HTM_USE_SYS_XTENSOR
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#endif

namespace HTM
{

namespace NuPIC
{
using nupic::algorithms::spatial_pooler::SpatialPooler;
using nupic::algorithms::temporal_memory::TemporalMemory;
using nupic::algorithms::Cells4::Cells4;
}

//Convert a dens array to the index of on bits (spare array)
inline std::vector<UInt> sparsify(const xt::xarray<bool>& t)
{
	std::vector<UInt> v;
	v.reserve(xt::sum(t)[0]);
	for(size_t i=0;i<t.size();i++) {
		if(t[i] == true)
			v.push_back(i);
	}
	return v;
}

inline xt::xarray<float> softmax(const xt::xarray<float>& x)
{
	auto z = x - xt::amax(x)[0];
	auto e = xt::eval(xt::exp(z));
	return e/xt::sum(e);
}

//Calcluate the ratio of 1 per category
inline xt::xarray<float> categroize(int num_category, int len_per_category,const xt::xarray<bool>& in, bool normalize = true)
{
	xt::xarray<float> res = xt::zeros<float>({num_category});
	assert(res.size()*len_per_category == in.size());
	for(size_t i=0;i<in.size();i++)
		res[i/len_per_category] += (float)in[i];
	if(normalize == true)
		res /= len_per_category;
	return res;
}

//Calculate anomaly score given the SDR
//Implementation in NuPIC deals with sparse array. This deals with dense ones
inline float anomaly(xt::xarray<bool> real_value, xt::xarray<bool> prediction)
{
	auto not_pred_bits = (!prediction)&real_value;
	return (float)xt::sum(not_pred_bits)[0]/xt::sum(real_value)[0];
}

//Converts container from one to another
template<typename ResType, typename InType>
inline ResType as(const InType& shape)
{
	return ResType(shape.begin(), shape.end());
}

template <typename V>
std::string vectorToString(const V& v)
{
	std::string str = "{";
	for(const auto& i : v)
		str += std::to_string(i) + ", ";
	str += "}";
	return str;
}

struct HTMLayerBase
{
	HTMLayerBase() = default;
	virtual ~HTMLayerBase() = default;
	HTMLayerBase(std::vector<size_t> inDim, std::vector<size_t> outDim)
		: input_shape(inDim), output_shape(outDim){}
	
	std::vector<size_t> input_shape;
	std::vector<size_t> output_shape;
	
	virtual xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn) = 0;
	xt::xarray<bool> operator() (const xt::xarray<bool>& t, bool learn=true) {return compute(t, learn);}

	void train(const xt::xarray<bool>& t)
	{
		compute(t, true);
	}

	//Unfortunatelly due to how HTM works, the ptedict function cannot be const.
	xt::xarray<bool> predict(const xt::xarray<bool>& t)
	{
		return compute(t, false);
	}
	
	size_t inputSize()
	{
		size_t s = 1;
		for(auto v : input_shape)
			s *= v;
		return s;
	}
	
	size_t outputSize()
	{
		size_t s = 1;
		for(auto v : output_shape)
			s *= v;
		return s;
	}

	virtual void reset() {}
};

struct SpatialPooler : public HTMLayerBase
{
	SpatialPooler() = default;
	SpatialPooler(std::vector<size_t> inDim, std::vector<size_t> outDim)
		: HTMLayerBase(inDim, outDim), sp(as<std::vector<UInt>>(input_shape), as<std::vector<UInt>>(output_shape))
	{
		std::vector<UInt> inSize = as<std::vector<UInt>>(input_shape);
		std::vector<UInt> outSize = as<std::vector<UInt>>(output_shape);
	}
	
	virtual xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn) override
	{
		auto in_shape = t.shape();
		if(std::equal(input_shape.begin(), input_shape.end(), in_shape.begin(), in_shape.end()) == false) {
			throw std::runtime_error("SpatialPooler: expecting input shape " + vectorToString(input_shape)
				+ ", but get " + vectorToString(in_shape));
		}
		std::vector<UInt> in(inputSize());
		std::vector<UInt> out(outputSize());
		for(size_t i=0;i<t.size();i++)
			in[i] = t[i];
		
		sp.compute(in.data(), learn, out.data());
		
		xt::xarray<bool> res = xt::zeros<bool>(output_shape);
		for(size_t i=0;i<out.size();i++)
			res[i] = out[i];
		return res;
	}

	NuPIC::SpatialPooler* operator-> ()
	{
		return &sp;
	}

	const NuPIC::SpatialPooler* operator-> () const
	{
		return &sp;
	}
	
	NuPIC::SpatialPooler sp;
};

struct TemporalPooler : public HTMLayerBase
{
	TemporalPooler() = default;
	TemporalPooler(std::vector<size_t> inDim, size_t numCol)
		: HTMLayerBase(inDim, inDim), colInTP(numCol)
		, tp(inputSize(), colInTP, 6, 6, 15, .1, .21, 0.23, 1.0, .1, .1, 0.002,
			false, 42, true, false)
	{
	}
	
	virtual xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn) override
	{
		auto in_shape = t.shape();
		if(std::equal(input_shape.begin(), input_shape.end(), in_shape.begin(), in_shape.end()) == false) {
			throw std::runtime_error("TemporalPooler: expecting input shape " + vectorToString(input_shape)
				+ ", but get " + vectorToString(in_shape));
		}
		std::vector<Real> in(t.begin(), t.end());
		std::vector<Real> out(t.size()*colInTP);
		xt::xarray<bool> res = xt::zeros<bool>(output_shape);
		
		tp.compute(in.data(), out.data(), true, learn);
		
		//Convert output into SDR
		for(size_t i=0;i<out.size()/colInTP;i++)
			res[i] = out[i*colInTP];
		return res;
	}
	NuPIC::Cells4* operator-> ()
	{
		return &tp;
	}

	const NuPIC::Cells4* operator-> () const
	{
		return &tp;
	}
	
	void reset()
	{
		tp.reset();
	}
	
	size_t colInTP;
	NuPIC::Cells4 tp;
};

struct TemporalMemory : public HTMLayerBase
{
	TemporalMemory() = default;
	TemporalMemory(std::vector<size_t> in_dim, size_t num_col, size_t max_segments_per_cell=255, size_t max_synapses_per_segment=255)
		: HTMLayerBase(in_dim, in_dim), col_in_tp(num_col)
	{
		std::vector<UInt> in_size = as<std::vector<UInt>>(in_dim);
		tm = NuPIC::TemporalMemory(in_size, num_col, 13, 0.21, 0.5, 10, 20, 0.1, 0.1, 0, 42, max_segments_per_cell, max_synapses_per_segment, true);
	}
	
	virtual xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn) override
	{
		auto in_shape = t.shape();
		if(std::equal(input_shape.begin(), input_shape.end(), in_shape.begin(), in_shape.end()) == false) {
			throw std::runtime_error("Temporalmemory: expecting input shape " + vectorToString(input_shape)
				+ ", but get " + vectorToString(in_shape));
		}
		std::vector<UInt> cols = sparsify(t);
		xt::xarray<bool> tp_output = xt::zeros<bool>(t.shape());
		tm.compute(cols.size(), cols.data(), learn);
		auto next = tm.getPredictiveCells();
		for(auto idx : next)
			tp_output[idx/col_in_tp] = true;
		return tp_output;
	}

	NuPIC::TemporalMemory* operator-> ()
	{
		return &tm;
	}

	const NuPIC::TemporalMemory* operator-> () const
	{
		return &tm;
	}
	
	void reset()
	{
		tm.reset();
	}
	
	size_t col_in_tp;
	NuPIC::TemporalMemory tm;
};
//Encoders

//Your standard ScalarEncoder.
struct ScalarEncoder
{
	ScalarEncoder() = default;
	ScalarEncoder(float minval, float maxval, size_t encode_len, size_t width)
		: min_val(minval), max_val(maxval), encode_length(encode_len), sdr_length(width)
	{
		if(min_val > max_val)
			throw std::runtime_error("ScalarEncoder error: min_val > max_val");
	}

	xt::xarray<bool> operator() (float value) const
	{
		return encode(value);
	}

	xt::xarray<bool> encode(float value) const
	{
		float encode_space = sdr_length - encode_length;
		float v = value - min_val;
		v /= max_val-min_val;
		v = std::max(std::min(v, 1.f), 0.f);
		int start = encode_space*v;
		int end = start + encode_length;
		xt::xarray<bool> res = xt::zeros<bool>({sdr_length});
		xt::view(res, xt::range(start, end))  = true;
		return res;
	}

	void setMiniumValue(float val) {min_val = val;}
	void setMaximumValue(float val) {max_val = val;}
	void setEncodeLengt(size_t val) {encode_length = val;}
	void setSDRLength(size_t val) {sdr_length = val;}

	float miniumValue() const {return min_val;}
	float maximumValue() const {return max_val;}
	size_t encodeLength() const {return encode_length;}
	size_t sdrLength() const {return sdr_length;}


protected:
	float min_val = 0;
	float max_val = 1;
	size_t encode_length = 8;
	size_t sdr_length = 32;
};

//Unlike in NuPIC. The CategoryEncoder in HTMHelper does NOT include space for
//an Unknown category. And the encoding is done by passing a size_t representing
//the category instread of a string.
struct CategoryEncoder
{
	CategoryEncoder(size_t num_cat, size_t encode_len)
		: num_category(num_cat), encode_length(encode_len)
	{}

	xt::xarray<bool> operator() (size_t category) const
	{
		return encode(category);
	}

	xt::xarray<bool> encode(size_t category) const
	{
		if(category > num_category)
			throw std::runtime_error("CategoryEncoder: category > num_category");
		xt::xarray<bool> res = xt::zeros<bool>({num_category, encode_length});
		xt::view(res, category) = true;
		return xt::flatten(res);
	}

	std::vector<size_t> decode(const xt::xarray<bool>& t)
	{
		std::vector<size_t> possible_category;
		for(size_t i=0;i<num_category;i++) {
			if(xt::sum(xt::view(t, xt::range(i*encode_length, (i+1)*encode_length)))[0] > 0)
				possible_category.push_back(i);
		}
		return possible_category;
	}

	void setNumCategorise(size_t num_cat) {num_category = num_cat;}
	void setEncodeLengt(size_t val) {encode_length = val;}

	size_t numCategories() const {return num_category;}
	size_t encodeLength() const {return encode_length;} 
	size_t sdrLength() const {return num_category*encode_length;}
protected:
	size_t num_category;
	size_t encode_length;
};


//Handy encode functions
inline xt::xarray<bool> encodeScalar(float value, float minval, float maxval, size_t encode_len, size_t width)
{
	ScalarEncoder e(minval, maxval, encode_len, width);
	return e.encode(value);
}

inline xt::xarray<bool> encodeCategory(size_t category, size_t num_cat, size_t encode_len)
{
	CategoryEncoder e(num_cat, encode_len);
	return e.encode(category);
}

// Neural Netowrk like model class
// Not used for now.
class SequentalNetwork : public HTMLayerBase
{
public:

	//Pushes a layer into the back of the netoek
	template <typename LayerType, typename ... Args>
	void add(Args ... args)
	{
		layers.push_back(std::move(std::make_unique<LayerType>(args ...)));

		//update info
		if(layers.size() == 1)
			input_shape = layers[0]->input_shape;
		output_shape = layers.back()->output_shape;
	}

	//Returns the address of the layer
	template <typename T = HTMLayerBase>
	T* at(size_t index)
	{
		if(index > layers.size())
			throw std::runtime_error("SequentalNetwork: Only has " + std::to_string(layers.size())
				+ " layers, but layer " + std::to_string(index) + " requested.");
		HTMLayerBase* layer_ptr = layers[index].get();
		T* ptr = dynamic_cast<T*>(layer_ptr);
		if(ptr == nullptr)
			throw std::runtime_error("SequentalNetwork: Layer " + std::to_string(index) + ", request type mismatch");
		return ptr;
	}

	//Reset layer state
	void reset()
	{
		for(auto& layer : layers)
			layer->reset();
	}

	virtual xt::xarray<bool> compute(const xt::xarray<bool>& in, bool learn) override
	{
		xt::xarray<bool> buffer = in;

		for(auto& layer : layers)
			buffer = layer->compute(buffer, learn);
		return buffer;
	}

protected:
	std::vector<std::unique_ptr<HTMLayerBase>> layers;
};

//Classifers
struct SDRClassifer
{
	SDRClassifer(size_t num_classes, std::vector<size_t> shape)
		: stored_patterns(num_classes, xt::zeros<int>(as<xt::xarray<int>::shape_type>(shape)))
		, pattern_sotre_num(num_classes)
	{}

	void add(size_t category, const xt::xarray<bool>& t)
	{
		stored_patterns[category] += t;
		pattern_sotre_num[category] += 1;
	}

	size_t compute(const xt::xarray<bool>& t, float bit_common_threhold = 0.5) const
	{
		assert(bit_common_threhold >= 0.f && bit_common_threhold <= 1.f);
		size_t best_pattern = 0;
		size_t best_score = 0;
		for(size_t i=0;i<numPatterns();i++) {
			auto overlap = t & xt::cast<bool>(stored_patterns[i] >= (int)(pattern_sotre_num[i]*bit_common_threhold));
			size_t overlap_score = xt::sum(overlap)[0];
			if(overlap_score > best_score) {
				best_score = overlap_score;
				best_pattern = i;
			}
		}

		return best_pattern;
	}

	size_t numPatterns() const
	{
		return stored_patterns.size();
	}

	void reset()
	{
		for(size_t i=0;i<numPatterns();i++) {
			stored_patterns[i] = 0;
			pattern_sotre_num[i] = 0;
		}
	}

protected:
	std::vector<xt::xarray<int>> stored_patterns;
	std::vector<size_t> pattern_sotre_num;
};

} //End of namespace HTM

