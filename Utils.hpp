#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

static xt::xarray<float> dct(const xt::xarray<float>& signal, const xt::xarray<float>& coeff)
{
	return xt::sum(coeff*signal, {1})/signal.shape()[0];
}

inline xt::xarray<float> ftView(const xt::xarray<float>& signal, size_t currentInd
	, size_t windowSize)
{
	if(currentInd < windowSize || currentInd > signal.shape()[1])
		currentInd = windowSize;
	auto v = xt::view(signal, 0, xt::range(currentInd-windowSize, currentInd));
	return v;
}


static xt::xarray<float> computeSpectrum(const xt::xarray<float>& signal, size_t currentInd
	, size_t windowSize, const xt::xarray<float>& curve, const xt::xarray<float>& coeff)
{
	if(curve.size() == coeff.size())
		throw std::runtime_error("curve.size() == coeff.size() failed.");
	auto v = ftView(signal, currentInd, windowSize);
	auto r = xt::abs(dct(v, coeff))*curve;
	xt::xarray<float> db = 20.f*xt::log10(r);
	if(currentInd == windowSize)
		db *= 0;
	return db;
}

xt::xarray<float> generateDCTCoeff (size_t length, size_t sampleRate, xt::xarray<float> freqs)
{
	xt::xarray<float> coeff = xt::zeros<float>({freqs.size(),length});
	for(size_t i=0;i<freqs.size();i++)
	{
		auto v = xt::view(coeff, i);
		auto phase = (xt::arange<float>(length))
			*(2.f*(float)M_PI*freqs[i]/(float)sampleRate);
		v = xt::cos(phase);
	}
	return coeff;
}

xt::xarray<float> weightCurve(const xt::xarray<float>& bins)
{
	//A-wighting
	xt::xarray<float> aWeight = (12194.f*12194.f*bins*bins*bins*bins)/
		((bins*bins+20.6f*20.6f)*xt::sqrt((bins*bins+107.7*107.7)*(bins*bins+737.9*737.9))*(12194.f*12194.f+bins*bins));
	//B-weightingsunrise
	xt::xarray<float> bWeight = (12194.f*12194.f*bins*bins*bins)/
		((bins*bins+20.6f*20.6f)*xt::sqrt((bins*bins+158.5*158.5))*(12194.f*12194.f+bins*bins));
	float mixVal = 0.5f;
	assert(mixVal >= 0.f && mixVal <= 1.f);
	return aWeight*mixVal + bWeight*(1.f-mixVal);
}

struct EarDFT
{
	EarDFT() = default;
	EarDFT(size_t numBins, size_t sampleRate, size_t windowSize)
                : ftSize(windowSize), rate(sampleRate)
        {
                bins = xt::linspace<float>(70, 1800, numBins); //2702
                bins = 700.f*(xt::pow(10.f, bins/2595.f)-1);//Mel to freq approx 75~8000Hz
                coeff = generateDCTCoeff(windowSize, sampleRate, bins);
                weights = weightCurve(bins);
        }
	
	xt::xarray<float> operator() (const xt::xarray<float>& signal, int index)
	{
		return computeSpectrum(signal, index, ftSize, weights, coeff);
	}
	
	xt::xarray<float> operator() (const xt::xarray<float>& signal, float t)
	{
		return computeSpectrum(signal, (double)t*rate, ftSize, weights, coeff);
	}
	
	xt::xarray<float> bins;
	xt::xarray<float> coeff;
	xt::xarray<float> weights;
	size_t ftSize;
	size_t rate;
};
