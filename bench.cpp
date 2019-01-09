#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <functional>

#include <xtensor/xio.hpp>
#include "HTMHelper.hpp"
#include "GridCell.hpp"

template <typename T>
float testAnomaly(HTM::TemporalMemory& tm, const T& encoder, std::function<glm::vec2(glm::vec2, float)> f)
{
        float t = 0;
        SDR sample_sdr = encoder.encode(glm::vec2(30,-1));
        SDR last_pred = xt::zeros<bool>(sample_sdr.shape());
        std::vector<float> vec(2000);
        for(int i=0;i<2000;i++) {
                t += 0.016;

                float x = cos(t*3.14)*100.f + 100;
		float y = sin(t*3.14)*100.f + 100;

                glm::vec2 c = f({x, y}, t);
                SDR input = encoder.encode(c);
                SDR pred = tm.predict(input);

                float score = HTM::anomaly(input, last_pred);
                last_pred = pred;
                vec[i] = score;
        }

        return std::accumulate(vec.begin(), vec.end(), 0.f)/vec.size();
}

int main()
{
        GridCellEncoder2D encoder;
	SDR sample_sdr = encoder.encode(glm::vec2(30,-1));
	HTM::TemporalMemory tm({sample_sdr.size()} , 32);
	SDR last_pred = xt::zeros<bool>(sample_sdr.shape());

        tm->setPermanenceIncrement(0.04);
	tm->setPermanenceDecrement(0.045);
	tm->setPredictedSegmentDecrement(density(sample_sdr)*1.3f*tm->getPermanenceIncrement());
	tm->setCheckInputs(false);
	tm->setMaxNewSynapseCount(24);

        float t = 0;

        //Pre train the TM
        for(int i=0;i<16000;i++) {
                t += 0.016;
                float x = cos(t*3.14)*100.f + 100;
		float y = sin(t*3.14)*100.f + 100;

                SDR input = encoder.encode({x, y});
		tm.train(input);
        }

        float score = 0;
        tm.reset();
        score = testAnomaly(tm, encoder, [](glm::vec2 c, float t)->glm::vec2{return c;});
        std::cout << "Normal: " << score << std::endl;

        tm.reset();
        score = testAnomaly(tm, encoder, [](glm::vec2 c, float t)->glm::vec2{return c+glm::vec2(0, 5);});
        std::cout << "Shift 5 px: " << score << std::endl;

        tm.reset();
        score = testAnomaly(tm, encoder, [](glm::vec2 c, float t)->glm::vec2{return c+glm::vec2(0, 50);});
        std::cout << "Shift 50 px: " << score << std::endl;

        tm.reset();
        score = testAnomaly(tm, encoder, [](glm::vec2 c, float t)->glm::vec2{return c+glm::vec2(200, 200);});
        std::cout << "Revolve at (200, 200): " << score << std::endl;

        tm.reset();
        score = testAnomaly(tm, encoder, [](glm::vec2 c, float t)->glm::vec2{return glm::vec2(cos(t*3.14)*105.f + 100, sin(t*3.14)*105.f + 100);});
        std::cout << "Radius at 105: " << score << std::endl;

        tm.reset();
        score = testAnomaly(tm, encoder, [](glm::vec2 c, float t)->glm::vec2{return glm::vec2(cos(-t*3.14)*105.f + 100, sin(-t*3.14)*105.f + 100);});
        std::cout << "Reverse rotation direction: " << score << std::endl;

        tm.reset();
        score = testAnomaly(tm, encoder, [](glm::vec2 c, float t)->glm::vec2{return glm::vec2(50, 50);});
        std::cout << "Fixed at (50, 50): " << score << std::endl;
}