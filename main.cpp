#include <vector>
#include <map>
#include <random>
#include <cmath>

#include <xtensor/xio.hpp>
#include "HTMHelper.hpp"
#include "GridCell.hpp"

#include <SFML/Graphics.hpp>
#include <SFML/Window/Keyboard.hpp>

#include "imgui.h"
#include "imgui-SFML.h"
#include "gui.hpp"

#include "CircularBuffer.h"

#include <chrono>
using namespace std::chrono;

std::vector<sf::Texture> guiTmpTexture;
std::vector<sf::Image> guiTmpImage;

void clearTmp()
{
	guiTmpTexture.clear();
	guiTmpImage.clear();
}

void ShowSDR(const xt::xarray<bool>& sdr, std::string title="")
{
	guiTmpImage.push_back(sf::Image());
	guiTmpTexture.push_back(sf::Texture());
	sf::Image& img = guiTmpImage.back();
	sf::Texture& texture = guiTmpTexture.back();
	sf::Uint8* pixels = new sf::Uint8[sdr.size()*4];
	for(size_t i=0;i<sdr.size();i++) {
		size_t j = sdr.size()-i-1;
		if(sdr[i]) {
			pixels[j*4+0] = 0.29*255;
			pixels[j*4+1] = 0.707*255;
			pixels[j*4+2] = 0.865*255;
			pixels[j*4+3] = 255;
		}
		else {
			pixels[j*4+0] = 255;
			pixels[j*4+1] = 255;
			pixels[j*4+2] = 255;
			pixels[j*4+3] = 255;
		}
	}
	img.create(sdr.shape()[1], sdr.shape()[0], pixels);
	img.flipVertically();
	if(texture.loadFromImage(img) == false)
		throw std::runtime_error("Create texture failed");
	unsigned int textureID = texture.getNativeHandle();
	ImGui::Image((void*)textureID, ImVec2(sdr.shape()[1]*4, sdr.shape()[0]*4));
	ImGui::SameLine(); ImGui::Text(title.c_str());
	delete [] pixels;
}

template<typename T>
void PlotCircularBuffer(CircularBuffer<T>& buffer, std::string title, std::string text, ImVec2 size, float minVal=0, float maxVal=1)
{
	ImGui::PlotLines(title.c_str(), [](void* ptr, int n)->float{return (*(CircularBuffer<float>*)ptr)[n];}, &buffer, buffer.size(), 0,text.c_str(), minVal, maxVal, size);
}

template<typename T>
void PlotCircularBufferAve(CircularBuffer<T>& buffer, std::string title, std::string text, ImVec2 size, float minVal=0, float maxVal=1)
{
	std::vector<float> res(buffer.size());
	for(size_t i=0;i<res.size();i++) {
		float s = 0;
		for(size_t j=0;j<10;j++)
			s += buffer[(i+j)%buffer.size()];
		s /= 10;
		res[i] = s;
	}
	ImGui::PlotLines(title.c_str(), res.data(), buffer.size(), 0,text.c_str(), minVal, maxVal, size);
}

int main()
{
	GridCellEncoder2D encoder;
	SDR sample_sdr = encoder.encode(glm::vec2(30,-1));
	HTM::TemporalMemory tm({sample_sdr.size()} , 32);
	SDR last_pred = xt::zeros<bool>(sample_sdr.shape());

	CircularBuffer<float> anomaly_history(256);
	for(size_t i=0;i<anomaly_history.capacity();i++)
		anomaly_history.add(0);

	tm->setPermanenceIncrement(0.04);
	tm->setPermanenceDecrement(0.045);
	tm->setPredictedSegmentDecrement(density(sample_sdr)*1.3f*tm->getPermanenceIncrement());
	tm->setCheckInputs(false);
	tm->setMaxNewSynapseCount(24);
	
	float t = 0;
	
	sf::RenderWindow window(sf::VideoMode(800, 600), "HTM Path");
	window.setFramerateLimit(60);
	sf::CircleShape circle;
	circle.setRadius(15);
	circle.setOutlineColor(sf::Color::Red);
	circle.setOutlineThickness(5);

	ImGui::SFML::Init(window);
	ImGui::setupImGuiStyle(false, 0.9);
	
	sf::Event event;
	auto t1 = high_resolution_clock::now();
	while (window.isOpen()) {
		static bool time_flow = true;
		if(time_flow)
			t += 0.016; //Assuming 60 FPS
		while (window.pollEvent(event)) {
			ImGui::SFML::ProcessEvent(event);
			if(event.type == sf::Event::Closed)
				window.close();
			if(event.type == sf::Event::KeyPressed) {
				if(event.key.code == sf::Keyboard::T) {
					time_flow = !time_flow;
				}
			}
		}

		auto t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		double delta = time_span.count();
		ImGui::SFML::Update(window, sf::seconds(delta));

		t1 = t2;

		
		float x = cos(t*3.14)*100.f + 100;
		float y = sin(t*3.14)*100.f + 100;
		
		/*
		float x = 0;
		float y = 0;
		float ts = t/4.f;
		int ti = ((int)ts)%4;
		float tf = ts - (int)ts;
		if(ti == 0)
			x += tf*100;
		else if(ti == 2)
			x = (1-tf)*100, y = 100;
		else if(ti == 1)
			y += tf*100, x = 100;
		else if(ti == 3)
			y = (1-tf)*100, x = 0;*/
		bool learn = time_flow;
		if(sf::Keyboard::isKeyPressed(sf::Keyboard::C))
			x = 50, y = 50, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
			y -= 50, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
			y += 50, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
			x -= 50, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
			x += 50, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::O))
			x += 200, y += 200, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::W))
			y -= 5, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			y += 5, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			x -= 5, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			x += 5, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::L))
			x = cos(t*3.14)*105.f + 100, y =  sin(t*3.14)*105.f + 100, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::R))
			x = cos(-t*3.14)*100.f + 100, y =  sin(-t*3.14)*100.f + 100, learn = false;
		else if(sf::Keyboard::isKeyPressed(sf::Keyboard::N))
			learn = false;
		
		if(sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) // force learning
			learn = true;

		static int max_framerate = 60; // default;
		bool p = sf::Keyboard::isKeyPressed(sf::Keyboard::P);
		bool f = sf::Keyboard::isKeyPressed(sf::Keyboard::F);
		if(p && max_framerate == 60) {
			window.setFramerateLimit(10);
			max_framerate = 10;
		}
		else if(p == false && max_framerate == 10) {
			window.setFramerateLimit(60);
			max_framerate = 60;
		}
		else if(f && max_framerate == 60) {
			window.setFramerateLimit(0);
			max_framerate = 0;
		}
		else if(!f && max_framerate == 0) {
			window.setFramerateLimit(60);
			max_framerate = 60;
		}

		circle.setPosition(x, y);

		SDR input = encoder.encode({x, y});
		SDR pred = tm.compute(input, learn);

		float score = HTM::anomaly(input, last_pred);
		last_pred = pred;
		static float anomaly_thr = 0.5;

		anomaly_history.add(score);

		//Status window
		ImGui::Begin("Status");
		ImGui::Text((std::string("Learning: ")+(learn?"Enable":"Disabled")).c_str());
		input.reshape({16, input.size()/16});
		pred.reshape({16, pred.size()/16});
		ShowSDR(input, "Grid Cells");
		ShowSDR(pred, "TemporalMemory Pediction");
		ShowSDR((!pred)&input, "Not predicted");
		PlotCircularBuffer(anomaly_history, "Anomaly", "", ImVec2(256, 64), 0, 1);
		ImGui::SetWindowSize(ImVec2(0,0));
		ImGui::Text("Anomaly Score: ");
		ImGui::ProgressBar(score, ImVec2(128+64, 16), std::to_string(score).c_str());
		ImGui::SliderFloat("Anomaly thr:", & anomaly_thr, 0, 1);
		ImGui::Text((std::string("Anomaly: ")+(score > anomaly_thr?"Yes":"No")).c_str());
		ImGui::End();

		window.clear(sf::Color(20, 20, 20));
		window.draw(circle);
		ImGui::Render();
		window.display();
		clearTmp();
	}

	ImGui::SFML::Shutdown();
}
