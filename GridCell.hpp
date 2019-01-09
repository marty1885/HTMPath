#pragma once

#include "HTMHelper.hpp"
#include <glm/mat2x2.hpp>
#include <glm/vec2.hpp>

using SDR = xt::xarray<bool>;

float density(const xt::xarray<bool>& a)
{
	return (float)xt::sum(a)[0]/a.size();
}

float random(float min=0, float max=1)
{
	std::random_device rd;
	static std::mt19937 eng(rd());
	std::uniform_real_distribution<float> dist;
	float diff = max-min;
	return diff*dist(eng) + min;
}

int roundCoord(float x)
{
	int v = (int)x + ((x-(int)x) > 0.5 ? 1 : -1);
	if(v < 0)
		v += 4;
	return v;
}

class GridCellUnit2D
{
public:
	GridCellUnit2D()
	{
		border_len = glm::vec2(4, 4);
		float scale_min = 6;
		float scale_max = 25;
		float theta = random(0,6.28);
		scale = random(scale_min, scale_max);
		bias = glm::vec2(random(0, 4), random(0, 4));
		transform_matrix = glm::mat2x2(cos(theta), -sin(theta), sin(theta), cos(theta));
	}

	SDR encode(glm::vec2 pos) const
	{
		SDR res = xt::zeros<bool>({border_len[0], border_len[1]});
		
		//Wrap the position
		glm::vec2 grid_cord = glm::mod(transform_matrix*pos/scale+bias, border_len);

		//Set the nearest cell to active
		xt::view(res, (int)grid_cord[1], (int)grid_cord[0]) = 1;

		//Set the 2nd nearest cell to active
		int cx = roundCoord(grid_cord[1])%4;
		int cy = roundCoord(grid_cord[0])%4;
		xt::view(res, cx, cy) = 1;

		res.reshape({res.size()});

		return res;
	}

	size_t encodeSize() const
	{
		return border_len[0] * border_len[1];
	}

	glm::mat2x2 transform_matrix;
	glm::vec2 border_len;
	glm::vec2 bias;
	float scale;
};

class GridCellEncoder2D
{
public:
	GridCellEncoder2D(int num_modules_ = 32)
	{
		for(int i=0;i<num_modules_;i++)
			units.push_back(GridCellUnit2D());
	}

	SDR encode(glm::vec2 pos) const
	{
		size_t num_cells = 0;
		for(const auto& u : units)
			num_cells += u.encodeSize();
		SDR res = xt::zeros<bool>({num_cells});
		size_t start = 0;
		for(const auto& u : units) {
			size_t l = u.encodeSize();
			xt::view(res, xt::range((int)start, l)) = u.encode(pos);
			start += l;
		}
		return res;
	}
	
	std::vector<GridCellUnit2D> units;
};

class LocEncoder2D
{
public:
	SDR encode(glm::vec2 pos) const
	{
		SDR res = xt::concatenate(xt::xtuple(HTM::encodeScalar(pos.x, 0, 800, 26, 16*16), HTM::encodeScalar(pos.y, 0, 600, 26, 16*16)));
		res.reshape({res.size()});

		return res;
	}
};