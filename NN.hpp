#pragma once
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <memory>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include "crc32.h"

/*
Класс двухслойной нейронной сети со смещениями (bias)
и функциями активации silu и softmax
фиксированные количества нейронов слоёв
параметры w и h отвечают за размер входного слоя,
s - количество нейронов скрытого слоя
k - количество нейронов выходного слоя
*/
template<size_t w, size_t h, size_t s, size_t k>
struct NN2 {
	typedef Eigen::Matrix<float, w*h, 1> Vec_in;
	typedef Eigen::Matrix<float, k, 1> Vec_out;
	typedef Eigen::Matrix<float, s, 1> Vec_h;
	typedef Eigen::Matrix<float, w*h, s> l1_t;
	typedef Eigen::Matrix<float, s, k> l2_t;
	//обучаемые параметры
	std::unique_ptr<l1_t> layer1;
	std::unique_ptr<l2_t> layer2;
	Vec_h bias1;
	Vec_out bias2;

	NN2() : layer1(std::make_unique<l1_t>()),
		layer2(std::make_unique<l2_t>()),
		bias1(), bias2() {
	}

	//инициализировать обучаемые параметры случайными значениями
	void init_rand() {
		layer1->noalias() = l1_t::Random();
		layer2->noalias() = l2_t::Random();
		bias1.noalias() = Vec_h::Random();
		bias2.noalias() = Vec_out::Random();
	}

	static Vec_h sigmoid(const Vec_h& input) {
		return 1.0f / (1.0f + (-input.array()).exp());
	}
	
	//функция активации скрытого слоя
	static Vec_h silu(const Vec_h& input) {
		return input.array() * sigmoid(input).array();
	}

	static Vec_h silu(Vec_h&& input) {
		return input.array() * sigmoid(input).array();
	}

	//производная silu
	static Vec_h silu_d(const Vec_h& input) {
		Vec_h sig = sigmoid(input);
		return (sig.array() * (1.0f + input.array() * (1.0f - sig.array())));
	}

	//функция активации выходного слоя
	static void softmax(Vec_out& inplace) {
		inplace = (inplace.array() - inplace.maxCoeff()).exp();
		inplace /= inplace.sum();
	}

	//выполнить только прямой проход вычисления
	Vec_out forward(const Vec_in& input) const {
		Vec_h hidden = silu(layer1->transpose() * input + bias1);
		Vec_out out = layer2->transpose() * hidden + bias2;
		softmax(out);
		return out;
	}

	//функция потерь классификатора (в данном случае для вывода в консоль)
	static float crossentropy_loss(const Vec_out& pred, const Vec_out& real_out) {
		return -(real_out.array() * log(pred.array() + 1e-9f)).sum();
	}

	//выполнить прямой и обратный проходы, вычислить и отмасштабировать градиент
	//применяет графиент к обучаемым параметрам
	//принимает ссылку для записи выходного значения,
	//входной вектор и ожидаемый выходной вектор,
	//а также скорость обучени (масштаб градиента)
	float forward_backward(Vec_out& pred, const Vec_in& input, 
		const Vec_out& real_out, float lr)
	{
		//прямой проход
		Vec_h hidden = layer1->transpose() * input + bias1;
		Vec_h hidden_post_silu = silu(hidden);
		pred.noalias() = layer2->transpose() * hidden_post_silu + bias2;
		softmax(pred);
		float loss = crossentropy_loss(pred, real_out);
		//функция потерь и градиент softmax вырождаются
		Vec_out out_d = pred - real_out; //Derivative and loss cancel out
		//обратный проход
		bias2.noalias() -= lr * out_d;
		layer2->noalias() -= lr * (hidden_post_silu * out_d.transpose());
		Vec_h h_d = ((*layer2) * out_d).array() * silu_d(hidden).array();
		bias1.noalias() -= lr * h_d;
		layer1->noalias() -= lr * (input * h_d.transpose());
		return loss;
	}

	//получить индекс наиболшего значения класса
	static int vec2class(const Vec_out& fout) {
		size_t idx = 0;
		fout.maxCoeff(&idx);
		return idx;
	}

	//вычислить класс входящего изображения
	int predict(const Vec_in& input) const {
		return vec2class(forward(input));
	}

	//выполняет эпоху обучения с заданной скоростью обучения
	//принимает входнящие вектора изображений и номера их классов, 
	//также скорость обучения
	//возвращает среднюю величину функции потерь
	//выводит в stdout среднюю величину функции потерь и метрики accuracy
	float train(const std::vector<Vec_in>& x, const std::vector<int>& y, float lr) {
		const size_t sz = x.size();
		Vec_out pred{};
		size_t valid = 0;
		assert(sz == y.size());
		Vec_h bg1{};
		Vec_out bg2{};
		float loss_avg = 0.0f, accuracy;
		for (int i = 0; i < sz; ++i) {
			Vec_out real_out = Vec_out::Zero();
			real_out[y[i]] = 1.0f;
			float loss = forward_backward(pred, x[i], real_out, lr);
			loss_avg += loss;
			if (y[i] == vec2class(pred)) valid++;
		}
		loss_avg /= (float)sz;
		accuracy = (float)valid / (float)sz;
		std::cout << "loss: " << loss_avg << ' ' <<
			"accuracy: " << accuracy << std::endl;
		return loss_avg;
	}

	//выполняет тестирование на тестовой выборке
	//принимает входнящие вектора изображений и номера их классов
	//возвращает среднюю величину функции потерь
	//выводит в stdout среднюю величину функции потерь и метрики accuracy
	float validate_score(const std::vector<Vec_in>& x, const std::vector<int>& y) {
		float loss_avg = 0.0f, accuracy;
		const size_t sz = x.size();
		size_t valid = 0;
		Vec_out pred;
		assert(sz == y.size());
		for (size_t i = 0; i < sz; ++i) {
			Vec_out real_out = Vec_out::Zero();
			real_out[y[i]] = 1.0f;
			pred = forward(x[i]);
			float loss = crossentropy_loss(pred, real_out);
			loss_avg += loss;
			if (y[i] == vec2class(pred)) valid++;
		}
		loss_avg /= (float)sz;
		accuracy = (float)valid / (float)sz;
		std::cout << "loss: " << loss_avg << ' ' <<
			"accuracy: " << accuracy << std::endl;
		return loss_avg;
	}

	//записывает обучаемые параметры в файл
	void save_weights(const char* filename) const {
		uint32_t crc_buf[256];
		std::ofstream file(filename, std::ios::out | std::ios::trunc | std::ios::binary);
		uint32_t rows = layer1->rows(), cols = layer1->cols(),
			rows2 = layer2->rows(), cols2 = layer2->cols();

		//вычисление контрольной суммы
		crc32::generate_table(crc_buf);
		uint32_t crc = crc32::update(crc_buf, 0, "NN2", 4);
		crc = crc32::update(crc_buf, crc, &rows, 4);
		crc = crc32::update(crc_buf, crc, &cols, 4);
		crc = crc32::update(crc_buf, crc, &cols2, 4);
		crc = crc32::update(crc_buf, crc, layer1->data(), 4ull * rows * cols);
		crc = crc32::update(crc_buf, crc, bias1.data(), 4ull * cols);
		crc = crc32::update(crc_buf, crc, layer2->data(), 4ull * rows2 * cols2);
		crc = crc32::update(crc_buf, crc, bias2.data(), 4ull * cols2);

		//запись
		file.write("NN2", 4);
		file.write(reinterpret_cast<const char*>(&rows), 4);
		file.write(reinterpret_cast<const char*>(&cols), 4);
		file.write(reinterpret_cast<const char*>(&cols2), 4);
		file.write(reinterpret_cast<const char*>(layer1->data()), 4ull * rows * cols);
		file.write(reinterpret_cast<const char*>(bias1.data()), 4ull * cols);
		file.write(reinterpret_cast<const char*>(layer2->data()), 4ull * rows2 * cols2);
		file.write(reinterpret_cast<const char*>(bias2.data()), 4ull * cols2);
		file.write(reinterpret_cast<const char*>(&crc), 4);

		//проверка корректности размера файла
		if(file.tellp() != 4ull * (5ull + (uint64_t)rows * cols + cols + (uint64_t)rows2 * cols2 + cols2))
			throw std::runtime_error("Wrong output file size");
		file.close();
	}

	//загружает обучаемые параметры из файла
	void load_weights(const char* filename) {
		uint32_t crc_buf[256];
		std::ifstream file(filename, std::ios::binary);
		if (!file)
			throw std::runtime_error("Cannot open file");
		uint32_t rows = layer1->rows(), cols = layer1->cols(),
			rows2 = layer2->rows(), cols2 = layer2->cols(), crc0;

		//проверка корректности размера файла
		file.seekg(0, std::ios::end);
		if (file.tellg() != 4ull * (5ull + (uint64_t)rows * cols + cols + (uint64_t)rows2 * cols2 + cols2))
			throw std::runtime_error("Wrong input file size");
		file.seekg(0);

		//проверка соответствия формата весов
		char magic[4];
		file.read(magic, 4);
		if (std::memcmp("NN2", magic, 4))
			throw std::runtime_error("Wrong file format");
		file.read(reinterpret_cast<char*>(&rows), 4);
		file.read(reinterpret_cast<char*>(&cols), 4);
		file.read(reinterpret_cast<char*>(&cols2), 4);
		if (rows != layer1->rows() || cols != layer1->cols() || cols2 != layer2->cols())
			throw std::runtime_error("Wrong nn format");

		//чтение
		file.read(reinterpret_cast<char*>(layer1->data()), 4ull * rows * cols);
		file.read(reinterpret_cast<char*>(bias1.data()), 4ull * cols);
		file.read(reinterpret_cast<char*>(layer2->data()), 4ull * cols * cols2);
		file.read(reinterpret_cast<char*>(bias2.data()), 4ull * cols2);
		file.read(reinterpret_cast<char*>(&crc0), 4);
		file.close();

		//проверка контрольной суммы
		crc32::generate_table(crc_buf);
		uint32_t crc = crc32::update(crc_buf, 0, "NN2", 4);
		crc = crc32::update(crc_buf, crc, &rows, 4);
		crc = crc32::update(crc_buf, crc, &cols, 4);
		crc = crc32::update(crc_buf, crc, &cols2, 4);
		crc = crc32::update(crc_buf, crc, layer1->data(), 4ull * rows * cols);
		crc = crc32::update(crc_buf, crc, bias1.data(), 4ull * cols);
		crc = crc32::update(crc_buf, crc, layer2->data(), 4ull * rows2 * cols2);
		crc = crc32::update(crc_buf, crc, bias2.data(), 4ull * cols2);
		if (crc0 != crc)
			throw std::runtime_error("CRC32 checksum test failed!");
	}
};
