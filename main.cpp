#include <boost/program_options.hpp>
#define EIGEN_STACK_ALLOCATION_LIMIT LLONG_MAX
#include "NN.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

namespace po = boost::program_options;

//нейронная сеть для датасета Mnist
//для классификации рукописных цифр
typedef NN2<28, 28, 800, 10> MnistNN;

//читает CSV файл датасета в массивы x и у.
//Разделитель чисел ',', одна запись в строке,
//первое число - номер класса, остальные: 
//значения яркости пикселя входного вектора
//от 0 до 255
static void ReadCsv(std::vector<MnistNN::Vec_in>& x, std::vector<int>& y, const char* filename) {
	std::ifstream file(filename);
	std::string line, cell;
	if (!file)
		throw std::runtime_error("Cannot open file");

	while (std::getline(file, line)) {
		size_t cnt = 0, sz = x.size();
		std::stringstream ll(line);
		std::getline(ll, cell, ',');
		y.push_back(std::atoi(cell.c_str()));
		x.resize(sz + 1);
		MnistNN::Vec_in& tmp = x.at(sz);
		while (std::getline(ll, cell, ',')) {
			tmp[cnt++] = static_cast<float>(std::atoi(cell.c_str())) / 255.0f;
		}
	}
}

int main(int argc, char *argv[]) {
	try {
		po::options_description desc("Program Options");
		desc.add_options()
			("help,h", "this help messsage")
			("epochs,N", po::value<uint32_t>(), "set training epochs number")
			("lr", po::value<float>(), "set learning rate")
			("cout,c", "pipe results to stdout")
			("seed", po::value<uint32_t>(), "rand init seed")
			("p_args", po::value<std::vector<std::string>>(), "mode and file paths")
		;
		po::positional_options_description pos{};
		pos.add("p_args", -1);
		po::variables_map vm{};
		po::store(po::command_line_parser(argc, argv).options(desc).
			positional(pos).run(), vm);
		po::notify(vm);

		//помощь по формату аргументов в программе
		if (vm.count("help")) {
			std::cout << "Usage:\n" <<
				"\tprogram train <train>.csv <validate>.csv <weights_out>.bin [options]\n" <<
				"\tprogram test <test>.csv <weights>.bin [options]\n" <<
				"\tprogram infer <numbers>.csv <weights>.bin <results>.txt/--cout [options]\n" <<
				desc << std::endl;
			return 0;
		}

		if (vm.count("p_args") == 0) {
			std::cout << "Usage: program --help" << std::endl;
			return -1;
		}

		const auto p_args = vm["p_args"].as<std::vector<std::string>>();

		MnistNN nn{};

		//режим обучения и записи весов в файл
		//принимает пути к файлам, скорость обучения, количество 
		//эпох и сид генератора случайных чисел
		if (p_args[0] == "train") {
			if (p_args.size() != 4) {
				std::cout << "Usage: program train <train>.csv <validate>.csv <weights>.bin [options]" << std::endl;
				return -1;
			}

			std::vector<MnistNN::Vec_in> train_x{}, test_x{};
			std::vector<int> train_y{}, test_y{};
			const uint32_t epoch = vm.count("epochs") ? vm["epochs"].as<uint32_t>() : 100u;
			const float lr = vm.count("lr") ? vm["lr"].as<float>() : 1e-4f;
			const uint32_t seed = vm.count("seed") ? vm["seed"].as<uint32_t>() : time(0);

			srand(seed);
			nn.init_rand();

			std::cout << "Loading train..." << std::endl;
			ReadCsv(train_x, train_y, p_args[1].c_str());
			std::cout << "Loading validate..." << std::endl;
			ReadCsv(test_x, test_y, p_args[2].c_str());

			std::cout << "Initial: ";
			nn.validate_score(test_x, test_y);

			for (uint32_t i = 0; i < epoch; ++i) {
				std::cout << "Epoch " << i << ": ";
				nn.train(train_x, train_y, lr);
				std::cout << "Val " << i << ": ";
				nn.validate_score(test_x, test_y);
			}
			std::cout << "Saving..." << std::endl;
			nn.save_weights(p_args[3].c_str());
			std::cout << "Result: ";
			nn.validate_score(test_x, test_y);
			return 0;
		}

		//режим тестирования и вывода метрик на тестовом датасете
		//принимает пути к файлам
		else if (p_args[0] == "test") {
			if (p_args.size() != 3) {
				std::cout << "Usage: program test <test>.csv <weights>.bin [options]" << std::endl;
				return -1;
			}

			std::vector<MnistNN::Vec_in> test_x{};
			std::vector<int> test_y{};

			std::cout << "Loading weights..." << std::endl;
			nn.load_weights(p_args[2].c_str());
			std::cout << "Loading test..." << std::endl;
			ReadCsv(test_x, test_y, p_args[1].c_str());

			std::cout << "Result: ";
			nn.validate_score(test_x, test_y);
			return 0;
		}

		//режим использования, принимает числовые данные (класс игнорируется)
		//выводит по одному номеру класса в строке в файл или stdout
		else if (p_args[0] == "infer") {
			std::ofstream file{};
			bool std_out = vm.count("cout");

			if (p_args.size() != (std_out ? 3 : 4)) {
				std::cout << "Usage: infer <numbers>.csv <weights>.bin <results>.txt/--cout [options]" << std::endl;
				return -1;
			}
			if (!std_out) file.open(p_args[3], std::ios::out | std::ios::trunc);
			std::ostream& out = std_out ? std::cout : file;

			std::vector<MnistNN::Vec_in> test_x{};
			std::vector<int> test_y{};

			std::cout << "Loading weights..." << std::endl;
			nn.load_weights(p_args[2].c_str());
			std::cout << "Loading data..." << std::endl;
			ReadCsv(test_x, test_y, p_args[1].c_str());

			const size_t sz = test_x.size();
			assert(sz == test_y.size());
			for (size_t i = 0; i < sz; ++i) {
				out << nn.predict(test_x[i]) << '\n';
			}
			return 0;
		}
		else {
			std::cout << "Usage: program --help" << std::endl;
			return -1;
		}
	}
	catch (std::exception& e) {
		std::cout << "ERROR: " << e.what() << std::endl;
		return -2;
	}
}
