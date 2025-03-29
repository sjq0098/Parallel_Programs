//Martix mutiplied by a vector
#include<iostream>
#include<vector>
#include<cstdio>
#include<ctime>
#include<string>
#include<windows.h>
#include<fstream>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

//获取计数器的值
LARGE_INTEGER GetPer_formance_Counter() {
	LARGE_INTEGER counter;
	QueryPerformanceCounter(&counter);
	return counter;
}

// 计算时间差
double Get_Elapsed_Time(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER frequency) {
	return (end.QuadPart - start.QuadPart) * 1000000.0 / frequency.QuadPart;
}

//朴素矩阵乘以向量算法
void Naive_Alg(const vector<vector<double>>&Martix,
	const vector<double>&Vector,
	vector<double>&Result){
		int n= Martix.size();
		for(int i=0;i<n;i++){
			Result[i]=0;
			for(int j=0;j<n;j++){
				Result[i]+=Martix[j][i]*Vector[j];
			}
		}
	}

//catch优化算法
void Cache_Optimized_Alg(const vector<vector<double>>&Martix,
	const vector<double>&Vector,
	vector<double>&Result){
		int n=Martix.size();
		for(int i=0;i<n;i++){
			Result[i]=0;
		}
		for(int j=0;j<n;j++){
			for(int i=0;i<n;i++){
				Result[i]+=Martix[j][i]*Vector[j];
			}
		}
	}

//朴素算法Uroll优化
void Navie_Uroll_Alg(const vector<vector<double>>&Martix,
	const vector<double>&Vector,
	vector<double>&Result){
		int n=Martix.size();
		for(int i=0;i<n;i++){
			int j=0;
			Result[i]=0;
			for(j;j+3<n;j+=4){
				Result[i]+=Martix[j][i]*Vector[j];
				Result[i]+=Martix[j+1][i]*Vector[j+1];
				Result[i]+=Martix[j+2][i]*Vector[j+3];
				Result[i]+=Martix[j+3][i]*Vector[j+3];
			}
			// 处理不是4的倍数的情况
			for (j; j < n; ++j) {
				Result[i] += Martix[j][i] * Vector[j];
			}
		}
	}


//实验数据生成，用于第一次生成数据到一个文本中进行保存，后续只需要进行读取即可
//此方法系询问chatgpt给出
bool fileExists(const string& filename) {
	return fs::exists(filename);
}

// 生成并保存测试数据
void generateTestData(const vector<size_t>& sizes, const string& matrixFile, const string& vectorFile) {
	ofstream matrixOut(matrixFile, ios::binary);
	ofstream vectorOut(vectorFile, ios::binary);
	
	for (size_t size : sizes) {
		vector<vector<double>> matrix(size, vector<double>(size));
		vector<double> vec(size);
		
		// 填充矩阵和向量
		for (size_t i = 0; i < size; ++i) {
			vec[i] = rand() % 100;
			for (size_t j = 0; j < size; ++j) {
				matrix[i][j] = rand() % 100;
			}
		}
		
		// 写入矩阵数据
		for (const auto& row : matrix) {
			matrixOut.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
		}
		
		// 写入向量数据
		vectorOut.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(double));
	}
}

// 读取测试数据
void readTestData(const vector<size_t>& sizes, const string& matrixFile, const string& vectorFile,
	vector<vector<vector<double>>>& matrices, vector<vector<double>>& vectors) {
		ifstream matrixIn(matrixFile, ios::binary);
		ifstream vectorIn(vectorFile, ios::binary);
		
		for (size_t size : sizes) {
			vector<vector<double>> matrix(size, vector<double>(size));
			vector<double> vec(size);
			
			// 读取矩阵数据
			for (auto& row : matrix) {
				matrixIn.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
			}
			
			// 读取向量数据
			vectorIn.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(double));
			
			matrices.push_back(matrix);
			vectors.push_back(vec);
		}
	}

int main(){
	const string matrixFile = "matrix_data.bin";
	const string vectorFile = "vector_data.bin";
	vector<size_t> sizes = {128,150, 256, 512, 1024, 2048};
	
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	
	vector<vector<vector<double>>> matrices;
	vector<vector<double>> vectors;
	
	if (fileExists(matrixFile) && fileExists(vectorFile)) {
		readTestData(sizes, matrixFile, vectorFile, matrices, vectors);
	} else {
		generateTestData(sizes, matrixFile, vectorFile);
		readTestData(sizes, matrixFile, vectorFile, matrices, vectors);
	}
	
	
	for(size_t i=0;i<sizes.size();i++){
		size_t siz=sizes[i];
		const auto&martix=matrices[i];
		const auto&vec=vectors[i];
		vector<double> result(siz);
		
		LARGE_INTEGER Start=GetPer_formance_Counter();
		for(int j=0;j<1000;j++){
			Navie_Uroll_Alg(martix,vec,result);
		}
		LARGE_INTEGER End=GetPer_formance_Counter();
		double naive_time=Get_Elapsed_Time(Start,End,freq);
		
		cout << "Size: " << siz << "x" << siz << endl;
		cout << "Naive Uroll Time: " << naive_time << " us" << endl;
		cout << "--------------------------" << endl;
	}
	return 0;
}
