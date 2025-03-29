//N Num Plus
#include<iostream>
#include<vector>
#include<cstdio>
#include<ctime>
#include<string>
#include<windows.h>
#include<cmath>
#include<fstream>
#include<algorithm>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

LARGE_INTEGER GetPerformanceCounter() {
	LARGE_INTEGER counter;
	QueryPerformanceCounter(&counter);
	return counter;
}

// 计算两个LARGE_INTEGER时间点的差值（单位：微秒）
double GetElapsedTime(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER frequency) {
	return (end.QuadPart - start.QuadPart) * 1000000.0 / frequency.QuadPart;
}

double NavieSum(vector<double>A){
	int n=A.size();
	double sum=0;
	for(int i=0;i<n;i++){
		sum+=A[i];
	}
	return sum;
}

double PairSum(vector<double>A){
	size_t m=A.size();
	while(m>1){
		size_t pos=m/2;//标志位
		for(int i=0;i<pos;i++){
			A[i]=A[i*2]+A[i*2+1];
		}
		if (m % 2 == 1) {//处理特殊情况
			A[pos] = A[m - 1];
			m = pos + 1;
		} else {
			m = pos;
		}
	}
	return A[0];
}

double UrollSum(vector<double>A){
	double sum1=0,sum2=0,sum3=0,sum4=0;
	size_t j=0;
	size_t n=A.size();
	for(j;j+3<n;j+=4){
		sum1+=A[j];
		sum2+=A[j+1];
		sum3+=A[j+2];
		sum4+=A[j+3];
	}
	double sum=sum1+sum2+sum3+sum4;
	for(j;j<n;j++){
		sum+=A[j];
	}
	return sum;
}

double Blank(vector<double>A){
	return A[0];
}

//实验数据生成与读取，此方法由上个脚本中询问chatgpt得到的函数修改获得
bool fileExists(const string& filename) {
	return fs::exists(filename);
}
void generateTestData(const vector<size_t>& sizes, const string& dataFile) {
	ofstream outFile(dataFile, ios::binary);
	if (!outFile) {
		cerr << "无法打开文件 " << dataFile << " 进行写入。" << endl;
		return;
	}
	
	size_t count = sizes.size();
	outFile.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
	
	for (size_t size : sizes) {
		vector<double> arr(size);
		for (size_t i = 0; i < size; ++i) {
			arr[i] = rand() % 100;  
		}
		// 写入当前数组的规模（元素个数）
		outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
		// 写入数组数据
		outFile.write(reinterpret_cast<const char*>(arr.data()), size * sizeof(double));
	}
	outFile.close();
}

void readTestData(const string& dataFile, vector<vector<double>>& arrays) {
	ifstream inFile(dataFile, ios::binary);
	if (!inFile) {
		cout << "无法打开文件 " << dataFile << " 进行读取。" << endl;
		return;
	}
	
	size_t count;
	inFile.read(reinterpret_cast<char*>(&count), sizeof(size_t));
	arrays.clear();
	
	for (size_t i = 0; i < count; i++) {
		size_t size;
		inFile.read(reinterpret_cast<char*>(&size), sizeof(size_t));
		vector<double> arr(size);
		inFile.read(reinterpret_cast<char*>(arr.data()), size * sizeof(double));
		arrays.push_back(arr);
	}
	inFile.close();
}


int main(){
	vector<size_t> sizes={1024, 1035,2048, 4096, 8192, 16384, 
		32768, 65536,131072,231311,262144,524288,2313119};
	//加一些特例测试不是2的整数次幂对算法的影响
	const string dataFile = "array_data.bin";
	
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	
	vector<vector<double>> testData;
	
	if(fileExists(dataFile)){
		readTestData(dataFile,testData);
	}
	else{
		srand(static_cast<unsigned int>(time(0)));
		generateTestData(sizes, dataFile);
		readTestData(dataFile, testData);
	}
	
	
	const int repetitions = 5;  // 重复次数
	
	for (size_t i = 0; i < sizes.size(); i++){
		size_t n = sizes[i];
		const auto & arr = testData[i];
		double resultNaive = 0, resultPair = 0, resultUroll = 0;
		double totalTimeNaive = 0, totalTimePair = 0, totalTimeUroll = 0,totalTimeCopy=0;
		//考虑到传参数时候的拷贝耗时问题
		//这里我们需要减去拷贝时间不然对pair是不公平的
		
		// 多次重复平均
		for (int rep = 0; rep < repetitions; rep++){
			
			LARGE_INTEGER start,end; 
			
			start = GetPerformanceCounter();
			resultNaive = NavieSum(arr);
			end = GetPerformanceCounter();
			totalTimeNaive += GetElapsedTime(start, end, freq);
			
		}

	}
	
	return 0;
}
