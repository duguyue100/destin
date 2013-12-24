/*
 * DeSTInCreateLabel.cpp
 *
 *  Created on: Dec 24, 2013
 *      Author: Yuhuang Hu (ARLAB)
 */

#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdio>
#include<cstdlib>
#include<vector>

using namespace std;

int main(void)
{
	//ifstream fin("trainingclassLabels.txt");
	ifstream fin("testingclassLabels.txt");
	//ofstream fout("trainingclassLabel.txt");
	ofstream fout("testingclassLabel.txt");

	vector<string> dataName;
	string className;

	while (fin >> className)
	{
		dataName.push_back(className);
	}

	fin.close();
	//fin.open("training.txt");
	fin.open("testing.txt");

	int dataLength=dataName.size();

	for (int i=1;i<=dataLength;i++)
	{
		string currClass=dataName[i-1];
		string imageName;
		fin >> imageName;
		while (!fin.eof() && imageName.find(currClass)!=string::npos)
		{
			fout << i << endl;
			fin >> imageName;
		}
		if (i!=1) fout << i << endl;
	}

	fin.close();
	fout.close();

	return 0;
}
