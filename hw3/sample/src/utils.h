#pragma once
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
using namespace std;

typedef unsigned WordId;
typedef unsigned Coverage;

Coverage MakeCoverage(unsigned i, unsigned j);

vector<string> tokenize(string input, string delimiter, unsigned max_times);
vector<string> tokenize(string input, string delimiter);
vector<string> tokenize(string input, char delimiter);

string strip(const string& input);
vector<string> strip(const vector<string>& input, bool removeEmpty);

vector<WordId> slice(const vector<WordId>& v, unsigned i, unsigned j);
