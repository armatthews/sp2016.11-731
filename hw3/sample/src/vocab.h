#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "utils.h"
using namespace std;

class Vocabulary {
public:
  Vocabulary();
  WordId Convert(const string& w);
  string Convert(WordId i) const;
  vector<WordId> Convert(const vector<string>& ws);
  vector<string> Convert(const vector<WordId>& is) const;
  bool Contains(const string& w) const;
  void Freeze();
private:
  bool frozen;
  unordered_map<string, WordId> w2i;
  vector<string> i2w;
};

