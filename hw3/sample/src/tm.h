#pragma once
#include <map>
#include "vocab.h"
#include "utils.h"

class TM {
public:
  TM(const string& filename, Vocabulary& vocab);
  void AddPhrase(const vector<WordId>& source, const vector<WordId>& target, float prob);
//private:
  map<vector<WordId>, map<vector<WordId>, float>> ttable;
};
