#pragma once
#include <map>
#include <vector>
#include "vocab.h"

using namespace std;

const unsigned int LM_ORDER = 3;

typedef vector<WordId> LMState;

class LM {
public:
  LM(const string& filename, Vocabulary& vocab);
  LMState begin();
  tuple<LMState, float> score(const LMState& state, WordId word);
  float end(const LMState& state);
  
private:
  map<vector<WordId>, tuple<float, float>> model;
};

