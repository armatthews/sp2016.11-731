#include <fstream>
#include "tm.h"

TM::TM(const string& filename, Vocabulary& vocab) {
  ifstream f(filename);
  assert (f.is_open());
  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    for (unsigned i = 0; i < parts.size(); ++i) {
      parts[i] = strip(parts[i]);
    }
    vector<WordId> src = vocab.Convert(tokenize(parts[0], " "));
    vector<WordId> tgt = vocab.Convert(tokenize(parts[1], " "));
    float prob = atof(parts[2].c_str());
    AddPhrase(src, tgt, prob);
  }
  f.close();
}

void TM::AddPhrase(const vector<WordId>& source, const vector<WordId>& target, float prob) {
  if (ttable.find(source) == ttable.end()) {
    ttable[source] = map<vector<WordId>, float>();
  }
  assert (ttable[source].find(target) == ttable[source].end() && "Invalid attempt to add an already-existant phrase to the phrase table!");
  ttable[source][target] = prob;
}
