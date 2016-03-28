#include <fstream>
#include "lm.h"

LM::LM(const string& filename, Vocabulary& vocab) {
  ifstream f(filename);
  assert (f.is_open());

  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "\t");
    if (parts.size() > 1 and parts[0] != "ngram") {
      float logprob = atof(parts[0].c_str());
      vector<string> ngram_string = tokenize(strip(parts[1]), " ");
      vector<WordId> ngram = vocab.Convert(ngram_string);
      float backoff = (parts.size() > 2) ? atof(parts[2].c_str()) : 0.0f;

      bool discard = false;
      for (WordId w : ngram) {
        if (w == 0) {
          discard = true;
        }
      }
      if (!discard) {
        model[ngram] = make_tuple(logprob, backoff);
      }
    }
  }
}

LMState LM::begin() {
  return LMState(1, 1);
}

tuple<LMState, float> LM::score(const LMState& state, WordId word) {
  vector<WordId> ngram = state;
  ngram.push_back(word);
  float score = 0.0;
  float logprob, backoff;
  while (ngram.size() > 0) {
    if (model.find(ngram) != model.end()) {
      tie(logprob, backoff) = model[ngram];
      LMState new_state;
      unsigned start = (unsigned)max((int)ngram.size() - (int)LM_ORDER + 1, 0);
      for (unsigned i = start; i < ngram.size(); ++i) {
        new_state.push_back(ngram[i]);
      }
      return make_tuple(new_state, score + logprob);
    }
    else { // backoff
      if (ngram.size() > 0) {
        vector<WordId> ngram_bo = ngram;
        ngram_bo.pop_back();
        tie(logprob, backoff) = model[ngram_bo];
        score += backoff;
      }
      ngram.erase(ngram.begin());
    }
  }
  tie(logprob, backoff) = model[LMState(1, 0)]; // Look up UNK
  return make_tuple(LMState(), score + logprob);
}

float LM::end(const LMState& state) {
  LMState new_state;
  float logprob;
  tie(new_state, logprob) = score(state, 2);
  return logprob;
}
