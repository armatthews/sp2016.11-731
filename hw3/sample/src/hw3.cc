#include "hw3.h"

//extern Vocabulary vocab; //XXX
 
struct AlignmentInfo {
  unsigned ej;
  float logprob;
  unsigned fi, fj;
};

float logadd(float a, float b) {
  if (a > b) {
    // exp(a - a) * exp(a) + exp(b - a) * exp(a) = exp(a) (1 + exp(b - a))
    return log(exp(a) * (1 + exp(b - a)));
  }
  else {
    return log(exp(b) * (1 + exp(a - b)));
  }
}

tuple<LMState, float> AdvanceLM(LM& lm, const LMState& in_state, const vector<WordId> phrase) {
  LMState state = in_state;
  float phrase_logprob = 0.0;
  float word_logprob;
  for (WordId w : phrase) {
    tie(state, word_logprob) = lm.score(state, w);
    phrase_logprob += word_logprob;
  }
  return make_tuple(state, phrase_logprob);
}

float Score(const vector<WordId>& source, const vector<WordId>& target, TM& tm, LM& lm, bool end) {
  float sent_logprob = 0.0f;

  LMState lm_state = lm.begin();
  float lm_logprob = 0.0f;
  tie(lm_state, lm_logprob) = AdvanceLM(lm, lm_state, target);
  if (end) {
    lm_logprob += lm.end(lm_state);
  }
  sent_logprob += lm_logprob;

  vector<vector<AlignmentInfo>> alignments(target.size());
  for (unsigned fi = 0; fi < source.size(); ++fi) {
    for (unsigned fj = fi + 1; fj <= source.size(); ++fj) {
      vector<WordId> source_phrase;
      for (unsigned fk = fi; fk < fj; ++fk) {
        source_phrase.push_back(source[fk]);
      }
      // if f[fi:fj] in tm:
      if (tm.ttable.find(source_phrase) != tm.ttable.end()) {
        // for phrase in tm[f[fi:fj]]:
        for (auto it : tm.ttable[source_phrase]) {
          vector<WordId> target_phrase = get<0>(it);
          float phrase_logprob = get<1>(it);
          // for ei in xrange(len(e)+1-len(ephrase)):
          if (target.size() + 1 < target_phrase.size()) {
            continue;
          }
          for (unsigned ei = 0; ei < target.size() + 1 - target_phrase.size(); ++ei) {
            unsigned ej = ei + target_phrase.size();
            // if ephrase == e[ei:ej]
            bool match = true;
            for (unsigned ek = ei; ek < ej; ++ek) {
              if (target[ek] != target_phrase[ek - ei]) {
                match = false;
                break;
              }
            }
            if (match) {
              alignments[ei].push_back({ej, phrase_logprob, fi, fj});
            }
          }
        }
      }
    }
  }

  vector<unordered_map<Coverage, float>> chart(target.size() + 1);
  chart[0][0] = 0.0f;
  for (unsigned ei = 0; ei < target.size(); ++ei) {
    auto& sums = chart[ei];
    for (auto it = sums.begin(); it != sums.end(); ++it) {
      Coverage v = it->first;
      float lp = it->second;
      for (AlignmentInfo& ai : alignments[ei]) {
        Coverage c = MakeCoverage(ai.fi, ai.fj);
        if ((c & v) == 0) {
          Coverage new_v = c | v;
          if (chart[ai.ej].find(new_v) != chart[ai.ej].end()) {
            chart[ai.ej][new_v] = logadd(chart[ai.ej][new_v], lp + ai.logprob);
          }
          else {
            chart[ai.ej][new_v] = lp + ai.logprob;
          }
        }
      }
    }
  }

  Coverage goal = MakeCoverage(0, source.size());
  auto& final_chart = chart[target.size()];
  assert (final_chart.find(goal) != final_chart.end() && "ERROR: COULD NOT ALIGN SENTENCE!");
  sent_logprob += final_chart[goal];
  return sent_logprob;
}

MatchingPhraseList FilterPhraseList(const vector<WordId>& target, MatchingPhraseList& mpl) {
  MatchingPhraseList r;
  for (auto it : mpl) {
    r[it.first] = vector<MatchingPhrase>();
    for (MatchingPhrase& p : it.second) {
      if (p.target.size() > target.size()) {
        continue;
      }

      for (unsigned ei = 0; ei <= target.size() - p.target.size(); ++ei) {
        unsigned ej = ei + p.target.size();
        bool match = true;
        for (unsigned ek = ei; ek < ej; ++ek) {
          if (target[ek] != p.target[ek - ei]) {
            match = false;
            break;
          }
        }
        if (match) {
          r[it.first].push_back(p);
        }
      }
    }
  }
  return r;
}

float Score(const vector<WordId>& source, const vector<WordId>& target, MatchingPhraseList& in_mpl, LM& lm) {
  float sent_logprob = 0.0f;

  MatchingPhraseList mpl = FilterPhraseList(target, in_mpl);
  LMState lm_state = lm.begin();
  float lm_logprob = 0.0f;
  tie(lm_state, lm_logprob) = AdvanceLM(lm, lm_state, target);
  lm_logprob += lm.end(lm_state);
  sent_logprob += lm_logprob;

  vector<unordered_map<Coverage, float>> chart(target.size() + 1);
  chart[0][0] = 0.0f;
  for (unsigned ei = 0; ei < target.size(); ++ei) {
    auto& sums = chart[ei];
    for (auto it = sums.begin(); it != sums.end(); ++it) {
      Coverage v = it->first;
      float lp = it->second;
      for (auto it : mpl) {
        Coverage c = it.first;
        if ((c & v) != 0) {
          continue;
        }
        Coverage new_v = c | v;

        for (MatchingPhrase& p : it.second) {
          unsigned ej = ei + p.target.size();
          if (ej > target.size()) {
            continue;
          }
          bool match = true;
          for (unsigned ek = ei; ek < ej; ek++) {
            if (target[ek] != p.target[ek - ei]) {
              match = false;
              break;
            }
          }
          if (!match) {
            continue;
          }
          if (chart[ej].find(new_v) != chart[ej].end()) {
            chart[ej][new_v] = logadd(chart[ej][new_v], lp + p.logprob);
          }
          else {
            chart[ej][new_v] = lp + p.logprob;
          }
        }
      }
    }
  }

  Coverage goal = MakeCoverage(0, source.size());
  auto& final_chart = chart[target.size()];
  assert (final_chart.find(goal) != final_chart.end() && "ERROR: COULD NOT ALIGN SENTENCE!");
  sent_logprob += final_chart[goal];
  return sent_logprob;
}

MatchingPhraseList FindMatchingPhrases(const vector<WordId>& source, TM& tm) {
  MatchingPhraseList mpl;
  for (unsigned i = 0; i < source.size(); ++i) {
    for (unsigned j = i + 1; j <= source.size(); ++j) {
      vector<WordId> phrase = slice(source, i, j);
      if (tm.ttable.find(phrase) == tm.ttable.end()) {
        continue;
      }

      Coverage coverage = MakeCoverage(i, j);
      if (mpl.find(coverage) == mpl.end()) {
        mpl[coverage] = vector<MatchingPhrase>();
      }
      for (auto it : tm.ttable[phrase]) {
        vector<WordId> target = it.first;
        float logprob = it.second;
        mpl[coverage].push_back({i, j, target, logprob});
      }
    }
  }
  return mpl;
}

float GhettoLMCost(const vector<WordId>& target, LM& lm) {
  LMState state;
  float word_score, total_score = 0.0f;
  for (WordId w : target) {
    tie(state, word_score) = lm.score(state, w);
    total_score += word_score;
  }
  return total_score;
}

vector<vector<float>> ComputeFutureCosts(const vector<WordId>& source, MatchingPhraseList& mpl, LM& lm) {
  const float inf = -std::numeric_limits<float>::infinity();
  vector<vector<float>> tm_costs(source.size() + 1, vector<float>(source.size() + 2, inf));
  vector<vector<float>> future_costs(source.size() + 1, vector<float>(source.size() + 2, inf));
  vector<vector<vector<WordId>>> targets(source.size() + 1, vector<vector<WordId>>(source.size() + 2));
  for (auto& it : mpl) {
    for (MatchingPhrase& p : it.second) {
      float cost = p.logprob + GhettoLMCost(p.target, lm);
      if (cost > future_costs[p.i][p.j]) {
        future_costs[p.i][p.j] = cost;
        targets[p.i][p.j] = p.target;
        tm_costs[p.i][p.j] = p.logprob;
      }
    }
  }
  // assert (vocab.Convert("</s>") == 2);
  targets[source.size()][source.size() + 1] = vector<unsigned>(1, 2);
  future_costs[source.size()][source.size() + 1] = lm.end(LMState());
  tm_costs[source.size()][source.size() + 1] = 0.0f;

  for (unsigned len = 2; len <= source.size() + 1; ++len) {
    for (unsigned i = 0; i <= source.size() + 1 - len; ++i) {
      unsigned j = i + len;
      for (unsigned k = i + 1; k < j; ++k) {
        assert (future_costs[i][k] != inf);
        assert (future_costs[k][j] != inf);
        vector<WordId> target = targets[i][k];
        target.insert(target.end(), targets[k][j].begin(), targets[k][j].end());
        float tm_cost = tm_costs[i][k] + tm_costs[k][j];
        float cost = tm_cost + GhettoLMCost(target, lm);
        if (cost > future_costs[i][j]) {
          future_costs[i][j] = cost;
          tm_costs[i][j] = tm_cost;
          assert (target.size() != 0);
          targets[i][j] = target;
        }
      }
    }
  }

  /*cerr << "Future costs table:" << endl;
  for (unsigned i = 0; i < source.size(); ++i) {
    for (unsigned j = i + 1; j <= source.size(); ++j) {
      cerr << "[" << i << ", " << j << ")\t" << future_costs[i][j] << "\t";
      for (WordId w : targets[i][j]) {
        cerr << vocab.Convert(w) << " ";
      }
      cerr << "\b\n";
    }
  }*/
  return future_costs;
}

float GetFutureCost(const vector<vector<float>>& future_costs, Coverage coverage) {
  //cerr << "Getting future cost for " << coverage << endl;
  float future_cost = 0.0f;

  unsigned i, j;
  bool in_uncovered_span = false;
  for (j = 0; j < future_costs.size(); ++j) {
    Coverage c = MakeCoverage(j, j + 1);
    if ((coverage & c) == 0 && !in_uncovered_span) {
      i = j;
      in_uncovered_span = true;
    }
    else if ((coverage & c) != 0 && in_uncovered_span) {
      future_cost += future_costs[i][j];
      //cerr << " + cost[" << i << ", " << j << "] = " << future_costs[i][j] << endl;
      in_uncovered_span = false;
    }
  }
  if (in_uncovered_span) {
    j = future_costs.size();
    //cerr << " + cost[" << i << ", " << j << "] = " << future_costs[i][j] << endl;
    future_cost += future_costs[i][j];
  }

  return future_cost;
}
