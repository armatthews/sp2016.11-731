#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cassert>
#include <random>
#include "lm.h"
#include "tm.h"
#include "vocab.h"
#include "utils.h"
#include "hw3.h"

using namespace std;
namespace po = boost::program_options;

random_device rd;
mt19937 rng(rd());
const string tm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/data/tm";
//const string tm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/sample/tm";
//const string lm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/data/lm";
const string lm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/sample/lm";

Vocabulary vocab;

struct PartialHypothesis {
  LMState lm_state;
  Coverage coverage;
  PartialHypothesis* parent;
  vector<WordId> latest_target_phrase;

  bool operator==(const PartialHypothesis& o) {
    return lm_state == o.lm_state && coverage == o.coverage;
  }
};

class KbestList {
public:
  KbestList() {}
  KbestList(unsigned size) : size(size) {}
  bool Add(float score, const PartialHypothesis& hyp);
  list<pair<float, PartialHypothesis>> kbest;
  unsigned size;
};

bool KbestList::Add(float score, const PartialHypothesis& hyp) {
  if (kbest.size() >= size && score < kbest.back().first) {
    return false;
  }
  else {
    auto it2 = kbest.begin();
    for (; it2 != kbest.end(); ++it2) { 
      if (it2->second == hyp) {
       // Recombine if two things are the same and the new one is better
        if (score > it2->first) {
          kbest.erase(it2);
          break;
        }
        // If the old one is better, just throw away the new one
        else {
          return false;
        }
      }
    }

    auto it = kbest.rbegin();
    while (it != kbest.rend() && it->first < score) {
      ++it;
    }

    kbest.insert(it.base(), make_pair(score, hyp));

    if (kbest.size() > size) {
      kbest.pop_back();
    }

    return true;
  }
}

unsigned Population(const Coverage cov) {
  Coverage c = cov;
  unsigned p = 0;
  while (c > 0) {
    c &= (c - 1);
    ++p;
  }
  return p;
}

string ConvertHypothesis(PartialHypothesis* hyp) {
  vector<WordId> target;
  while (hyp != nullptr) {
    vector<WordId>& p = hyp->latest_target_phrase;
    target.insert(target.end(), p.rbegin(), p.rend());
    hyp = hyp->parent;
  }
  reverse(target.begin(), target.end());

  vector<string> words;
  for (WordId w : target) {
    words.push_back(vocab.Convert(w));
  }
  return boost::algorithm::join(words, " ");
}

vector<WordId> translate(const vector<WordId>& source, const unsigned beam_size, TM& tm, LM& lm) {
  MatchingPhraseList mpl = FindMatchingPhrases(source, tm);
  vector<vector<float>> future_costs = ComputeFutureCosts(source, mpl, lm);
  vector<WordId> target;
  vector<KbestList> stacks(source.size() + 1, KbestList(beam_size));
  PartialHypothesis root = {lm.begin(), 0, nullptr, vector<WordId>()};
  stacks[0].Add(GetFutureCost(future_costs, 0), root);
  assert (stacks[0].kbest.size() > 0);
  for (unsigned i = 1; i <= source.size(); ++i) {
    for (auto it : mpl) {
      unsigned src_len = Population(it.first);
      if (src_len > i) {
        continue;
      }
      unsigned j = i - src_len;

      for (MatchingPhrase& p : it.second) {
        for (auto& score_and_hyp : stacks[j].kbest) {
          float score = score_and_hyp.first;
          PartialHypothesis& hyp = score_and_hyp.second;
          if ((hyp.coverage & it.first) != 0) {
            continue;
          }
          LMState new_state;
          float lm_logprob;
          tie(new_state, lm_logprob) = AdvanceLM(lm, hyp.lm_state, p.target);
          if (i == source.size()) {
            lm_logprob += lm.end(new_state);
          }
          Coverage new_coverage = hyp.coverage | it.first;
          assert (Population(new_coverage) == i);
          PartialHypothesis new_hyp = {new_state, new_coverage, &hyp, p.target};
          float new_score = score + lm_logprob + p.logprob;
          new_score -= GetFutureCost(future_costs, hyp.coverage);
          new_score += GetFutureCost(future_costs, new_coverage);
          if (Population(new_coverage + 1) == 1 || true) {
            stacks[i].Add(new_score, new_hyp);
          }
        }
      }
    }
    /*cout << "Final stack #" << i << endl;
    for (auto it : stacks[i].kbest) {
      cout << "  " << it.first << "\t";
      cout << it.second.coverage << "\t<";
      for (WordId w : it.second.lm_state) {
        cout << vocab.Convert(w) << " ";
      }
      cout << "\b>\t";
      cout << ConvertHypothesis(&it.second) << endl;
    }
    cout << endl;*/
  }

  assert (stacks[source.size()].kbest.size() > 0);

  PartialHypothesis* best = &stacks[source.size()].kbest.front().second;
  while (best != nullptr) {
    vector<WordId>& p = best->latest_target_phrase;
    target.insert(target.end(), p.rbegin(), p.rend());
    best = best->parent;
  }
  
  reverse(target.begin(), target.end());
  return target;
}

int main(int argc, char** argv) {
  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("stack_size,s", po::value<unsigned>()->default_value(1), "Training bitext in source_tree ||| target format");
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  //Vocabulary vocab; // Temporarily made global, for debugging purposes
  cerr << "Loading TM..." << endl;
  TM tm(tm_filename, vocab);
  vocab.Freeze();
  cerr << "Loading LM..." << endl;
  LM lm(lm_filename, vocab);
  cerr << "Done loading." << endl;

  const unsigned beam_size = vm["stack_size"].as<unsigned>();

  for (string input_string; getline(cin, input_string);) {
    vector<WordId> source = vocab.Convert(tokenize(input_string, " "));
    vector<WordId> target = translate(source, beam_size, tm, lm);
    cout << Score(source, target, tm, lm) << " ||| ";
    for (unsigned i = 0; i < target.size(); ++i) {
      if (i != 0) { cout << " "; }
      cout << vocab.Convert(target[i]);
    }
    //cout << " ||| " << Score(source, target, tm, lm);
    cout << endl;
  }

  return 0;
}
