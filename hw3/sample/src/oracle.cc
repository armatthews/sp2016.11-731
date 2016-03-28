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
const string tm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/data/tm.bak";
//const string tm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/sample/tm";
//const string lm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/data/lm";
const string lm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/sample/lm";

Vocabulary vocab;
typedef vector<WordId> PartialHypothesis;

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

vector<WordId> translate(const vector<WordId>& source, unsigned beam_size, TM& tm, LM& lm) {
  MatchingPhraseList mpl = FindMatchingPhrases(source, tm);

  map<Coverage, KbestList> table;
  for (unsigned len = 1; len <= source.size(); ++len) {
    for (unsigned i = 0; i <= source.size() - len; ++i) {
      unsigned j = i + len;
      Coverage c = MakeCoverage(i, j);
      vector<WordId> slice(j - i);
      for (unsigned k = i; k < j; ++k) {
        slice[k - i] = source[k];
      }
      table[c] = KbestList(beam_size);

      for (const MatchingPhrase& p : mpl[c]) {
        float score = Score(slice, p.target, tm, lm);
        table[c].Add(score, p.target);
      }

      for (unsigned k = i + 1; k < j; ++k) {
        for (const auto& left : table[MakeCoverage(i, k)].kbest) {
          for (const auto& right : table[MakeCoverage(k, j)].kbest) {
            bool end = (j == source.size());
            end = true;
            float score;
            vector<WordId> target;
            target.insert(target.end(), left.second.begin(), left.second.end());
            target.insert(target.end(), right.second.begin(), right.second.end());
            score = Score(slice, target, tm, lm, end);
            table[c].Add(score, target);

            target.clear();
            target.insert(target.end(), right.second.begin(), right.second.end());
            target.insert(target.end(), left.second.begin(), left.second.end());
            score = Score(slice, target, tm, lm, end);
            table[c].Add(score, target);
          }
        }
      }

      for (auto it = table[c].kbest.begin(); it != table[c].kbest.end(); ++it) {
        cerr << "[" << i << ", " << j << ") ||| " << it->first << " |||";
        for (WordId w : slice) { cerr << " " << vocab.Convert(w); }
        cerr << " |||";
        for (WordId w : it->second) { cerr << " " << vocab.Convert(w); }
        cerr << endl;
        break;
      }

    }
  }

  /*for (unsigned len = 1; len < source.size() + 1; ++len) {
    for (unsigned i = 0; i < source.size() + 1 - len; ++i) {
      unsigned j = i + len;
      Coverage c = MakeCoverage(i, j);
      vector<WordId> slice(j - i);
      for (unsigned k = i; k < j; ++k) {
        slice[k - i] = source[k];
      }

      for (const auto& it : table[c].kbest) {
        cerr << "[" << i << ", " << j << ") ||| " << it.first << " |||";
        for (WordId w : slice) { cerr << " " << vocab.Convert(w); }
        cerr << " |||";
        for (WordId w : it.second) { cerr << " " << vocab.Convert(w); }
        cerr << endl;
        break;
      }
    }
  }*/

  Coverage goal = MakeCoverage(0, source.size());
  return table[goal].kbest.front().second;
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
