#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cassert>
#include <random>
#include "hw3.h"
#include "vocab.h"

using namespace std;
const string tm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/data/tm";
//const string tm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/sample/tm";
//const string lm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/data/lm";
const string lm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/sample/lm";

int main() {

  Vocabulary vocab;
  cerr << "Loading TM..." << endl;
  TM tm(tm_filename, vocab);
  vocab.Freeze();
  cerr << "Loading LM..." << endl;
  LM lm(lm_filename, vocab);
  cerr << "Done loading." << endl;

  for (string input_string; getline(cin, input_string);) {

    vector<string> pieces = tokenize(input_string, "|||");
    assert (pieces.size() == 2);
    vector<WordId> source = vocab.Convert(tokenize(strip(pieces[0]), " "));
    vector<WordId> target = vocab.Convert(tokenize(strip(pieces[1]), " "));
    cout << Score(source, target, tm, lm) << endl;
    //MatchingPhraseList mpl = FindMatchingPhrases(source, tm);
    //cout << Score(source, target, mpl, lm) << endl;
  }

  return 0;
}
