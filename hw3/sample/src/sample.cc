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
random_device rd;
mt19937 rng(rd());
const string tm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/data/tm";
//const string tm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/sample/tm";
//const string lm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/data/lm";
const string lm_filename = "/usr0/home/austinma/git/11731/austinma-2016/hw3/sample/lm";

Vocabulary vocab;

unsigned sample_multinomial(const vector<float>& logprobs) {
  assert (logprobs.size() > 0);
  float M = logprobs[0];
  for (unsigned i = 1; i < logprobs.size(); ++i) {
    M = max(logprobs[i], M);
  }

  float sum = 0.0;
  for (unsigned i = 0; i < logprobs.size(); ++i) {
    sum += exp(logprobs[i] - M);
  }

  float r = sum * ((float)rng() / RAND_MAX);
  for (unsigned i = 0; i < logprobs.size(); ++i) {
    float p = exp(logprobs[i] - M);
    if (r < p) {
      return i;
    }
    else {
      r -= p;
    }
  }
  return logprobs.size() - 1;
}

TreeNode::TreeNode(TreeNode* parent, const DecoderState& state, const MatchingPhrase& phrase, const MatchingPhraseList& mpl, LM& lm, const vector<vector<float>>& future_costs) {
  this->state = state;
  this->phrase = phrase;

  this->value = 0.0f;
  this->value = -1e50;
  this->visits = 0;

  this->parent = parent;
  InitializeChildren(mpl, lm, future_costs);
}

TreeNode::~TreeNode() {
  for (TreeNode* child : children) {
    delete child;
  }
}

void TreeNode::InitializeChildren(const MatchingPhraseList& mpl, LM& lm, const vector<vector<float>>& future_costs) {
  for (auto it : mpl) {
    Coverage c = it.first;
    if ((c & state.coverage) != 0) {
      continue;
    }

    for (MatchingPhrase& p : it.second) {
      LMState _;
      float score;
      //cerr << "Prior on [" << p.i << ", " << p.j << ")";
      //for (WordId w : p.target) { cerr << " " << vocab.Convert(w); }
      //cerr << " = ";
      tie(_, score) = AdvanceLM(lm, state.lm_state, p.target);
      //cerr << score << " + " << p.logprob << " + " << GetFutureCost(future_costs, state.coverage | c);
      score += p.logprob;
      score += GetFutureCost(future_costs, state.coverage | c);
      //cerr << " = " << score << endl;
      unexplored_children.push_back(p);
      unexplored_child_scores.push_back(score);
    }
  }
}

TreeNode* TreeNode::Expand(const MatchingPhraseList& mpl, LM& lm, const vector<vector<float>>& future_costs) {
  assert (unexplored_children.size() > 0);
  assert (unexplored_children.size() == unexplored_child_scores.size());
  //unsigned r = rng() % unexplored_children.size();
  unsigned r = sample_multinomial(unexplored_child_scores);

  MatchingPhrase phrase = unexplored_children[r];
  float prior = unexplored_child_scores[r];
  unexplored_children.erase(unexplored_children.begin() + r);
  unexplored_child_scores.erase(unexplored_child_scores.begin() + r);

  DecoderState child_state;
  child_state.coverage = state.coverage | MakeCoverage(phrase.i, phrase.j);
  TreeNode* child = new TreeNode(this, child_state, phrase, mpl, lm, future_costs);
  child->prior = prior;
  children.push_back(child);
  return child;
}

float TreeNode::UCB() const {
  return (N * prior + visits * value) / (visits + N) + C * sqrt(log(parent->visits) / visits);
  //return (N * prior + value) / (visits + N) + C * sqrt(log(parent->visits) / visits);
}

TreeNode* TreeNode::ChooseChild() {
  assert (children.size() > 0);

  float best_score = children[0]->UCB();
  unsigned best_index = 0;
  for (unsigned i = 1; i < children.size(); ++i) {
    float score = children[i]->UCB();
    if (score > best_score) {
      best_score = score;
      best_index = i;
    }
  }
  return children[best_index];
}

void TreeNode::Update(float final_score) {
  TreeNode* node = this;
  while (node != nullptr) {
    node->value = max(final_score, node->value);
    node->visits += 1;

    /*node->value += final_score;
    node->visits += 1;*/

    node = node->parent;
  }
}

float Rollout(TreeNode* node, const vector<WordId>& source, const MatchingPhraseList& mpl, const vector<vector<float>>& future_costs, TM& tm, LM& lm) {
  Coverage coverage = node->state.coverage;
  //cerr << "Starting rollout with coverage = " << coverage << endl;
  Coverage goal = MakeCoverage(0, source.size());
  vector<WordId> target;

  // Init target from node
  TreeNode* n = node;
  while (n->parent != nullptr) {
    target.insert(target.begin(), n->phrase.target.begin(), n->phrase.target.end());
    n = n->parent;
  }

  LMState lm_state = node->state.lm_state;
  float lm_lp;
  while (coverage != goal) {
    vector<MatchingPhrase> candidates;
    vector<float> scores;
    for (auto it : mpl) {
      if ((coverage & it.first) != 0) {
        continue;
      }

      float future_cost = GetFutureCost(future_costs, coverage | it.first);

      for (MatchingPhrase& p : it.second) {
        LMState new_lm_state;
        tie(new_lm_state, lm_lp) = AdvanceLM(lm, lm_state, p.target);
        if ((it.first | coverage) == goal) {
          lm_lp += lm.end(new_lm_state);
        }
        float lp = lm_lp + p.logprob + future_cost;
        candidates.push_back(p);
        scores.push_back(lp);
        /*cerr << lp << "\t" << p.logprob << "\t" << lm_lp << "\t";
        for (WordId w : p.target) { cerr << " " << vocab.Convert(w); }
        cerr << endl;*/
      }
    }

    assert (candidates.size() > 0);
    unsigned r = sample_multinomial(scores);
    MatchingPhrase& phrase = candidates[r];
    coverage |= MakeCoverage(phrase.i, phrase.j);
    target.insert(target.end(), phrase.target.begin(), phrase.target.end());
    tie(lm_state, lm_lp) = AdvanceLM(lm, lm_state, phrase.target);
    if (coverage == goal) {
      lm_lp += lm.end(lm_state);
    }
    //cerr << "Sampled phrase";
    //for (WordId w : phrase.target) { cerr << " " << vocab.Convert(w); }  
    //cerr << " to cover [" << phrase.i << ", " << phrase.j << "). Coverage is now " << coverage << endl;
  }

  float score = Score(source, target, tm, lm);
  /*cerr << "Scoring sentence pair:";
  for (WordId w : source) { cerr << " " << vocab.Convert(w); }  
  cerr << " |||";
  for (WordId w : target) { cerr << " " << vocab.Convert(w); }  
  cerr << " = ";
  cerr << score << endl;*/
  return score;
}

vector<WordId> Search(const vector<WordId>& source, TM& tm, LM& lm) {
  MatchingPhraseList mpl = FindMatchingPhrases(source, tm);
  vector<vector<float>> future_costs = ComputeFutureCosts(source, mpl, lm);
  vector<WordId> target;
  TreeNode* root = new TreeNode(nullptr, {0, lm.begin()}, {0, 0, vector<WordId>()}, mpl, lm, future_costs);

  unsigned step_number = 0;
  while (root->state.coverage != MakeCoverage(0, source.size())) {
    unsigned samples = (step_number == 0) ? 100000 : 100;
    for (unsigned i = 0; i < samples; ++i) {
      TreeNode* node = root;

      while (node->unexplored_children.size() == 0 && node->children.size() != 0) {
        node = node->ChooseChild();
      }

      if (node->unexplored_children.size() > 0) {
        node = node->Expand(mpl, lm, future_costs);
      }

      float result = Rollout(node, source, mpl, future_costs, tm, lm);
      node->Update(result);
    }

    //cout << "Final scores:" << endl;
    float best_score;
    TreeNode* best = nullptr;
    for (TreeNode* child : root->children) {
      if (best == nullptr || child->value > best_score) {
        best = child;
        best_score = child->value;
      }
      /*if (best == nullptr || child->visits > best_score) {
        best = child;
        best_score = child->visits;
      }*/
      MatchingPhrase phrase = child->phrase;
      /*cout << step_number << "\t" << child->visits << "\t" << child->prior << "\t" << child->value << "\t" << "[" << phrase.i << ", " << phrase.j << ")";
      for (WordId w : phrase.target) { cout << " " << vocab.Convert(w); }
      cout << endl;*/
    }

    assert (best != nullptr);
    target.insert(target.end(), best->phrase.target.begin(), best->phrase.target.end());
    /*cout << "Final choice:";
    cout << "[" << best->phrase.i << ", " << best->phrase.j << ")\t";
    for (WordId w : best->phrase.target) { cout << " " << vocab.Convert(w); }
    cout << endl;*/

    root = best;
    step_number++;
  }

  if (root != nullptr) {
    delete root;
    root = nullptr;
  }

  return target;
}

int main() {
  //Vocabulary vocab; // Temporarily made global, for debugging purposes
  cerr << "Loading TM..." << endl;
  TM tm(tm_filename, vocab);
  vocab.Freeze();
  cerr << "Loading LM..." << endl;
  LM lm(lm_filename, vocab);
  cerr << "Done loading." << endl;

  cout.setf(ios::fixed, ios::floatfield);
  cerr.setf(ios::fixed, ios::floatfield);

  for (string input_string; getline(cin, input_string);) {
    vector<WordId> source = vocab.Convert(tokenize(input_string, " "));
    for (unsigned i = 0; i < 1; ++i) {
      vector<WordId> target = Search(source, tm, lm);
      cout << Score(source, target, tm, lm) << " ||| ";
      for (unsigned i = 0; i < target.size(); ++i) {
        if (i != 0) { cout << " "; }
        cout << vocab.Convert(target[i]);
      }
      cout << endl;
    }
  }

  return 0;
}
