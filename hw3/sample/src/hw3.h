#include <vector>
#include "lm.h"
#include "tm.h"

using namespace std;

struct DecoderState {
  Coverage coverage;
  LMState lm_state;
};

struct MatchingPhrase {
  unsigned i, j;
  vector<WordId> target;
  float logprob;
};
typedef unordered_map<Coverage, vector<MatchingPhrase>> MatchingPhraseList;

struct TreeNode {
  DecoderState state;
  MatchingPhrase phrase;

  float value;
  float prior;
  unsigned visits;

  TreeNode* parent;
  vector<TreeNode*> children;
  vector<MatchingPhrase> unexplored_children;
  vector<float> unexplored_child_scores;

  TreeNode(TreeNode* parent, const DecoderState& state, const MatchingPhrase& phrase, const MatchingPhraseList& mpl, LM& lm, const vector<vector<float>>& future_costs);
  ~TreeNode();
  void InitializeChildren(const MatchingPhraseList& mpl, LM& lm, const vector<vector<float>>& future_costs);
  TreeNode* Expand(const MatchingPhraseList& mpl, LM& lm, const vector<vector<float>>& future_costs);
  TreeNode* ChooseChild();
  void Update(float final_score);

  constexpr static float N = 1;
  constexpr static float C = 10 * sqrt(2);
  float UCB() const;
};

tuple<LMState, float> AdvanceLM(LM& lm, const LMState& in_state, const vector<WordId> phrase);
float Score(const vector<WordId>& source, const vector<WordId>& target, TM& tm, LM& lm, bool end = true);
float Score(const vector<WordId>& source, const vector<WordId>& target, MatchingPhraseList& mpl, LM& lm);
MatchingPhraseList FindMatchingPhrases(const vector<WordId>& source, TM& tm);
vector<vector<float>> ComputeFutureCosts(const vector<WordId>& source, MatchingPhraseList& mpl, LM& lm);
float GetFutureCost(const vector<vector<float>>& future_costs, Coverage coverage);
