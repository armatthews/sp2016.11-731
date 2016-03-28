#include "vocab.h"

Vocabulary::Vocabulary() : frozen(false) {
  Convert("UNK");
  Convert("<s>");
  Convert("</s>");
}

WordId Vocabulary::Convert(const string& w) {
  if (w2i.find(w) == w2i.end()) {
    if (frozen) {
      return Convert("UNK");
    }
    w2i[w] = i2w.size();
    i2w.push_back(w);
  }

  return w2i[w];
}

string Vocabulary::Convert(WordId i) const {
  assert (i != 0);
  assert (i < i2w.size());
  return i2w[i];
}

vector<WordId> Vocabulary::Convert(const vector<string>& ws) {
  vector<WordId> is;
  for (const string& w : ws) {
    is.push_back(Convert(w));
  }
  return is;
}

vector<string> Vocabulary::Convert(const vector<WordId>& is) const {
  vector<string> ws;
  for (WordId i : is) {
    ws.push_back(Convert(i));
  }
  return ws;
}

bool Vocabulary::Contains(const string& w) const {
  return frozen;
}

void Vocabulary::Freeze() {
  frozen = true;
}
