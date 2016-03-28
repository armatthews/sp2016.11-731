#include "utils.h"

Coverage MakeCoverage(unsigned i, unsigned j) {
  const static Coverage ONE = 1;
  return ((ONE << j) - 1) ^ ((ONE << i) - 1);
}

vector<string> tokenize(string input, string delimiter, unsigned max_times) {
  vector<string> tokens;
  //tokens.reserve(max_times);
  size_t last = 0;
  size_t next = 0;
  while ((next = input.find(delimiter, last)) != string::npos && tokens.size() < max_times) {
    tokens.push_back(input.substr(last, next-last));
    last = next + delimiter.length();
  }
  tokens.push_back(input.substr(last));
  return tokens;
}

vector<string> tokenize(string input, string delimiter) {
  return tokenize(input, delimiter, input.length());
}

vector<string> tokenize(string input, char delimiter) {
  return tokenize(input, string(1, delimiter));
}

string strip(const string& input) {
  string output = input;
  boost::algorithm::trim(output);
  return output;
}

vector<string> strip(const vector<string>& input, bool removeEmpty) {
  vector<string> output;
  for (unsigned i = 0; i < input.size(); ++i) {
    string s = strip(input[i]);
    if (s.length() > 0 || !removeEmpty) {
      output.push_back(s);
    }
  }
  return output;
}

vector<WordId> slice(const vector<WordId>& v, unsigned i, unsigned j) {
  assert (i < j);
  vector<WordId> r;
  for (unsigned k = i; k < j; ++k) {
    r.push_back(v[k]);
  }
  return r;
}
