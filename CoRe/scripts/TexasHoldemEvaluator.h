#ifndef TEXASHOLDEMEVALUATOR_H
#define TEXASHOLDEMEVALUATOR_H

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>

// Define the rank values for each card
extern std::unordered_map<char, int> rank_values;

// Define the hand rankings
extern std::vector<std::string> hand_rankings;

// Define the sequence of values
extern std::vector<char> values_sequence;

// Function to evaluate a hand in Texas Hold'em poker
std::pair<std::string, std::vector<char>> evaluate_hand(const std::vector<std::string>& cards);

// Function to compare the high cards of two hands
int compare_high_cards(const std::vector<char>& high_cards1, const std::vector<char>& high_cards2);

// Function to compare two hands in Texas Hold'em poker
int compare_hands(const std::vector<std::string>& hand1, const std::vector<std::string>& hand2, const std::vector<std::string>& community_cards);

#endif // TEXASHOLDEMEVALUATOR_H
