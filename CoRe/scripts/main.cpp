#include <iostream>
#include<algorithm>
#include <vector>
#include <string>
#include "TexasHoldemEvaluator.h"
using namespace std;

int main() {
    // Example hands and community cards
    std::vector<std::string> hand1 = {"Ac", "Tc"};
    std::vector<std::string> hand2 = {"Tc", "7c"};
    std::vector<std::string> community_cards = {"Ac", "Te", "Tc", "7c", "7e"};

    // Compare the hands
    int winner = compare_hands(hand1, hand2, community_cards);

    // Print the result
    if (winner == 1) {
        std::cout << "Player 1 wins!" << std::endl;
    } else if (winner == 2) {
        std::cout << "Player 2 wins!" << std::endl;
    } else {
        std::cout << "It's a tie!" << std::endl;
    }

    // Test the evaluate_hand function
    std::vector<std::string> test_hand = {"Td", "7d", "Ac", "Td", "Tc", "7d", "7c"};
    auto hand_ranking = evaluate_hand(test_hand);
    std::cout << "Hand ranking: " << hand_ranking.first << std::endl;
    std::cout << "High cards: ";
    for (char card : hand_ranking.second) {
        std::cout << card << " ";
    }
    std::cout << std::endl;

    return 0;
}
