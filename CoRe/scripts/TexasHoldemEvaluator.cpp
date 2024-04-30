#include "TexasHoldemEvaluator.h"
using namespace std;


// Define the rank values for each card
std::unordered_map<char, int> rank_values = {
    {'2', 2}, {'3', 3}, {'4', 4}, {'5', 5}, {'6', 6}, {'7', 7}, {'8', 8}, {'9', 9},
    {'T', 10}, {'J', 11}, {'Q', 12}, {'K', 13}, {'A', 14}
};

// Define the hand rankings
std::vector<std::string> hand_rankings = {
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"
};

// Define the sequence of values
std::vector<char> values_sequence = {
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'
};

std::pair<std::string, std::vector<char>> evaluate_hand(const std::vector<std::string>& cards) {
    std::vector<char> ranks(13, 0);  // Count of each rank
    std::vector<char> suits(4, 0);   // Count of each suit
    bool flush = false, straight = false;
    int straight_high = 0;

    // Count the ranks and suits
    for (const auto& card : cards) {
        ranks[rank_values[card[0]] - 2]++;
        suits[static_cast<int>(card[1]) - static_cast<int>('c')]++;
    }

    // Check for flush
    flush = std::any_of(suits.begin(), suits.end(), [](int count) { return count >= 5; });

    // Check for straight
    int straight_start = 0;
    while (straight_start < 9 && ranks[straight_start] == 0) straight_start++;
    straight = std::all_of(ranks.begin() + straight_start, ranks.begin() + straight_start + 5, [](int count) { return count == 1; });
    if (straight) {
        straight_high = straight_start + 4;
    } else if (ranks[0] == 1 && ranks[1] == 1 && ranks[2] == 1 && ranks[3] == 1 && ranks[12] == 1) {
        straight = true;
        straight_high = 3;
    }

    // Determine the hand ranking
    std::vector<char> high_cards;
    if (flush && straight && straight_high == 12) {
        // Royal Flush
        high_cards = {'A', 'K', 'Q', 'J', 'T'};

        return {"Royal Flush", high_cards};
    }else if (flush && straight) {
        // Straight Flush
        high_cards.push_back(values_sequence[straight_high]);

        return {"Straight Flush", high_cards};
    } else if (std::any_of(ranks.begin(), ranks.end(), [](int count) { return count == 4; })) {
        // Four of a Kind
        for (int i = ranks.size() - 1; i >= 0; i--) {
            if (ranks[i] == 4) {
                high_cards.push_back(values_sequence[i]);
                break;
            }
        }

        return {"Four of a Kind", high_cards};
    } else if ((std::count_if(ranks.begin(), ranks.end(), [](int count) { return count == 3; }) == 1 &&
               std::count_if(ranks.begin(), ranks.end(), [](int count) { return count == 2; }) >= 1) || 
               (std::count_if(ranks.begin(), ranks.end(), [](int count) { return count == 3; }) == 2)) {
        // Full House
        for (int i = ranks.size() - 1; i >= 0; i--) {
            if (ranks[i] == 3) {
                high_cards.push_back(values_sequence[i]);
                break;
            }
        }

        return {"Full House", high_cards};
    } else if (flush) {
        // Flush
        for (int i = ranks.size() - 1; i >= 0; i--) {
            if (ranks[i] > 0) {
                high_cards.push_back(values_sequence[i]);
            }
        }

        return {"Flush", high_cards};
    } else if (straight) {
        // Straight
        for (int i = straight_high; i > straight_high - 5; i--) {
            high_cards.push_back(values_sequence[i]);
        }

        return {"Straight", high_cards};
    } else if (std::count_if(ranks.begin(), ranks.end(), [](int count) { return count == 3; }) >= 1) {
        // Three of a Kind
        for (int i = ranks.size() - 1; i >= 0; i--) {
            if (ranks[i] == 3) {
                high_cards.push_back(values_sequence[i]);
            }
        }

        return {"Three of a Kind", high_cards};
    } else if (std::count_if(ranks.begin(), ranks.end(), [](int count) { return count == 2; }) >= 2) {
        // Two Pair
        int pair_count = 0;
        for (int i = ranks.size() - 1; i >= 0; i--) {
            if (ranks[i] == 2) {
                high_cards.push_back(values_sequence[i]);
                pair_count++;
            }
        }

        return {"Two Pair", high_cards};
    } else if (std::count_if(ranks.begin(), ranks.end(), [](int count) { return count == 2; }) == 1) {
        // Pair
        for (int i = ranks.size() - 1; i >= 0; i--) {
            if (ranks[i] == 2) {
                high_cards.push_back(values_sequence[i]);
            }
        }

        return {"Pair", high_cards};
    } else {
        // High Card
        for (int i = ranks.size() - 1; i >= 0; i--) {
            if (ranks[i] > 0) {
                high_cards.push_back(values_sequence[i]);
            }
        }

        return {"High Card", high_cards};
    }
}

int compare_high_cards(const std::vector<char>& high_cards1, const std::vector<char>& high_cards2) {
    // Compare the high cards one by one
    int compare_flag = 0;
    int cards2Win_count = 0;
    for (auto it1 = high_cards1.begin();it1 != high_cards1.end(); ++it1) {
        int cards1Win_count = 0;
        for(auto it2 = high_cards2.begin();it2 != high_cards2.end(); ++it2){
            if (rank_values[*it1] > rank_values[*it2]) {
                compare_flag = 1;
                cards1Win_count++;
            }
        }
        for(auto it2 = high_cards2.begin();it2 != high_cards2.end(); ++it2){
            if (rank_values[*it1] < rank_values[*it2]) {
                compare_flag = 2;
                cards2Win_count++;
                break;
            } 
        }
        if (cards1Win_count == int(high_cards2.size())){
            return compare_flag;
        }
        if (cards2Win_count == int(high_cards1.size())){
            return compare_flag;
        }
        compare_flag = 0;
    }
    // If all high cards are the same, it's a tie
    return compare_flag;
}

int compare_hands(const std::vector<std::string>& hand1, const std::vector<std::string>& hand2, const std::vector<std::string>& community_cards) {
    // Combine each player's hand with the community cards
    std::vector<std::string> player1_cards(hand1);
    player1_cards.insert(player1_cards.end(), community_cards.begin(), community_cards.end());
    std::vector<std::string> player2_cards(hand2);
    player2_cards.insert(player2_cards.end(), community_cards.begin(), community_cards.end());

    // Evaluate the hand rankings
    auto player1_ranking = evaluate_hand(player1_cards);
    auto player2_ranking = evaluate_hand(player2_cards);

    // Compare the hand rankings
    auto player1_index = std::find(hand_rankings.begin(), hand_rankings.end(), player1_ranking.first);
    auto player2_index = std::find(hand_rankings.begin(), hand_rankings.end(), player2_ranking.first);
    if (player1_index != hand_rankings.end() && player2_index != hand_rankings.end()) {
        if (player1_index - hand_rankings.begin() > player2_index - hand_rankings.begin()) {
            return 1;
        } else if (player1_index - hand_rankings.begin() < player2_index - hand_rankings.begin()) {
            return 2;
        } else {
            // If the hand rankings are the same, compare the high cards
            return compare_high_cards(player1_ranking.second, player2_ranking.second);
        }
    }
    // If the hand rankings are not found, return a tie
    return 0;
}
