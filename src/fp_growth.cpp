// FP-Growth for transactional frequent itemset mining.
// C++17, standard library only (Linux/macOS; optional Windows peak RSS fallback).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef USE_CUDA
# include "cuda_histogram.hpp"
#endif
#if defined(__unix__) || defined(__APPLE__)
# include <sys/resource.h>
#endif

using Item = int;
using Count = std::uint64_t;
using Transaction = std::vector<Item>;
struct WeightedTransaction { Transaction items; Count count; };

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char ch : s) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: if (ch < 0x20) { out << "\\u" << std::hex << std::setw(4)
                                               << std::setfill('0') << (int)ch << std::dec; }
                     else out << ch;
        }
    }
    return out.str();
}

class Dataset {
public:
    std::vector<Transaction> transactions;
    std::vector<std::string> names;
    std::unordered_map<std::string, Item> index;

    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Cannot open input file: " + path);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            std::string word;
            Transaction row;
            while (ss >> word) {
                if (!word.empty() && word[0] == '#') break; // comments at end of line
                auto it = index.find(word);
                if (it == index.end()) {
                    const Item id = (Item)names.size();
                    names.push_back(word);
                    index.emplace(word, id);
                    row.push_back(id);
                } else row.push_back(it->second);
            }
            if (row.empty()) continue; // ignore empty/comment-only lines
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            transactions.push_back(std::move(row));
        }
        if (transactions.empty()) throw std::runtime_error("No transactions in input file");
    }
};

struct Node {
    Item item;
    Count count;
    Node* parent;
    std::unordered_map<Item, Node*> children;
    Node(Item i, Count c, Node* p): item(i), count(c), parent(p) {}
};

// The support-counting stage is deliberately shared by the root tree AND
// every recursive conditional tree. In the CUDA build, weighted support
// histograms are evaluated by GPU kernels instead of the CPU loop.
class SupportCounter {
    std::size_t unique_items;
#ifdef USE_CUDA
    CudaHistogram cuda;
#endif
public:
    std::uint64_t root_calls = 0;
    std::uint64_t conditional_calls = 0;
    explicit SupportCounter(std::size_t count): unique_items(count) {}

    std::unordered_map<Item, Count> count(const std::vector<WeightedTransaction>& rows,
                                         bool root) {
        std::unordered_map<Item, Count> result;
#ifdef USE_CUDA
        std::size_t entries = 0;
        for (const auto& row : rows) entries += row.items.size();
        std::vector<int> ids;
        std::vector<std::uint64_t> weights;
        ids.reserve(entries);
        weights.reserve(entries);
        for (const auto& row : rows) for (Item id : row.items) {
            ids.push_back(id);
            weights.push_back(row.count);
        }
        const auto histogram = cuda.count(ids, weights, unique_items);
        for (std::size_t id = 0; id < histogram.size(); ++id)
            if (histogram[id]) result.emplace(static_cast<Item>(id), histogram[id]);
#else
        for (const auto& row : rows)
            for (Item id : row.items) result[id] += row.count;
#endif
        if (root) ++root_calls;
        else ++conditional_calls;
        return result;
    }
#ifdef USE_CUDA
    CudaHistogramStats gpu_stats() const { return cuda.stats(); }
#endif
};

class FPTree {
    std::vector<std::unique_ptr<Node>> owned;
    Node* new_node(Item item, Count count, Node* parent) {
        owned.emplace_back(std::make_unique<Node>(item, count, parent));
        return owned.back().get();
    }
public:
    Node* root;
    std::unordered_map<Item, Count> support;
    std::unordered_map<Item, std::vector<Node*>> header;
    std::vector<Item> desc_items; // descending support, item ID breaks ties

    explicit FPTree(const std::vector<WeightedTransaction>& rows, Count min_support,
                    SupportCounter& counter, bool root_tree=false) {
        root = new_node(-1, 0, nullptr);
        support = counter.count(rows, root_tree);
        for (auto it = support.begin(); it != support.end();) {
            if (it->second < min_support) it = support.erase(it);
            else ++it;
        }
        desc_items.reserve(support.size());
        for (const auto& e : support) desc_items.push_back(e.first);
        std::sort(desc_items.begin(), desc_items.end(), [&](Item a, Item b) {
            if (support.at(a) != support.at(b)) return support.at(a) > support.at(b);
            return a < b;
        });
        std::unordered_map<Item, size_t> rank;
        for (size_t i = 0; i < desc_items.size(); ++i) rank[desc_items[i]] = i;
        for (const auto& row : rows) {
            Transaction ordered;
            ordered.reserve(row.items.size());
            for (Item id : row.items) if (support.count(id)) ordered.push_back(id);
            std::sort(ordered.begin(), ordered.end(), [&](Item a, Item b) {
                return rank.at(a) < rank.at(b);
            });
            Node* ptr = root;
            for (Item id : ordered) {
                auto it = ptr->children.find(id);
                if (it == ptr->children.end()) {
                    Node* child = new_node(id, row.count, ptr);
                    ptr->children[id] = child;
                    header[id].push_back(child);
                    ptr = child;
                } else {
                    it->second->count += row.count;
                    ptr = it->second;
                }
            }
        }
    }
    size_t node_count() const { return owned.size(); }
    bool empty() const { return support.empty(); }
};

class Miner {
    Count minsup;
    const std::vector<std::string>& names;
    std::ofstream csv;
    std::size_t top_limit;
public:
    Count total_patterns = 0;
    size_t largest_pattern = 0;
    size_t root_nodes = 0;
    size_t trees_built = 0;
    size_t tree_nodes_total = 0; // nodes allocated cumulatively (not concurrent peak)
    size_t max_depth = 0;
    SupportCounter* counter = nullptr;
    std::vector<std::pair<Count, std::string>> top;
    Miner(Count minimum, const std::vector<std::string>& labels,
          const std::string& output, size_t top_count):
          minsup(minimum), names(labels), top_limit(top_count) {
        if (output != "-") {
            csv.open(output);
            if (!csv) throw std::runtime_error("Cannot open output: " + output);
            csv << "support,itemset\n";
        }
    }
    void emit(const Transaction& pattern, Count count) {
        ++total_patterns;
        largest_pattern = std::max(largest_pattern, pattern.size());
        std::vector<std::string> words;
        for (Item id : pattern) words.push_back(names.at(id));
        std::sort(words.begin(), words.end());
        std::ostringstream joined;
        for (size_t i = 0; i < words.size(); ++i) {
            if (i) joined << ';';
            joined << words[i];
        }
        const auto key = joined.str();
        if (csv) csv << count << ",\"" << key << "\"\n"; // tokens should not contain quote
        if (top_limit > 0) {
            top.emplace_back(count, key);
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
            if (top.size() > top_limit) top.pop_back();
        }
    }
    void recursive_mine(const FPTree& tree, const Transaction& suffix, size_t depth) {
        max_depth = std::max(max_depth, depth);
        // Mine items from least frequent to most frequent, avoiding duplicates.
        for (auto it = tree.desc_items.rbegin(); it != tree.desc_items.rend(); ++it) {
            const Item id = *it;
            const Count count = tree.support.at(id);
            Transaction pattern = suffix;
            pattern.push_back(id);
            emit(pattern, count);
            std::vector<WeightedTransaction> conditional_paths;
            for (const Node* node : tree.header.at(id)) {
                Transaction path;
                for (const Node* anc = node->parent; anc && anc->item != -1; anc = anc->parent)
                    path.push_back(anc->item);
                if (!path.empty()) conditional_paths.push_back({std::move(path), node->count});
            }
            if (!conditional_paths.empty()) {
                FPTree conditional_tree(conditional_paths, minsup, *counter);
                ++trees_built;
                tree_nodes_total += conditional_tree.node_count();
                if (!conditional_tree.empty()) recursive_mine(conditional_tree, pattern, depth + 1);
            }
        }
    }
    void run(const Dataset& dataset, SupportCounter& support_counter) {
        counter = &support_counter;
        std::vector<WeightedTransaction> rows;
        rows.reserve(dataset.transactions.size());
        for (const auto& transaction : dataset.transactions) rows.push_back({transaction, 1});
        FPTree tree(rows, minsup, support_counter, true);
        root_nodes = tree.node_count();
        ++trees_built;
        tree_nodes_total += tree.node_count();
        recursive_mine(tree, {}, 0);
    }
};

Count parse_support(const std::string& raw, size_t n) {
    try {
        if (!raw.empty() && raw.back() == '%') {
            const double pct = std::stod(raw.substr(0, raw.size()-1));
            if (!(pct > 0 && pct <= 100)) throw std::invalid_argument("invalid percentage");
            return std::max<Count>(1, (Count)std::ceil(pct * n / 100.0 - 1e-12));
        }
        if (raw.find('.') != std::string::npos) {
            const double ratio = std::stod(raw);
            if (!(ratio > 0 && ratio <= 1)) throw std::invalid_argument("invalid ratio");
            return std::max<Count>(1, (Count)std::ceil(ratio * n - 1e-12));
        }
        const long long threshold = std::stoll(raw);
        if (threshold <= 0) throw std::invalid_argument("minimum must be positive");
        return (Count)threshold;
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid minimum support '" + raw +
                                    "' (use 0.25, 25%, or integer 100)");
    }
}

long peak_rss_kb() {
#if defined(__linux__)
    // /proc/self/status VmHWM resets on exec; ru_maxrss may inherit the
    // Python benchmark driver's pre-exec high-water mark after fork.
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream parse(line.substr(6));
            long kib = -1;
            if (parse >> kib) return kib;
        }
    }
#endif
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
# if defined(__APPLE__)
        return usage.ru_maxrss / 1024; // macOS returns bytes
# else
        return usage.ru_maxrss;        // Linux returns KiB
# endif
    }
#endif
    return -1;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: fp_growth DATA.txt SUPPORT [--output FILE.csv | --no-output]"
                         " [--summary FILE.json] [--top N]\n"
                         "SUPPORT: 0.2 (=20%); 20%; 100 (=100 transactions).\n"
                         "DATA: one transaction per line, whitespace-separated tokens.\n";
            return 2;
        }
        const std::string input = argv[1];
        const std::string raw_min = argv[2];
        std::string output = "patterns.csv";
        std::string summary = "summary.json";
        size_t top_count = 10;
        for (int i = 3; i < argc; ++i) {
            const std::string opt = argv[i];
            if ((opt == "--output" || opt == "--summary" || opt == "--top") && i + 1 >= argc)
                throw std::invalid_argument("Option needs an argument: " + opt);
            if (opt == "--output") output = argv[++i];
            else if (opt == "--summary") summary = argv[++i];
            else if (opt == "--top") top_count = std::stoull(argv[++i]);
            else if (opt == "--no-output") output = "-";
            else throw std::invalid_argument("Unknown option: " + opt);
        }
        const auto start = std::chrono::steady_clock::now();
        Dataset dataset;
        dataset.load(input);
        const Count minimum = parse_support(raw_min, dataset.transactions.size());
        Miner miner(minimum, dataset.names, output, top_count);
        SupportCounter support_counter(dataset.names.size());
        miner.run(dataset, support_counter);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const long memory_kb = peak_rss_kb();
        std::ostringstream body;
        body << std::fixed << std::setprecision(6);
        body << "{\n  \"backend\": \""
#ifdef USE_CUDA
             << "cuda-hybrid-all-level-counting"
#else
             << "cpu"
#endif
             << "\",\n  \"input\": \"" << json_escape(input) << "\",\n"
             << "  \"transactions\": " << dataset.transactions.size() << ",\n"
             << "  \"unique_items\": " << dataset.names.size() << ",\n"
             << "  \"minimum_support_count\": " << minimum << ",\n"
             << "  \"minimum_support_percent\": " << 100.0 * minimum / dataset.transactions.size() << ",\n"
             << "  \"frequent_itemsets\": " << miner.total_patterns << ",\n"
             << "  \"max_pattern_length\": " << miner.largest_pattern << ",\n"
             << "  \"root_tree_nodes\": " << miner.root_nodes << ",\n"
             << "  \"conditional_trees_built\": " << miner.trees_built - 1 << ",\n"
             << "  \"all_tree_nodes_allocated\": " << miner.tree_nodes_total << ",\n"
             << "  \"runtime_seconds\": " << seconds << ",\n"
             << "  \"peak_rss_kib\": " << memory_kb << ",\n"
             << "  \"support_root_calls\": " << support_counter.root_calls << ",\n"
             << "  \"support_conditional_calls\": " << support_counter.conditional_calls << ",\n";
#ifdef USE_CUDA
        const auto gpu = support_counter.gpu_stats();
        body << "  \"cuda_device\": \"" << json_escape(gpu.device_name) << "\",\n"
             << "  \"cuda_kernel_calls\": " << gpu.kernel_calls << ",\n"
             << "  \"cuda_entries_processed\": " << gpu.entries_processed << ",\n"
             << "  \"cuda_buffer_high_water_bytes\": " << gpu.max_reserved_bytes << ",\n";
#endif
        body << "  \"top_itemsets\": [\n";
        for (size_t i=0; i<miner.top.size(); ++i) {
            body << "    {\"support\": " << miner.top[i].first << ", \"items\": \""
                 << json_escape(miner.top[i].second) << "\"}"
                 << (i + 1 != miner.top.size() ? "," : "") << "\n";
        }
        body << "  ]\n}\n";
        if (summary != "-") {
            std::ofstream out(summary);
            if (!out) throw std::runtime_error("Cannot create summary: " + summary);
            out << body.str();
        }
        std::cout << body.str();
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "Error: " << err.what() << '\n';
        return 1;
    }
}
