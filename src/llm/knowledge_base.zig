const std = @import("std");
const Allocator = std.mem.Allocator;

pub const KnowledgeBase = struct {
    allocator: Allocator,
    topics: std.StringHashMap(TopicInfo),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .topics = std.StringHashMap(TopicInfo).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        var iterator = self.topics.iterator();
        while (iterator.next()) |entry| {
            // Free the duplicated key
            self.allocator.free(entry.key_ptr.*);
            // Free the topic info
            entry.value_ptr.deinit(self.allocator);
        }
        self.topics.deinit();
    }

    pub fn loadKnowledge(self: *Self) !void {
        // Load comprehensive knowledge base
        try self.addAIKnowledge();
        try self.addProgrammingKnowledge();
        try self.addScienceKnowledge();
        try self.addTechnologyKnowledge();
        try self.addGeneralKnowledge();
    }

    pub fn getResponse(self: *Self, query: []const u8, max_length: u32) ![]u8 {
        const query_lower = try std.ascii.allocLowerString(self.allocator, query);
        defer self.allocator.free(query_lower);

        // Find the most relevant topic
        var best_match: ?TopicInfo = null;
        var best_score: f32 = 0.0;

        var iterator = self.topics.iterator();
        while (iterator.next()) |entry| {
            const score = self.calculateRelevanceScore(query_lower, entry.key_ptr.*);
            if (score > best_score) {
                best_score = score;
                best_match = entry.value_ptr.*;
            }
        }

        if (best_match) |topic| {
            return self.generateResponseFromTopic(topic, query, max_length);
        }

        // Fallback to general response
        return self.generateGeneralResponse(query, max_length);
    }

    fn calculateRelevanceScore(self: *Self, query: []const u8, topic_key: []const u8) f32 {
        _ = self;
        var score: f32 = 0.0;

        // Simple keyword matching
        if (std.mem.indexOf(u8, query, topic_key) != null) {
            score += 1.0;
        }

        // Check for related terms
        const topic_words = std.mem.split(u8, topic_key, " ");
        var word_iter = topic_words;
        while (word_iter.next()) |word| {
            if (std.mem.indexOf(u8, query, word) != null) {
                score += 0.5;
            }
        }

        return score;
    }

    fn generateResponseFromTopic(self: *Self, topic: TopicInfo, query: []const u8, max_length: u32) ![]u8 {
        _ = query;
        var response = std.ArrayList(u8).init(self.allocator);

        try response.appendSlice(topic.definition);

        if (max_length > 150 and topic.details.len > 0) {
            try response.appendSlice(" ");
            try response.appendSlice(topic.details);
        }

        if (max_length > 300 and topic.examples.len > 0) {
            try response.appendSlice(" ");
            try response.appendSlice(topic.examples);
        }

        if (max_length > 400 and topic.applications.len > 0) {
            try response.appendSlice(" ");
            try response.appendSlice(topic.applications);
        }

        return response.toOwnedSlice();
    }

    fn generateGeneralResponse(self: *Self, query: []const u8, max_length: u32) ![]u8 {
        _ = query;
        var response = std.ArrayList(u8).init(self.allocator);

        try response.appendSlice("That's an interesting question. While I don't have specific information about this exact topic, I can provide some general insights. ");

        if (max_length > 150) {
            try response.appendSlice("This subject likely involves complex concepts that benefit from systematic study and practical application. ");
        }

        if (max_length > 250) {
            try response.appendSlice("For more detailed information, I'd recommend consulting specialized resources or experts in the relevant field. ");
        }

        return response.toOwnedSlice();
    }

    fn addAIKnowledge(self: *Self) !void {
        try self.addTopic("artificial intelligence", TopicInfo{
            .definition = "Artificial Intelligence (AI) is a field of computer science focused on creating intelligent machines that can perform tasks typically requiring human intelligence.",
            .details = "AI encompasses machine learning, natural language processing, computer vision, robotics, and expert systems. It uses algorithms to process data, learn patterns, and make decisions.",
            .examples = "Examples include virtual assistants like Siri and Alexa, recommendation systems on Netflix and Amazon, autonomous vehicles, medical diagnosis systems, and game-playing AI like AlphaGo.",
            .applications = "AI applications span healthcare (drug discovery, medical imaging), finance (fraud detection, algorithmic trading), transportation (autonomous vehicles), and entertainment (content recommendation, game AI).",
        });

        try self.addTopic("machine learning", TopicInfo{
            .definition = "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task.",
            .details = "It includes supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error).",
            .examples = "Common algorithms include neural networks, decision trees, support vector machines, and ensemble methods like random forests.",
            .applications = "Used in image recognition, natural language processing, recommendation systems, predictive analytics, and autonomous systems.",
        });

        try self.addTopic("neural networks", TopicInfo{
            .definition = "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information.",
            .details = "They consist of layers of neurons with weighted connections. Deep neural networks with many layers enable deep learning, which has revolutionized AI.",
            .examples = "Types include feedforward networks, convolutional neural networks (CNNs) for image processing, and recurrent neural networks (RNNs) for sequential data.",
            .applications = "Used in computer vision, speech recognition, natural language processing, and game playing.",
        });
    }

    fn addProgrammingKnowledge(self: *Self) !void {
        try self.addTopic("programming", TopicInfo{
            .definition = "Programming is the process of creating instructions for computers to execute, using programming languages to solve problems and build applications.",
            .details = "It involves problem-solving, algorithm design, data structure selection, and code implementation. Good programming requires understanding of logic, mathematics, and system design.",
            .examples = "Popular languages include Python (data science, web development), JavaScript (web development), Java (enterprise applications), C++ (system programming), and Rust (systems programming).",
            .applications = "Used to create websites, mobile apps, desktop software, games, operating systems, and embedded systems.",
        });

        try self.addTopic("algorithms", TopicInfo{
            .definition = "Algorithms are step-by-step procedures or formulas for solving problems, fundamental to computer science and programming.",
            .details = "They have properties like correctness, efficiency (time and space complexity), and clarity. Algorithm analysis helps choose the best approach for specific problems.",
            .examples = "Common algorithms include sorting (quicksort, mergesort), searching (binary search), graph algorithms (Dijkstra's algorithm), and dynamic programming solutions.",
            .applications = "Used in database operations, web search, route planning, data compression, and optimization problems.",
        });

        try self.addTopic("data structures", TopicInfo{
            .definition = "Data structures are ways of organizing and storing data in computers to enable efficient access and modification.",
            .details = "Choice of data structure affects program performance. They provide different trade-offs between memory usage, access time, and modification complexity.",
            .examples = "Common structures include arrays, linked lists, stacks, queues, trees, hash tables, and graphs.",
            .applications = "Used in database design, compiler construction, operating systems, and algorithm implementation.",
        });
    }

    fn addScienceKnowledge(self: *Self) !void {
        try self.addTopic("quantum computing", TopicInfo{
            .definition = "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers.",
            .details = "Quantum bits (qubits) can exist in multiple states simultaneously, potentially enabling exponential speedups for certain problems. However, quantum systems are fragile and require extreme conditions.",
            .examples = "Quantum algorithms include Shor's algorithm for factoring large numbers and Grover's algorithm for searching unsorted databases.",
            .applications = "Potential applications include cryptography, drug discovery, financial modeling, optimization problems, and machine learning.",
        });

        try self.addTopic("physics", TopicInfo{
            .definition = "Physics is the fundamental science that studies matter, energy, and their interactions in the universe.",
            .details = "It encompasses classical mechanics, thermodynamics, electromagnetism, quantum mechanics, and relativity. Physics provides the foundation for understanding natural phenomena.",
            .examples = "Key concepts include Newton's laws of motion, Einstein's theory of relativity, quantum mechanics, and the laws of thermodynamics.",
            .applications = "Physics principles enable technologies like lasers, computers, medical imaging, GPS, and renewable energy systems.",
        });
    }

    fn addTechnologyKnowledge(self: *Self) !void {
        try self.addTopic("blockchain", TopicInfo{
            .definition = "Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) linked and secured using cryptography.",
            .details = "Each block contains a cryptographic hash of the previous block, timestamp, and transaction data. This creates an immutable record that's resistant to modification.",
            .examples = "Bitcoin and Ethereum are well-known blockchain implementations. Other applications include supply chain tracking and digital identity verification.",
            .applications = "Used in cryptocurrencies, smart contracts, supply chain management, voting systems, and digital asset management.",
        });

        try self.addTopic("cloud computing", TopicInfo{
            .definition = "Cloud computing delivers computing services including servers, storage, databases, networking, software, and analytics over the internet.",
            .details = "It offers on-demand access to computing resources with pay-as-you-use pricing. Service models include Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS).",
            .examples = "Major providers include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform. Services range from virtual machines to AI/ML platforms.",
            .applications = "Enables scalable web applications, data analytics, machine learning, backup and disaster recovery, and remote work solutions.",
        });
    }

    fn addGeneralKnowledge(self: *Self) !void {
        try self.addTopic("mathematics", TopicInfo{
            .definition = "Mathematics is the study of numbers, shapes, patterns, and logical reasoning, providing tools for understanding and describing the world.",
            .details = "It includes areas like algebra, geometry, calculus, statistics, and discrete mathematics. Mathematics provides the foundation for science, engineering, and technology.",
            .examples = "Key concepts include functions, derivatives, integrals, probability, linear algebra, and graph theory.",
            .applications = "Used in physics, engineering, computer science, economics, cryptography, and data analysis.",
        });
    }

    fn addTopic(self: *Self, key: []const u8, topic: TopicInfo) !void {
        const owned_key = try self.allocator.dupe(u8, key);
        const owned_topic = try topic.clone(self.allocator);
        try self.topics.put(owned_key, owned_topic);
    }
};

const TopicInfo = struct {
    definition: []const u8,
    details: []const u8,
    examples: []const u8,
    applications: []const u8,

    fn clone(self: TopicInfo, allocator: Allocator) !TopicInfo {
        return TopicInfo{
            .definition = try allocator.dupe(u8, self.definition),
            .details = try allocator.dupe(u8, self.details),
            .examples = try allocator.dupe(u8, self.examples),
            .applications = try allocator.dupe(u8, self.applications),
        };
    }

    fn deinit(self: *TopicInfo, allocator: Allocator) void {
        allocator.free(self.definition);
        allocator.free(self.details);
        allocator.free(self.examples);
        allocator.free(self.applications);
    }
};
