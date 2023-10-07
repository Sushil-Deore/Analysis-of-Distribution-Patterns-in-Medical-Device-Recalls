# Loading necessary libraries
library(arules)
library(tm)
library(SnowballC)
library(arulesViz)

# Reading openFDA dataset
openFDA_data <- read.csv("clean_openFDA.csv")

# Extracting the columns to use for association rule mining
selected_columns <- openFDA_data$cleaned_action

# Tokenization and stemming code
corpus <- Corpus(VectorSource(selected_columns))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

# Tokenizing the text
tokenized_text <- lapply(corpus, function(x) unlist(strsplit(as.character(x), " ")))

# Sampling 15% of tokenized_text
sample_size <- ceiling(0.15 * length(tokenized_text))
sampled_text <- sample(tokenized_text, size = sample_size)

# Creating transactions from the sampled text
openFDA_transactions <- as(sampled_text, "transactions")

# Performing Association Rule Mining (ARM) with lower support and higher confidence thresholds
association_rules <- apriori(openFDA_transactions, parameter = list(support = 0.2, confidence = 0.5, minlen = 2))

# Getting the top 15 rules for support, confidence, and lift
top_support <- head(sort(association_rules, by = "support", decreasing = TRUE), 15)
top_confidence <- head(sort(association_rules, by = "confidence", decreasing = TRUE), 15)
top_lift <- head(sort(association_rules, by = "lift", decreasing = TRUE), 15)

# Plotting item frequency
my_colors <- heat.colors(20)
par(mar = c(5, 4, 2, 2))  # Adjust the margin values as needed
options(repr.plot.width = 15, repr.plot.height = 15)  # Adjust plot width and height

# Create the item frequency plot with colors
itemFrequencyPlot(openFDA_transactions, topN = 20, type = "absolute", col = my_colors)

# Plotting the association rules for support
plot(top_support, method = "graph")
plot(top_support, method = "matrix")

# Plotting the association rules for confidence
plot(top_confidence, method = "graph")
plot(top_confidence, method = "matrix")

# Plotting the association rules for lift
plot(top_lift, method = "graph")
plot(top_lift, method = "matrix")

