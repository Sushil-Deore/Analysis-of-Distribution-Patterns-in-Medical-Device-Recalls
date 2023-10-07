# Loading required libraries
library(data.table)
library(cluster)
library(proxy)

# Reading data from the CSV file
data <- fread("clean_openFDA.csv")

# Selecting numeric columns
numeric_cols <- data[, c('time_duration', 'product_quantity_cleaned', 'cost_to_recall_in_dollers')]

# Sampling a subset of data (adjust the fraction to your desired subset size)
subset_size <- 0.001  # 1% of the data
set.seed(123)  # For reproducibility
sampled_data <- numeric_cols[sample(1:nrow(numeric_cols), nrow(numeric_cols) * subset_size), ]

# Scaling the data
scaled_data <- scale(sampled_data)

# Calculating cosine similarity
cosine_similarity <- proxy::simil(as.matrix(scaled_data), method = "cosine")

# Converting cosine similarity to distance
cosine_distance <- 1 - cosine_similarity

# Performing hierarchical clustering with different linkage methods
methods <- c("complete", "single", "average")
hclust_results <- lapply(methods, function(method) {
  hclust(as.dist(cosine_distance), method = method)
})

# Plotting the dendrograms for each linkage method with clusters highlighted
for (i in 1:length(methods)) {
  plot(hclust_results[[i]], main = paste("Hierarchical Clustering (", methods[i], ")"))
}

# Plotting the dendrograms for each linkage method with clusters highlighted
for (i in 1:length(methods)) {
  plot(hclust_results[[i]], main = paste("Hierarchical Clustering (", methods[i], ")"))
  
  # Add rectangles to highlight clusters
  rect.hclust(hclust_results[[i]], k = 4)  
}

