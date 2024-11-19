
# Load necessary libraries
library(tidyverse)
library(tm)
library(SnowballC)
library(caret)
library(randomForest)
library(wordcloud)
library(RColorBrewer)

# Customer sample review dataset with equal number of rows for review_text and sentiment
reviews <- data.frame(
  review_text = c(
    "I love the product! Great quality.","Worst purchase ever, very disappointed. ","The service was okay, but the product is good.",
    "Absolutely fantastic! I will buy again.","Very bad experience, the quality is poor.","This phone case is amazing. It fits perfectly and looks great!",
    "Horrible customer service, I will never order again.", "The product exceeded my expectations, very satisfied.","I returned the product, it wasn't what I expected.",
    "The shoes are very comfortable, highly recommend them.", "The delivery was fast, and the product was well packaged.",
    "I’m very happy with the watch I bought, it’s stylish and functional.", "I had to wait for weeks for my order, and it arrived damaged.",
    "I bought the wrong size, but the customer service helped me exchange it.","Great quality for the price! I would buy again.",
    "The product was okay, but the shipping time was too long.","This was the worst online shopping experience ever.",
    "Love the variety of colors available, it makes shopping fun!", "Not happy with the product. It broke after one week.",
    "Customer service is very friendly and helpful.", "I’m very impressed with how well this item works.","The packaging was poor, and the item arrived damaged.",
    "I had no issues with my purchase. The item works great.","I would recommend this product to anyone looking for quality.",
    "Terrible product, the quality doesn’t match the description.","The product is as described, but it took a bit longer to arrive.",
    "The item arrived early and was exactly what I expected.","I ordered this as a gift and the recipient loved it.",
    "Really good customer service and fast delivery.","The clothing fits perfectly and is of great material.",
    "I had a lot of issues with this product. Would not buy again.","Fantastic quality, looks just like the pictures!",
    "The product stopped working after a few days.","Great packaging and quick delivery!","The item didn’t meet my expectations. ",
    "Very satisfied with the product and service.","The color wasn’t as shown in the picture, disappointing.",
    "Amazing product, worth the price.","Extremely helpful customer service, resolved my issue quickly.","It’s decent but not as durable as I hoped.",
    "I am absolutely in love with the quality of the products! Everything arrived on time and exceeded my expectations.",
    "Temu never disappoints! The item I ordered is exactly as described and works perfectly. Highly recommend.",
    "Great experience shopping on Temu. Fast shipping, great customer service, and the product is amazing!",
    "Temu's selection is top-notch! I found exactly what I was looking for, and the quality is fantastic.",
    "I’m so impressed with Temu! The item was well-packaged, arrived quickly, and looks exactly like the photos. Will buy again!",
    "Temu's products are always high quality, and I’ve had nothing but positive experiences. Delivery was prompt, and the item was perfect.",
    "The service at Temu was exceptional. I received my order on time and was so happy with the product. Definitely coming back!",
    "Temu always delivers the best! Fantastic product, quick shipping, and excellent customer service. Very satisfied with my purchase.",
    "I’ve ordered multiple times from Temu, and every single time I’m impressed with the quality and speed of delivery. Highly recommend!",
    "Temu is my go-to place for shopping! Everything I’ve ordered has been exactly what I expected, and customer support is always helpful.",
    "Very disappointed with my order. The product was damaged when it arrived, and the customer service was unhelpful in resolving the issue.",
    "I ordered from Temu, and it took forever to arrive. The product quality wasn’t as good as expected, and I won’t be buying again.",
    "Not happy with my purchase. The item was poorly made and didn't match the description at all. Waste of money.",
    "Temu has really gone downhill. The product I received was defective, and it was impossible to get a refund.",
    "The shipping was so slow. I waited weeks for my order, and when it finally arrived, it wasn’t what I ordered!",
    "Extremely disappointed in the quality. The item looked cheap and didn’t function as expected. I would not recommend Temu.",
    "I had such a frustrating experience with Temu. I ordered a phone case, and it was the wrong size. Trying to get customer service to help was a nightmare.",
    "Horrible experience. The product I bought broke after one week, and the return process is overly complicated.",
    "I ordered a pair of shoes from Temu, and they were nothing like the pictures. Poor quality and very uncomfortable. Very upset!",
    "I had high hopes for Temu, but my order took way too long to ship, and the product was not at all as described. Definitely won’t be ordering again.",
    "The product was fine, but the shipping took a bit longer than expected. Overall, not a bad experience, but nothing exceptional either.",
    "The item works as expected, though the quality isn’t amazing. It’s decent for the price, but I’m not sure if I’d buy it again.",
    "Temu’s service was average. The product was okay, but there wasn’t anything special about it. Shipping was on time, though.",
    "I received my order without any issues. It’s exactly as described, but the quality could be better. It’s an average product for the price.",
    "The purchase went smoothly, but I’m not overly impressed. The product does the job, but it’s nothing to rave about. Shipping was decent."
  ),
  sentiment = factor(c(
    "positive", "negative", "neutral", "positive", "negative","positive", "negative", "positive", "negative", "positive",
    "positive", "negative", "negative", "neutral", "positive","neutral", "negative", "positive", "negative", "positive",
    "negative", "negative", "positive", "positive", "negative", "neutral", "positive", "positive", "positive", "positive",
    "negative", "positive", "negative", "positive", "neutral", "positive", "negative", "positive", "positive", "neutral",
    "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative", "negative",  "negative", "negative", "negative", "negative", "negative",
    "neutral", "neutral", "neutral", "neutral", "neutral" )),stringsAsFactors = FALSE
)

# Data Preprocessing
# Clean and preprocess the review text
preprocess_text <- function(text) {
  text <- tolower(text)  # Convert text to lower case
  text <- removePunctuation(text)  # Remove punctuation
  text <- removeNumbers(text)  # Remove numbers
  text <- removeWords(text, stopwords("en"))  # Remove stopwords
  text <- stripWhitespace(text)  # Remove extra whitespace
  text <- wordStem(text)  # Stemming
  return(text)
}

# Apply preprocessing to reviews
reviews$cleaned_text <- sapply(reviews$review_text, preprocess_text)

# Create a Document-Term Matrix (DTM)
corpus <- Corpus(VectorSource(reviews$cleaned_text))
dtm <- DocumentTermMatrix(corpus)

# Convert DTM to a matrix
dtm_matrix <- as.matrix(dtm)

# Calculate word frequencies
word_freqs <- colSums(dtm_matrix)

# Visualize word frequencies in the reviews (Wordcloud)
wordcloud(names(word_freqs), freq = word_freqs, min.freq = 3, scale=c(3,0.5), colors=brewer.pal(8, "Dark2"))

# Split the dataset into training and testing into 80% and 20%.
set.seed(123)  # Set seed for reproducibility
trainIndex <- createDataPartition(reviews$sentiment, p = 0.8, list = FALSE)
train_data <- reviews[trainIndex, ]
test_data <- reviews[-trainIndex, ]

# Convert text data to a sparse matrix (for use in machine learning models)
train_dtm <- dtm_matrix[trainIndex, ]
test_dtm <- dtm_matrix[-trainIndex, ]

# Train the model using Random Forest with default ntree
set.seed(123)
rf_model <- randomForest(x = train_dtm, y = train_data$sentiment)

# Model Summary
print(rf_model)

# Make predictions on the test set
predictions <- predict(rf_model, test_dtm)

# Evaluate model performance using confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$sentiment)
print(conf_matrix)

# Show model accuracy
accuracy <- conf_matrix$overall['Accuracy']
cat("Model Accuracy: ", accuracy, "\n")

### Optimize the Model Performance
set.seed(123)
# Set up training control for 10-fold cross-validation
train_control <- trainControl(
  method = "cv",      # Cross-validation
  number = 10,        # Number of folds
  verboseIter = TRUE  # Show progress during training
)

# Define the model grid for hyperparameter tuning
tune_grid <- expand.grid(
  .mtry = floor(sqrt(ncol(train_dtm)))  # Default recommended value for mtry
)

# Train the Random Forest model with cross-validation
optimized_rf <- train(
  x = as.data.frame(train_dtm), 
  y = as.factor(train_data$sentiment), # Ensure target variable is a factor
  method = "rf",
  trControl = train_control,
  tuneGrid = tune_grid,
  ntree = 100                        # Set number of trees to 100
)

# Print the optimized model results
print(optimized_rf)


# Make predictions with the optimized model on the test set
predictions_op <- predict(optimized_rf, as.data.frame(test_dtm))

# Confusion Matrix for Optimized Model
confu_matrix <- confusionMatrix(predictions_op, test_data$sentiment)
print(confu_matrix) 

# Show optimized model accuracy
accuracy_op <- confu_matrix$overall['Accuracy']
cat("Optimized Model Accuracy: ", accuracy_op, "\n")


# Save the model using RDS.
saveRDS(optimized_rf, file = "C:\\Users\\deeps\\OneDrive\\Documents\\WEBSTER\\Analytics Practicum\\TEMU Sentiment Analysis\\SavedData\\Temu Sentiment Analysis - Model.RData")
saveRDS(dtm, file = "C:\\Users\\deeps\\OneDrive\\Documents\\WEBSTER\\Analytics Practicum\\TEMU Sentiment Analysis\\SavedData\\Temu Sentiment Analysis - DTM.RData")
saveRDS(train_dtm, file = "C:\\Users\\deeps\\OneDrive\\Documents\\WEBSTER\\Analytics Practicum\\TEMU Sentiment Analysis\\SavedData\\train_dtm.RData")






# Word cloud for positive, negative, and neutral reviews
wordcloud(words = unlist(strsplit(paste(train_data$cleaned_text[train_data$sentiment == "positive"], collapse = " "), " ")),
          min.freq = 3, scale = c(3, 0.5), colors = brewer.pal(8, "Set1"), main = "Positive Reviews Word Cloud")

wordcloud(words = unlist(strsplit(paste(train_data$cleaned_text[train_data$sentiment == "negative"], collapse = " "), " ")),
          min.freq = 3, scale = c(3, 0.5), colors = brewer.pal(8, "Set2"), main = "Negative Reviews Word Cloud")

wordcloud(words = unlist(strsplit(paste(train_data$cleaned_text[train_data$sentiment == "neutral"], collapse = " "), " ")),
          min.freq = 3, scale = c(3, 0.5), colors = brewer.pal(8, "Set3"), main = "Neutral Reviews Word Cloud")

