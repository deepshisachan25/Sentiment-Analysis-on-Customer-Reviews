# Load necessary libraries
library(shiny)

optimized_rf_Temu = readRDS("C:\\Users\\deeps\\OneDrive\\Documents\\WEBSTER\\Analytics Practicum\\TEMU Sentiment Analysis\\SavedData\\Temu Sentiment Analysis - Model.RData")
dtm_Temu = readRDS("C:\\Users\\deeps\\OneDrive\\Documents\\WEBSTER\\Analytics Practicum\\TEMU Sentiment Analysis\\SavedData\\Temu Sentiment Analysis - DTM.RData")
train_dtm = readRDS("C:\\Users\\deeps\\OneDrive\\Documents\\WEBSTER\\Analytics Practicum\\TEMU Sentiment Analysis\\SavedData\\train_dtm.RData")

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

### Shiny UI and Server

# Define UI for the Shiny App
ui <- fluidPage(
  titlePanel("Temu Customer Review Sentiment Analysis"),
  sidebarLayout(
    sidebarPanel(
      textAreaInput("review", "Enter your review:", rows = 5, placeholder = "Type your review here..."),
      actionButton("submit", "Submit Review")
    ),
    mainPanel(
      h4("Sentiment Prediction:"),
      textOutput("sentiment_prediction"),
      h4("Word Cloud of Input Review:"),
      plotOutput("wordcloud_plot")
    )
  )
)

# Define Server Logic for the Shiny App
server <- function(input, output) {
  
  observeEvent(input$submit, {
    # Check if input is not empty
    if (nchar(input$review) > 0) {
      # Preprocess input review
      user_review <- preprocess_text(input$review)
      
      # Create DTM for prediction using the same terms as training data
      user_corpus <- Corpus(VectorSource(user_review))
      user_dtm <- DocumentTermMatrix(user_corpus, control = list(dictionary = Terms(dtm_Temu)))
      user_dtm_df <- as.data.frame(as.matrix(user_dtm))
      colnames(user_dtm_df) <- make.names(colnames(user_dtm_df))
      
      # Handle missing terms by adding zero columns
      missing_terms <- setdiff(colnames(train_dtm), colnames(user_dtm_df))
      if(length(missing_terms) > 0){
        for(term in missing_terms){
          user_dtm_df[[term]] <- 0
        }
      }
      
      # Align columns to match training data
      user_dtm_df <- user_dtm_df[, colnames(train_dtm)]
      
      # Predict sentiment using the optimized model
      prediction <- predict(optimized_rf_Temu, as.data.frame(user_dtm_df))
      
      # Render sentiment prediction
      output$sentiment_prediction <- renderText({ paste("Sentiment: ", prediction) })
      
      # Generate word cloud for the input review
      output$wordcloud_plot <- renderPlot({
        wordcloud(
          words = unlist(strsplit(user_review, " ")),
          min.freq = 1,
          scale = c(3, 0.5),
          colors = brewer.pal(8, "Dark2")
        )
      })
    } else {
      output$sentiment_prediction <- renderText({ "Please enter a review to analyze." })
      output$wordcloud_plot <- renderPlot({ NULL })
    }
  })
}

# Run the Shiny App
shinyApp(ui = ui, server = server)

##Not happy with my purchase. Very bad experience, the quality is poor
##Overall experience was good, will buy it again from temu. Faster delivery than expected.
##The product quality was worst. Dissatisfied.
## Products are okay but not as durable as I hoped.

## Quality is not great, not satisfied. 
## I received a smaller item than expected. However, the product itself is okay..