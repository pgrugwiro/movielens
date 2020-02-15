################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")
library(lubridate)

# String processing to extract the movie release year and the rating date.
movielens <- movielens %>% mutate(rating_date = as_datetime(timestamp), rating_year = as.character(format(as_datetime(timestamp), '%Y'))) %>% 
  mutate(release_year = str_extract(movielens$title,  "\\d{4}"))



# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#EDA: Exploratory Data Analysis


#EDA - Understand the distribution of Users vs. Ratings per user:
edx %>% group_by(userId) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30) + 
  scale_x_log10() +
  xlab("Total Ratings per User") + 
  ylab("Number of Users")

#EDA - Understand the distribution of Movies vs. Ratings per movie:
edx %>% group_by(movieId) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30) + 
  scale_x_log10() +
  xlab("Total Ratings per Movie") + 
  ylab("Number of Movies")

#EDA - Understand if movies that came out in a certain time period have more ratings:
edx %>% group_by(movieId) %>% 
  summarize(N_ratings = n(), year_out = as.character(first(release_year))) %>%
  ggplot(aes(year_out, N_ratings)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90)) +
  scale_y_log10()+
  xlab("Release Year")+
  ylab("Total Ratings")


#EDA - Understand if the average rating has changed over time as new movies came out
edx %>%  group_by(release_year) %>%
  filter(release_year < 2020 & release_year > 1900) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(as.numeric(release_year), rating)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme(axis.text.x = element_text(angle = 90))+
  xlab("Release Year")+
  ylab("Avg Rating")


#EDA - Understand if the average rating has changed over time as users get more critical
edx %>% mutate(rating_date = round_date(rating_date, unit = "week")) %>%
  group_by(rating_date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(rating_date, rating)) +
  geom_point() +
  geom_smooth(method = "lm") +
  xlab("Time of Rating")+
  ylab("Avg Rating")


edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 50000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  xlab("Movie Category (genres)") +
  ylab("Average Rating")



#CREATE THE TRAINING AND TESTING SETS: Train on 90% of data, and test on the other 10%

set.seed(2, sample.kind="Rounding")

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in validation set are also in edx set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

              
# Define the RMSE function to be used at the validation state
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#PREDICTION
# 0. Predict using just the mean rating for every movie in the test set:

mu <- mean(train_set$rating) 
predicted_ratings_0 <- rep(mu, length(test_set$rating))
model_0_rmse <- RMSE(predicted_ratings_0, test_set$rating)
model_0_rmse


# 1. Predict the rating, accounting for movie effects, b_i

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

predicted_ratings_1 <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>% .$b_i
model_1_rmse <- RMSE(predicted_ratings_1, test_set$rating)
model_1_rmse


# 2. Predict the rating, accounting for user effect, b_u

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

user_avgs %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("black"))

predicted_ratings_2 <- test_set %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movie_avgs, by = "movieId") %>% 
  mutate(prediction =mu+b_i+b_u) %>% 
  pull(prediction)
model_2_rmse <- RMSE(predicted_ratings_2, test_set$rating)
model_2_rmse


# 3. Predict the rating, accounting for effect of time of rating

time_avgs <- train_set %>%
  mutate(week = round_date(rating_date, unit = "week")) %>%
  left_join(movie_avgs, by= "movieId") %>%
  left_join(user_avgs, by= "userId") %>%
  group_by(week) %>%
  summarize(ti = mean(rating - b_i - b_u - mu))

time_avgs %>% qplot(ti, geom ="histogram", bins = 20, data = ., color = I("black"))

predicted_ratings_3 <- test_set %>%
  mutate(week = round_date(rating_date, unit = "week")) %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(time_avgs, by = "week") %>%
  mutate(prediction =mu+b_i+b_u+ti) %>% 
  pull(prediction)
model_3_rmse <- RMSE(predicted_ratings_3, test_set$rating)
model_3_rmse


# 4. Predict the rating, accounting for the effect of movie genre


genre_avgs <- train_set %>%
  mutate(week = round_date(rating_date, unit = "week")) %>%
  left_join(movie_avgs, by= "movieId") %>%
  left_join(user_avgs, by= "userId") %>%
  group_by(week) %>%
  left_join(time_avgs, by= "week") %>%
  ungroup() %>%
  group_by(genres) %>% 
  summarize(ge = mean(rating - ti - b_i - b_u - mu))

genre_avgs %>% qplot(ge, geom ="histogram", bins = 20, data = ., color = I("black"))

predicted_ratings_4 <- test_set %>%
  mutate(week = round_date(rating_date, unit = "week")) %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(time_avgs, by = "week") %>%
  left_join(genre_avgs, by = "genres") %>%
  mutate(prediction =mu+b_i+b_u+ti+ge) %>% 
  pull(prediction)
model_4_rmse <- RMSE(predicted_ratings_4, test_set$rating)
model_4_rmse


#5. REGULARIZATION OF ALL PARAMETERS ABOVE::

#regularization parameter lambda
l <- 5.5

#movie (item) effect  
b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

#user effect  
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

#time of rating effect  
ti <- train_set %>%
  mutate(week = round_date(rating_date, unit = "week")) %>%
  left_join(b_i, by= "movieId") %>%
  left_join(b_u, by= "userId") %>%
  group_by(week) %>%
  summarize(ti = sum(rating - b_i - b_u - mu)/(n()+l))

#genre effect  
ge <- train_set %>%
  mutate(week = round_date(rating_date, unit = "week")) %>%
  left_join(b_i, by= "movieId") %>%
  left_join(b_u, by= "userId") %>%
  group_by(week) %>%
  left_join(ti, by= "week") %>% ungroup() %>%
  group_by(genres) %>%
  summarize(ge = sum(rating - b_i - b_u - mu - ti)/(n()+l))
  
#regularized parameters prediction  
predicted_ratings_5 <- test_set %>% 
  mutate(week = round_date(rating_date, unit = "week")) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(ge, by = "genres") %>%
  left_join(ti, by = "week") %>%
  mutate(pred = mu + b_i + b_u + ti + ge) %>%
  .$pred
model_5_rmse <-  RMSE(predicted_ratings_5, test_set$rating)
model_5_rmse


#6. Final Prediction with the Validation Set

final_ratings <- 
  validation %>% 
  mutate(week = round_date(rating_date, unit = "week")) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(ge, by = "genres") %>%
  left_join(ti, by = "week") %>%
  mutate(pred = mu + b_i + b_u + ti + ge) %>%
  .$pred

final_rmse <- RMSE(final_ratings, validation$rating)
final_rmse


rmse_results <- data_frame(Method= c("Just Average", "Movie Effect", "User Effect",
                                     "Time of Rating Effect", "Genre Effect", 
                                     "Regularized Parameters", "Final RMSE/Validation Set"),
                           RMSE_result =c(model_0_rmse, model_1_rmse, model_2_rmse,
                                          model_3_rmse, model_4_rmse, model_5_rmse,
                                          final_rmse))


#SUMMARY TABLE
rmse_results %>% knitr::kable()

####################################################################
####################################################################
############################END#####################################


#LOOP TO DETERMINE THE OPTIMAL LAMBDA :: DO NOT RUN
#lambdas <- seq(1,20,0.5)

#rmses <- sapply(lambdas, function(l){
  
#  b_i <- train_set %>%
#    group_by(movieId) %>%
#    summarize(b_i = sum(rating - mu)/(n()+l))
  
#  b_u <- train_set %>% 
#    left_join(b_i, by="movieId") %>%
#    group_by(userId) %>%
#    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
#  ti <- train_set %>%
#    mutate(week = round_date(rating_date, unit = "week")) %>%
#    left_join(b_i, by= "movieId") %>%
#    left_join(b_u, by= "userId") %>%
#    group_by(week) %>%
#   summarize(ti = sum(rating - b_i - b_u - mu)/(n()+l))
#   
#   ge <- train_set %>%
#     mutate(week = round_date(rating_date, unit = "week")) %>%
#     left_join(b_i, by= "movieId") %>%
#     left_join(b_u, by= "userId") %>%
#     group_by(week) %>%
#     left_join(ti, by= "week") %>% ungroup() %>%
#     group_by(genres) %>%
#     summarize(ge = sum(rating - b_i - b_u - mu - ti)/(n()+l))
#   
#   
#   predicted_ratings_5 <- test_set %>% 
#     mutate(week = round_date(rating_date, unit = "week")) %>%
#     left_join(b_i, by = "movieId") %>%
#     left_join(b_u, by = "userId") %>%
#     left_join(ge, by = "genres") %>%
#     left_join(ti, by = "week") %>%
#     mutate(pred = mu + b_i + b_u + ti + ge) %>%
#     .$pred
#   
#   
#   RMSE(predicted_ratings_5, test_set$rating)
#   
#   
# })
# lambdas[which.min(rmses)]
