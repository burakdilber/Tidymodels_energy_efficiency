library(tidymodels)

library(readxl)
ENB2012_data <- read_excel("C:/Users/hp/Documents/ENB2012_data.xlsx")

set.seed(123)

##rsample

enb_split <- initial_split(ENB2012_data, prop = 0.75)
enb_split

enb_train <- training(enb_split)
enb_test  <- testing(enb_split)

###------------------------recipes---------------##
##recipes

## heating load
enb_recipe_hl <- 
  recipe(Y1 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = enb_train) %>%
  step_sqrt(all_predictors())

prepped_recipe_hl <- prep(enb_recipe_hl, training = enb_train)
prepped_recipe_hl

enb_train_preprocessed_hl <- bake(prepped_recipe_hl, enb_train) 
enb_train_preprocessed_hl

enb_test_preprocessed_hl <- bake(prepped_recipe_hl, enb_test)
enb_test_preprocessed_hl

set.seed(123)
enb_cv_preprocessed_hl <- vfold_cv(enb_train_preprocessed_hl, v = 10)
enb_cv_preprocessed_hl


## cooling load
enb_recipe_cl <- 
  recipe(Y2 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = enb_train) %>%
  step_sqrt(all_predictors())

prepped_recipe_cl <- prep(enb_recipe_cl, training = enb_train)
prepped_recipe_cl

enb_train_preprocessed_cl <- bake(prepped_recipe_cl, enb_train) 
enb_train_preprocessed_cl

enb_test_preprocessed_cl <- bake(prepped_recipe_cl, enb_test)
enb_test_preprocessed_cl

set.seed(123)
enb_cv_preprocessed_cl <- vfold_cv(enb_train_preprocessed_cl, v = 10)
enb_cv_preprocessed_cl


####-----------------------Linear Regression------------------------------

## Heating Load ------------------------------------

linear_model <- 
  linear_reg(penalty = tune(),
             mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("regression")

set.seed(123)
linear_wf <-
  workflow() %>%
  add_model(linear_model) %>% 
  add_recipe(enb_recipe_hl)
linear_wf

linear_results <-
  linear_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_hl)

linear_results %>% 
  show_best(metric = "rmse")

linear_best <-
  linear_results %>% 
  select_best(metric = "rmse")
linear_best

linear_model_final <- 
  linear_reg(penalty = linear_best$penalty,
             mixture = linear_best$mixture) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")
linear_model_final

##workflow
linear_wflow <- 
  linear_wf %>% 
  update_model(linear_model_final)
linear_wflow

linear_fit <- fit(linear_wflow, enb_train_preprocessed_hl)

##performance (yardstick)
enb_test_res <- predict(linear_fit, new_data = enb_test_preprocessed_hl %>% select(-Y1))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_hl %>% select(Y1))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y1, estimate = .pred)


## Cooling Load ------------------------------------

linear_model <- 
  linear_reg(penalty = tune(),
             mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("regression")

set.seed(123)
linear_wf <-
  workflow() %>%
  add_model(linear_model) %>% 
  add_recipe(enb_recipe_cl)
linear_wf

linear_results <-
  linear_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_cl)

linear_results %>% 
  show_best(metric = "rmse")

linear_best <-
  linear_results %>% 
  select_best(metric = "rmse")
linear_best

linear_model_final <- 
  linear_reg(penalty = linear_best$penalty,
             mixture = linear_best$mixture) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")
linear_model_final

##workflow
linear_wflow <- 
  linear_wf %>% 
  update_model(linear_model_final)
linear_wflow

linear_fit <- fit(linear_wflow, enb_train_preprocessed_cl)

##performance (yardstick)
enb_test_res <- predict(linear_fit, new_data = enb_test_preprocessed_cl %>% select(-Y2))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_cl %>% select(Y2))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y2, estimate = .pred)


####-----------------------Quantile Regression------------------------------

## Heating Load ----------------------

##parsnip
quantile_model <- 
  quantile_reg() %>% 
  set_engine("quantreg") %>%
  translate()
quantile_model

quantile_form_fit <- 
  quantile_model %>% 
  fit(Y1 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = enb_train_preprocessed_hl)
quantile_form_fit

model_res <- 
  quantile_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res


##test data
enb_test_small <- enb_test_preprocessed_hl %>% slice(1:5)
predict(quantile_form_fit, new_data = enb_test_small)


##workflow
quantile_wflow <- 
  workflow() %>% 
  add_model(quantile_model)

quantile_wflow

quantile_wflow <- 
  quantile_wflow %>% 
  add_formula(Y1 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8)

quantile_wflow

quantile_fit <- fit(quantile_wflow, enb_train_preprocessed_hl)
quantile_fit

predict(quantile_fit, enb_test_preprocessed_hl %>% slice(1:3))


quantile_fit %>% 
  pull_workflow_fit() %>% 
  tidy() %>% 
  slice(1:5)

##performance (yardstick)
enb_test_res <- predict(quantile_fit, new_data = enb_test_preprocessed_hl %>% select(-Y1))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_hl %>% select(Y1))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y1, estimate = .pred)


## Cooling Load ----------------------------

##parsnip
quantile_model <- 
  quantile_reg() %>% 
  set_engine("quantreg") %>%
  translate()
quantile_model

quantile_form_fit <- 
  quantile_model %>% 
  fit(Y2 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = enb_train_preprocessed_cl)
quantile_form_fit

model_res <- 
  quantile_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res


##test data
enb_test_small <- enb_test_preprocessed_cl %>% slice(1:5)
predict(quantile_form_fit, new_data = enb_test_small)


##workflow
quantile_wflow <- 
  workflow() %>% 
  add_model(quantile_model)

quantile_wflow

quantile_wflow <- 
  quantile_wflow %>% 
  add_formula(Y2 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8)

quantile_wflow

quantile_fit <- fit(quantile_wflow, enb_train_preprocessed_cl)
quantile_fit

predict(quantile_fit, enb_test_preprocessed_cl %>% slice(1:3))


quantile_fit %>% 
  pull_workflow_fit() %>% 
  tidy() %>% 
  slice(1:5)

##performance (yardstick)
enb_test_res <- predict(quantile_fit, new_data = enb_test_preprocessed_cl %>% select(-Y2))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_cl %>% select(Y2))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y2, estimate = .pred)


####-----------------------Boosted Trees------------------------------

## Heating Load ------------------------------------

boost_model <- 
  boost_tree( tree_depth = tune(),
              trees = tune(),
              mtry = tune(),
              min_n = tune(),
              sample_size = tune()
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

set.seed(123)
boost_wf <-
  workflow() %>%
  add_model(boost_model) %>% 
  add_recipe(enb_recipe_hl)
boost_wf

boost_results <-
  boost_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_hl)

boost_results %>% 
  show_best(metric = "rmse")

boost_best <-
  boost_results %>% 
  select_best(metric = "rmse")
boost_best

boost_model_final <- 
  boost_tree(mtry = boost_best$mtry,
             trees = boost_best$trees,
             min_n = boost_best$min_n,
             tree_depth = boost_best$tree_depth,
             sample_size = boost_best$sample_size) %>% 
  set_engine("xgboost") %>%
  set_mode("regression")
boost_model_final

##workflow
boost_wflow <- 
  boost_wf %>% 
  update_model(boost_model_final)
boost_wflow

boost_fit <- fit(boost_wflow, enb_train_preprocessed_hl)

##performance (yardstick)
enb_test_res <- predict(boost_fit, new_data = enb_test_preprocessed_hl %>% select(-Y1))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_hl %>% select(Y1))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y1, estimate = .pred)


## Cooling Load ------------------------------------

boost_model <- 
  boost_tree( tree_depth = tune(),
              trees = tune(),
              mtry = tune(),
              min_n = tune(),
              sample_size = tune()
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

set.seed(123)
boost_wf <-
  workflow() %>%
  add_model(boost_model) %>% 
  add_recipe(enb_recipe_cl)
boost_wf

boost_results <-
  boost_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_cl)

boost_results %>% 
  show_best(metric = "rmse")

boost_best <-
  boost_results %>% 
  select_best(metric = "rmse")
boost_best

boost_model_final <- 
  boost_tree(mtry = boost_best$mtry,
             trees = boost_best$trees,
             min_n = boost_best$min_n,
             tree_depth = boost_best$tree_depth,
             sample_size = boost_best$sample_size) %>% 
  set_engine("xgboost") %>%
  set_mode("regression")
boost_model_final

##workflow
boost_wflow <- 
  boost_wf %>% 
  update_model(boost_model_final)
boost_wflow

boost_fit <- fit(boost_wflow, enb_train_preprocessed_cl)

##performance (yardstick)
enb_test_res <- predict(boost_fit, new_data = enb_test_preprocessed_cl %>% select(-Y2))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_cl %>% select(Y2))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y2, estimate = .pred)


### --------------------------Decision Tree-----------------------------------

## Heating Load ------------------------------------

decision_model <- 
  decision_tree( tree_depth = tune(),
                 min_n = tune(),
                 cost_complexity = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

set.seed(123)
decision_wf <-
  workflow() %>%
  add_model(decision_model) %>% 
  add_recipe(enb_recipe_hl)
decision_wf

decision_results <-
  decision_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_hl)

decision_results %>% 
  show_best(metric = "rmse")

decision_best <-
  decision_results %>% 
  select_best(metric = "rmse")
decision_best

decision_model_final <- 
  decision_tree(cost_complexity = decision_best$cost_complexity,
                tree_depth = decision_best$tree_depth,
                min_n = decision_best$min_n) %>% 
  set_engine("rpart") %>%
  set_mode("regression")
decision_model_final

##workflow
decision_wflow <- 
  decision_wf %>% 
  update_model(decision_model_final)
decision_wflow

decision_fit <- fit(decision_wflow, enb_train_preprocessed_hl)


##performance (yardstick)
enb_test_res <- predict(decision_fit, new_data = enb_test_preprocessed_hl %>% select(-Y1))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_hl %>% select(Y1))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y1, estimate = .pred)


## Cooling Load ------------------------------------

decision_model <- 
  decision_tree( tree_depth = tune(),
                 min_n = tune(),
                 cost_complexity = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

set.seed(123)
decision_wf <-
  workflow() %>%
  add_model(decision_model) %>% 
  add_recipe(enb_recipe_cl)
decision_wf

decision_results <-
  decision_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_cl)

decision_results %>% 
  show_best(metric = "rmse")

decision_best <-
  decision_results %>% 
  select_best(metric = "rmse")
decision_best

decision_model_final <- 
  decision_tree(cost_complexity = decision_best$cost_complexity,
                tree_depth = decision_best$tree_depth,
                min_n = decision_best$min_n) %>% 
  set_engine("rpart") %>%
  set_mode("regression")
decision_model_final

##workflow
decision_wflow <- 
  decision_wf %>% 
  update_model(decision_model_final)
decision_wflow

decision_fit <- fit(decision_wflow, enb_train_preprocessed_cl)


##performance (yardstick)
enb_test_res <- predict(decision_fit, new_data = enb_test_preprocessed_cl %>% select(-Y2))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_cl %>% select(Y2))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y2, estimate = .pred)


## ------------------------ K - Nearest Neighbor --------------------------

## Heating Load ------------------------------------

knn_model <- 
  nearest_neighbor( neighbors = tune(),
                    weight_func = tune(),
                    dist_power = tune()
  ) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

set.seed(123)
knn_wf <-
  workflow() %>%
  add_model(knn_model) %>% 
  add_recipe(enb_recipe_hl)
knn_wf

knn_results <-
  knn_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_hl)

knn_results %>% 
  show_best(metric = "rmse")

knn_best <-
  knn_results %>% 
  select_best(metric = "rmse")
knn_best

knn_model_final <- 
  nearest_neighbor(neighbors = knn_best$neighbors,
                   weight_func = knn_best$weight_func,
                   dist_power = knn_best$dist_power) %>% 
  set_engine("kknn") %>%
  set_mode("regression")
knn_model_final

##workflow
knn_wflow <- 
  knn_wf %>% 
  update_model(knn_model_final)
knn_wflow

knn_fit <- fit(knn_wflow, enb_train_preprocessed_hl)

##performance (yardstick)
enb_test_res <- predict(knn_fit, new_data = enb_test_preprocessed_hl %>% select(-Y1))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_hl %>% select(Y1))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y1, estimate = .pred)


## Cooling Load ------------------------------------

knn_model <- 
  nearest_neighbor( neighbors = tune(),
                    weight_func = tune(),
                    dist_power = tune()
  ) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

set.seed(123)
knn_wf <-
  workflow() %>%
  add_model(knn_model) %>% 
  add_recipe(enb_recipe_cl)
knn_wf

knn_results <-
  knn_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_cl)

knn_results %>% 
  show_best(metric = "rmse")

knn_best <-
  knn_results %>% 
  select_best(metric = "rmse")
knn_best

knn_model_final <- 
  nearest_neighbor(neighbors = knn_best$neighbors,
                   weight_func = knn_best$weight_func,
                   dist_power = knn_best$dist_power) %>% 
  set_engine("kknn") %>%
  set_mode("regression")
knn_model_final

##workflow
knn_wflow <- 
  knn_wf %>% 
  update_model(knn_model_final)
knn_wflow

knn_fit <- fit(knn_wflow, enb_train_preprocessed_cl)

##performance (yardstick)
enb_test_res <- predict(knn_fit, new_data = enb_test_preprocessed_cl %>% select(-Y2))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_cl %>% select(Y2))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y2, estimate = .pred)


### --------------------------- Random Forest ------------------------------------

## Heating Load ------------------------------------

rf_model <- 
  rand_forest( mtry = tune(),
               trees = tune(),
               min_n = tune()
  ) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

set.seed(123)
rf_wf <-
  workflow() %>%
  add_model(rf_model) %>% 
  add_recipe(enb_recipe_hl)
rf_wf

rf_results <-
  rf_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_hl)

rf_results %>% 
  show_best(metric = "rmse")

rf_best <-
  rf_results %>% 
  select_best(metric = "rmse")
rf_best

rf_model_final <- 
  rand_forest(mtry = rf_best$mtry,
              trees = rf_best$trees,
              min_n = rf_best$min_n) %>% 
  set_engine("ranger") %>%
  set_mode("regression")
rf_model_final

##workflow
rf_wflow <- 
  rf_wf %>% 
  update_model(rf_model_final)
rf_wflow

rf_fit <- fit(rf_wflow, enb_train_preprocessed_hl)

##performance (yardstick)
enb_test_res <- predict(rf_fit, new_data = enb_test_preprocessed_hl %>% select(-Y1))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_hl %>% select(Y1))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y1, estimate = .pred)


## Cooling Load ------------------------------------

rf_model <- 
  rand_forest( mtry = tune(),
               trees = tune(),
               min_n = tune()
  ) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

set.seed(123)
rf_wf <-
  workflow() %>%
  add_model(rf_model) %>% 
  add_recipe(enb_recipe_cl)
rf_wf

rf_results <-
  rf_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_cl)

rf_results %>% 
  show_best(metric = "rmse")

rf_best <-
  rf_results %>% 
  select_best(metric = "rmse")
rf_best

rf_model_final <- 
  rand_forest(mtry = rf_best$mtry,
              trees = rf_best$trees,
              min_n = rf_best$min_n) %>% 
  set_engine("ranger") %>%
  set_mode("regression")
rf_model_final

##workflow
rf_wflow <- 
  rf_wf %>% 
  update_model(rf_model_final)
rf_wflow

rf_fit <- fit(rf_wflow, enb_train_preprocessed_cl)

##performance (yardstick)
enb_test_res <- predict(rf_fit, new_data = enb_test_preprocessed_cl %>% select(-Y2))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_cl %>% select(Y2))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y2, estimate = .pred)


## --------------------------- Support Vector Machine ---------------------

## Heating Load ------------------------------------

svm_model <- 
  svm_rbf(  cost = tune(),
            rbf_sigma = tune(),
            margin = tune()
  ) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

set.seed(123)
svm_wf <-
  workflow() %>%
  add_model(svm_model) %>% 
  add_recipe(enb_recipe_hl)
svm_wf

svm_results <-
  svm_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_hl)

svm_results %>% 
  show_best(metric = "rmse")

svm_best <-
  svm_results %>% 
  select_best(metric = "rmse")
svm_best

svm_model_final <- 
  svm_rbf(cost = svm_best$cost,
           rbf_sigma = svm_best$rbf_sigma,
           margin = svm_best$margin) %>% 
  set_engine("kernlab") %>%
  set_mode("regression")
svm_model_final

##workflow
svm_wflow <- 
  svm_wf %>% 
  update_model(svm_model_final)
svm_wflow

svm_fit <- fit(svm_wflow, enb_train_preprocessed_hl)

##permormance (yardstick)
enb_test_res <- predict(svm_fit, new_data = enb_test_preprocessed_hl %>% select(-Y1))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_hl %>% select(Y1))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y1, estimate = .pred)


## Cooling Load ------------------------------------

svm_model <- 
  svm_rbf(  cost = tune(),
            rbf_sigma = tune(),
            margin = tune()
  ) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

set.seed(123)
svm_wf <-
  workflow() %>%
  add_model(svm_model) %>% 
  add_recipe(enb_recipe_cl)
svm_wf

svm_results <-
  svm_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_cl)

svm_results %>% 
  show_best(metric = "rmse")

svm_best <-
  svm_results %>% 
  select_best(metric = "rmse")
svm_best

svm_model_final <- 
  svm_rbf(cost = svm_best$cost,
          rbf_sigma = svm_best$rbf_sigma,
          margin = svm_best$margin) %>% 
  set_engine("kernlab") %>%
  set_mode("regression")
svm_model_final

##workflow
svm_wflow <- 
  svm_wf %>% 
  update_model(svm_model_final)
svm_wflow

svm_fit <- fit(svm_wflow, enb_train_preprocessed_cl)

##permormance (yardstick)
enb_test_res <- predict(svm_fit, new_data = enb_test_preprocessed_cl %>% select(-Y2))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_cl %>% select(Y2))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y2, estimate = .pred)


## --------------------------- Neural Networks ---------------------

## Heating Load ------------------------------------

nn_model <- 
  mlp( hidden_units = tune(),
       penalty = tune(),
       epochs = tune()
  ) %>% 
  set_engine("nnet") %>% 
  set_mode("regression")

set.seed(123)
nn_wf <-
  workflow() %>%
  add_model(nn_model) %>% 
  add_recipe(enb_recipe_hl)
nn_wf

nn_results <-
  nn_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_hl)

nn_results %>% 
  show_best(metric = "rmse")

nn_best <-
  nn_results %>% 
  select_best(metric = "rmse")
nn_best

nn_model_final <- 
  mlp( hidden_units = nn_best$hidden_units,
       penalty = nn_best$penalty,
       epochs = nn_best$epochs
  ) %>% 
  set_engine("nnet") %>% 
  set_mode("regression")
nn_model_final

##workflow
nn_wflow <- 
  nn_wf %>% 
  update_model(nn_model_final)
nn_wflow

nn_fit <- fit(nn_wflow, enb_train_preprocessed_hl)

##permormance (yardstick)
enb_test_res <- predict(nn_fit, new_data = enb_test_preprocessed_hl %>% select(-Y1))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_hl %>% select(Y1))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y1, estimate = .pred)


## Cooling Load ------------------------------------

nn_model <- 
  mlp( hidden_units = tune(),
       penalty = tune(),
       epochs = tune()
  ) %>% 
  set_engine("nnet") %>% 
  set_mode("regression")

set.seed(123)
nn_wf <-
  workflow() %>%
  add_model(nn_model) %>% 
  add_recipe(enb_recipe_cl)
nn_wf

nn_results <-
  nn_wf %>% 
  tune_grid(resamples = enb_cv_preprocessed_cl)

nn_results %>% 
  show_best(metric = "rmse")

nn_best <-
  nn_results %>% 
  select_best(metric = "rmse")
nn_best

nn_model_final <- 
  mlp( hidden_units = nn_best$hidden_units,
       penalty = nn_best$penalty,
       epochs = nn_best$epochs
  ) %>% 
  set_engine("nnet") %>% 
  set_mode("regression")
nn_model_final

##workflow
nn_wflow <- 
  nn_wf %>% 
  update_model(nn_model_final)
nn_wflow

nn_fit <- fit(nn_wflow, enb_train_preprocessed_cl)

##permormance (yardstick)
enb_test_res <- predict(nn_fit, new_data = enb_test_preprocessed_cl %>% select(-Y2))
enb_test_res

enb_test_res <- bind_cols(enb_test_res, enb_test_preprocessed_cl %>% select(Y2))
enb_test_res

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(enb_test_res, truth = Y2, estimate = .pred)
