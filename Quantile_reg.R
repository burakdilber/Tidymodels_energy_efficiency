str(quantreg::rq)
library(tidymodels)

set_new_model("quantile_reg")
set_model_mode(model = "quantile_reg", mode = "regression")
set_model_engine(
  "quantile_reg",
  mode = "regression",
  eng = "quantreg"
)
set_dependency("quantile_reg", eng = "quantreg", pkg = "quantreg")

show_model_info("quantile_reg")


set_model_arg(
  model = "quantile_reg",
  eng = "quantreg",
  parsnip = "tau",
  original = "tau",
  func = list(pkg = "quantreg", fun = "rq"),
  has_submodel = FALSE
)

set_model_arg(
  model = "quantile_reg",
  eng = "quantreg",
  parsnip = "sub_set",
  original = "subset",
  func = list(pkg = "quantreg", fun = "rq"),
  has_submodel = FALSE
)

set_model_arg(
  model = "quantile_reg",
  eng = "quantreg",
  parsnip = "weights",
  original = "weights",
  func = list(pkg = "quantreg", fun = "rq"),
  has_submodel = FALSE
)

set_model_arg(
  model = "quantile_reg",
  eng = "quantreg",
  parsnip = "method",
  original = "method",
  func = list(pkg = "quantreg", fun = "rq"),
  has_submodel = FALSE
)

show_model_info("quantile_reg")


quantile_reg <-
  function(mode = "regression", tau = 0.5, sub_set = NULL, weights = NULL, method = "br") {
    # Check for correct mode
    if (mode  != "regression") {
      stop("`mode` should be 'regression'", call. = FALSE)
    }
    
    # Capture the arguments in quosures
    args <- list(sub_set = rlang::enquo(sub_set), 
                 weights = rlang::enquo(weights),
                 method = rlang::enquo(method),
                 tau = rlang::enquo(tau))
    
    # Save some empty slots for future parts of the specification
    out <- list(args = args, eng_args = NULL,
                mode = mode, method = NULL, engine = NULL)
    
    # set classes in the correct order
    class(out) <- make_classes("quantile_reg")
    out
  }

set_fit(
  model = "quantile_reg",
  eng = "quantreg",
  mode = "regression",
  value = list(
    interface = "formula",
    protect = c("formula", "data"),
    func = c(pkg = "quantreg", fun = "rq"),
    defaults = list()
  )
)

show_model_info("quantile_reg")


set_encoding(
  model = "quantile_reg",
  eng = "quantreg",
  mode = "regression",
  options = list(
    predictor_indicators = "traditional",
    compute_intercept = TRUE,
    remove_intercept = TRUE,
    allow_sparse_x = FALSE
  )
)

set_pred(
  model = "quantile_reg",
  eng = "quantreg",
  mode = "regression",
  type = "numeric",
  value = list(
    post = NULL,
    pre = NULL,
    func = c(fun = "predict"),
    args =
      list(
        object = expr(object$fit),
        newdata = expr(new_data),
        type = "response"
      )
  )
)

show_model_info("quantile_reg")

quantile_reg() %>% 
  set_engine("quantreg") %>% 
  set_mode("regression") %>%
  fit(mpg ~ ., data = mtcars)
