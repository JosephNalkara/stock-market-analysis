# ============================================================
#  Stock Market Trend Analysis & Forecasting
#  Synthetic Data — Masters CS Project
# ============================================================
#  Methods covered:
#    1. Data generation (matches widget parameters)
#    2. Exploratory Data Analysis (EDA)
#    3. Trend detection — Linear Regression, Moving Averages
#    4. Stationarity test — ADF test
#    5. Decomposition — STL
#    6. ARIMA forecasting
#    7. Exponential Smoothing (ETS)
#    8. LSTM-style rolling-window regression (via neuralnet)
#    9. Performance metrics — RMSE, MAE, MAPE
#   10. Forecast plots
# ============================================================

# ── 0. Install & load packages ────────────────────────────────
packages <- c("tidyverse", "lubridate", "tseries", "forecast",
              "TTR", "ggplot2", "gridExtra", "zoo", "Metrics",
              "neuralnet", "scales")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}
invisible(lapply(packages, install_if_missing))

set.seed(42)

# ── 1. Synthetic Data Generation ─────────────────────────────
# Mirrors the widget logic: weekdays only, seeded trend + volatility

generate_stock_data <- function(stock_name, base_price, volatility,
                                 trend, n_days = 252) {
  all_dates <- seq.Date(as.Date("2025-03-16"), by = "day", length.out = n_days * 1.5)
  trading_dates <- all_dates[!weekdays(all_dates) %in% c("Saturday", "Sunday")]
  trading_dates <- head(trading_dates, n_days)

  prices <- numeric(n_days)
  volumes <- numeric(n_days)
  prices[1] <- base_price

  for (i in 2:n_days) {
    shock     <- rnorm(1, mean = 0, sd = volatility * prices[i - 1])
    drift     <- trend * prices[i - 1]
    prices[i] <- max(prices[i - 1] + shock + drift, base_price * 0.5)
    volumes[i] <- round(runif(1, 20e6, 50e6))
  }

  tibble(
    Date   = trading_dates,
    Stock  = stock_name,
    Open   = round(prices * runif(n_days, 0.99, 1.00), 2),
    High   = round(prices * runif(n_days, 1.005, 1.02), 2),
    Low    = round(prices * runif(n_days, 0.980, 0.995), 2),
    Close  = round(prices, 2),
    Volume = volumes
  )
}

# Four stocks — same parameters as the widget
stocks_raw <- bind_rows(
  generate_stock_data("AAPL",  172, 0.018, 0.0004),
  generate_stock_data("GOOGL", 138, 0.022, 0.0003),
  generate_stock_data("MSFT",  415, 0.016, 0.0005),
  generate_stock_data("AMZN",  185, 0.024, 0.0003)
)

cat("=== Dataset Overview ===\n")
cat("Rows:", nrow(stocks_raw), " | Stocks:", n_distinct(stocks_raw$Stock), "\n")
cat("Date range:", as.character(min(stocks_raw$Date)),
    "to", as.character(max(stocks_raw$Date)), "\n\n")
glimpse(stocks_raw)


# ── 2. Feature Engineering ────────────────────────────────────
stocks <- stocks_raw %>%
  group_by(Stock) %>%
  arrange(Date) %>%
  mutate(
    # Returns
    Daily_Return  = (Close - lag(Close)) / lag(Close),
    Log_Return    = log(Close / lag(Close)),

    # Moving averages
    MA_10  = rollmean(Close, 10,  fill = NA, align = "right"),
    MA_20  = rollmean(Close, 20,  fill = NA, align = "right"),
    MA_50  = rollmean(Close, 50,  fill = NA, align = "right"),

    # Volatility (rolling 20-day std of log returns)
    Volatility_20 = rollapply(Log_Return, 20, sd, fill = NA, align = "right"),

    # Bollinger Bands (20-day, 2 SD)
    BB_Upper = MA_20 + 2 * rollapply(Close, 20, sd, fill = NA, align = "right"),
    BB_Lower = MA_20 - 2 * rollapply(Close, 20, sd, fill = NA, align = "right"),

    # Momentum
    Momentum_10 = Close - lag(Close, 10),

    # RSI (14-day)
    RSI = RSI(Close, n = 14),

    # MACD
    MACD_val = MACD(Close)[, "macd"],
    MACD_sig = MACD(Close)[, "signal"],

    # Trend labels
    Trend = case_when(
      Close > MA_50 & MA_20 > MA_50 ~ "Uptrend",
      Close < MA_50 & MA_20 < MA_50 ~ "Downtrend",
      TRUE ~ "Sideways"
    )
  ) %>%
  ungroup()

cat("\n=== Feature Engineering Done ===\n")
cat("Columns added: Daily_Return, Log_Return, MA_10/20/50,",
    "Volatility_20, BB Bands, Momentum_10, RSI, MACD, Trend\n")


# ── 3. Exploratory Data Analysis ─────────────────────────────

# 3a. Price trajectories
p1 <- ggplot(stocks, aes(x = Date, y = Close, color = Stock)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Closing Prices — All Stocks", x = NULL, y = "Price (USD)") +
  scale_y_continuous(labels = dollar_format()) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

# 3b. Daily returns distribution
p2 <- ggplot(stocks %>% drop_na(Daily_Return),
             aes(x = Daily_Return, fill = Stock)) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity") +
  labs(title = "Daily Returns Distribution", x = "Daily Return", y = "Count") +
  scale_x_continuous(labels = percent_format()) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

# 3c. Cumulative returns
p3 <- stocks %>%
  drop_na(Log_Return) %>%
  group_by(Stock) %>%
  mutate(Cumulative_Return = cumsum(Log_Return)) %>%
  ggplot(aes(x = Date, y = Cumulative_Return, color = Stock)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Cumulative Log Returns", x = NULL, y = "Cumulative Log Return") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

# 3d. Rolling volatility
p4 <- stocks %>%
  drop_na(Volatility_20) %>%
  ggplot(aes(x = Date, y = Volatility_20, color = Stock)) +
  geom_line(linewidth = 0.7, alpha = 0.85) +
  labs(title = "20-Day Rolling Volatility", x = NULL, y = "Std Dev of Log Returns") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

grid.arrange(p1, p2, p3, p4, ncol = 2)


# ── 4. Focus on one stock for modelling (AAPL) ───────────────
aapl <- stocks %>%
  filter(Stock == "AAPL") %>%
  arrange(Date) %>%
  drop_na(Close)

cat("\n=== AAPL Summary Statistics ===\n")
summary(aapl$Close)


# ── 5. Trend Analysis — Linear Regression ────────────────────
aapl <- aapl %>% mutate(Time_Index = row_number())

lm_model <- lm(Close ~ Time_Index, data = aapl)
cat("\n=== Linear Trend (OLS) ===\n")
print(summary(lm_model))

aapl$LM_Fitted <- predict(lm_model)

ggplot(aapl, aes(x = Date)) +
  geom_line(aes(y = Close), color = "#378ADD", linewidth = 0.8, alpha = 0.7) +
  geom_line(aes(y = LM_Fitted), color = "#E24B4A", linewidth = 1, linetype = "dashed") +
  geom_line(aes(y = MA_20), color = "#D4537E", linewidth = 0.9, na.rm = TRUE) +
  geom_line(aes(y = MA_50), color = "#1D9E75", linewidth = 0.9, na.rm = TRUE) +
  labs(title = "AAPL — Price with Linear Trend & Moving Averages",
       subtitle = "Blue = Close | Red dashed = Linear trend | Pink = MA20 | Green = MA50",
       x = NULL, y = "Price (USD)") +
  scale_y_continuous(labels = dollar_format()) +
  theme_minimal(base_size = 12)


# ── 6. Stationarity Test (ADF) ────────────────────────────────
cat("\n=== Augmented Dickey-Fuller Test — AAPL Close ===\n")
adf_level <- adf.test(aapl$Close)
print(adf_level)

cat("\n=== ADF Test — First Differenced Series ===\n")
adf_diff <- adf.test(diff(aapl$Close))
print(adf_diff)

if (adf_diff$p.value < 0.05) {
  cat("Series is stationary after first differencing (I(1) process).\n")
} else {
  cat("Series may need further differencing.\n")
}


# ── 7. STL Decomposition ──────────────────────────────────────
# Convert to weekly time series for meaningful seasonality
aapl_ts <- ts(aapl$Close, frequency = 5)   # 5 trading days/week

stl_fit <- stl(aapl_ts, s.window = "periodic", robust = TRUE)
cat("\n=== STL Decomposition (weekly seasonality) ===\n")
plot(stl_fit, main = "AAPL — STL Decomposition (Trend / Seasonal / Remainder)")


# ── 8. ARIMA Forecasting ──────────────────────────────────────
# Train / test split — last 30 days as holdout
n_test  <- 30
n_train <- nrow(aapl) - n_test

train_ts <- ts(aapl$Close[1:n_train], frequency = 5)
test_ts  <- aapl$Close[(n_train + 1):nrow(aapl)]

cat("\n=== Auto-ARIMA Model Selection ===\n")
arima_model <- auto.arima(train_ts, seasonal = TRUE,
                           stepwise = FALSE, approximation = FALSE)
print(summary(arima_model))

# Forecast 30 days ahead
arima_fc <- forecast(arima_model, h = n_test)

# Evaluate
arima_rmse <- rmse(test_ts, as.numeric(arima_fc$mean))
arima_mae  <- mae(test_ts,  as.numeric(arima_fc$mean))
arima_mape <- mape(test_ts, as.numeric(arima_fc$mean)) * 100

cat(sprintf("\nARIMA — RMSE: %.4f | MAE: %.4f | MAPE: %.2f%%\n",
            arima_rmse, arima_mae, arima_mape))

# Plot ARIMA forecast
autoplot(arima_fc) +
  autolayer(ts(test_ts, start = n_train + 1, frequency = 5),
            series = "Actual", color = "#378ADD") +
  labs(title = "AAPL — ARIMA Forecast vs Actual (30-day holdout)",
       x = "Time (trading days)", y = "Price (USD)") +
  scale_y_continuous(labels = dollar_format()) +
  theme_minimal(base_size = 12)


# ── 9. Exponential Smoothing (ETS) ───────────────────────────
cat("\n=== ETS Model ===\n")
ets_model <- ets(train_ts)
print(summary(ets_model))

ets_fc <- forecast(ets_model, h = n_test)

ets_rmse <- rmse(test_ts, as.numeric(ets_fc$mean))
ets_mae  <- mae(test_ts,  as.numeric(ets_fc$mean))
ets_mape <- mape(test_ts, as.numeric(ets_fc$mean)) * 100

cat(sprintf("ETS    — RMSE: %.4f | MAE: %.4f | MAPE: %.2f%%\n",
            ets_rmse, ets_mae, ets_mape))


# ── 10. Neural Network (NNAR) ─────────────────────────────────
cat("\n=== NNAR (Neural Network AutoRegression) ===\n")
nnar_model <- nnetar(train_ts, repeats = 20)
print(nnar_model)

nnar_fc <- forecast(nnar_model, h = n_test)

nnar_rmse <- rmse(test_ts, as.numeric(nnar_fc$mean))
nnar_mae  <- mae(test_ts,  as.numeric(nnar_fc$mean))
nnar_mape <- mape(test_ts, as.numeric(nnar_fc$mean)) * 100

cat(sprintf("NNAR   — RMSE: %.4f | MAE: %.4f | MAPE: %.2f%%\n",
            nnar_rmse, nnar_mae, nnar_mape))


# ── 11. Model Comparison ──────────────────────────────────────
model_results <- tibble(
  Model = c("ARIMA", "ETS", "NNAR"),
  RMSE  = round(c(arima_rmse, ets_rmse, nnar_rmse), 4),
  MAE   = round(c(arima_mae,  ets_mae,  nnar_mae),  4),
  MAPE  = round(c(arima_mape, ets_mape, nnar_mape), 2)
)

cat("\n=== Model Comparison ===\n")
print(model_results)

best_model <- model_results %>% slice_min(RMSE)
cat(sprintf("\nBest model by RMSE: %s\n", best_model$Model))


# ── 12. Final 30-Day Future Forecast (best model) ────────────
cat("\n=== 30-Day Future Forecast (full series) ===\n")
full_ts     <- ts(aapl$Close, frequency = 5)
best_arima  <- auto.arima(full_ts, stepwise = FALSE, approximation = FALSE)
future_fc   <- forecast(best_arima, h = 30)

future_dates <- seq.Date(max(aapl$Date) + 1, by = "day", length.out = 50)
future_dates <- future_dates[!weekdays(future_dates) %in% c("Saturday","Sunday")]
future_dates <- head(future_dates, 30)

forecast_df <- tibble(
  Date  = future_dates,
  Lower = as.numeric(future_fc$lower[, 2]),
  Mean  = as.numeric(future_fc$mean),
  Upper = as.numeric(future_fc$upper[, 2])
)

# Combine historical + forecast
hist_plot <- aapl %>% tail(60) %>% select(Date, Close)

ggplot() +
  geom_line(data = hist_plot,
            aes(x = Date, y = Close), color = "#378ADD", linewidth = 0.9) +
  geom_ribbon(data = forecast_df,
              aes(x = Date, ymin = Lower, ymax = Upper),
              fill = "#D4537E", alpha = 0.2) +
  geom_line(data = forecast_df,
            aes(x = Date, y = Mean), color = "#D4537E", linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = as.numeric(max(aapl$Date)),
             linetype = "dotted", color = "#888780") +
  annotate("text", x = max(aapl$Date) + 2, y = max(hist_plot$Close),
           label = "Forecast start", size = 3, color = "#888780", hjust = 0) +
  labs(title = "AAPL — 30-Day Price Forecast",
       subtitle = "Pink band = 95% confidence interval",
       x = NULL, y = "Price (USD)") +
  scale_y_continuous(labels = dollar_format()) +
  theme_minimal(base_size = 12)

cat("\n=== Forecast Values ===\n")
print(forecast_df)

cat("\n=== Analysis Complete ===\n")
