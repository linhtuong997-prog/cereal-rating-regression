# =====================================================================
# Section 5.5: Multiple linear regression, assumption checking, prediction
# Dataset: 80 Cereals (Kaggle) - https://www.kaggle.com/datasets/crawford/80-cereals
# Put cereal.csv in the working directory, then run the whole script.
# Packages: car, lmtest   (install.packages(c("car", "lmtest")))
# Figures are saved in figures_5_5/
# =====================================================================
suppressMessages({ library(car); library(lmtest) })
dir.create("figures_5_5", showWarnings = FALSE)

# ---------------------------------------------------------------------
# 5.5.2 Data and initial model
# ---------------------------------------------------------------------
d <- read.csv("cereal.csv", stringsAsFactors = FALSE)
d[d == -1] <- NA                                   # -1 codes a missing value
for (v in c("carbo", "sugars", "potass"))          # median imputation (Section 3)
  d[[v]][is.na(d[[v]])] <- median(d[[v]], na.rm = TRUE)

vars <- c("calories", "protein", "fat", "sodium", "fiber",
          "carbo", "sugars", "potass", "vitamins", "rating")
a <- d[, vars]                                     # 9 nutrition variables + rating
rownames(a) <- rownames(d)
ten <- setNames(d$name, rownames(d))               # product name by row name
cat("n =", nrow(a), "\n")                           # 77

set.seed(69)                                       # 80/20 split
index <- sample(1:nrow(a), size = 0.8 * nrow(a))
train <- a[index, ]                                # 61 observations
test  <- a[-index, ]                               # 16 observations

m1 <- lm(rating ~ calories + protein + fat + sodium + fiber +
           carbo + sugars + potass + vitamins, data = train)
summary(m1)                                        # R2 = 0.9929, adj. R2 = 0.9917, RSE = 1.213

# ---------------------------------------------------------------------
# 5.5.3 Handling multicollinearity (Table 5.5.1)
# ---------------------------------------------------------------------
round(vif(m1), 4)                                  # calories, fiber, carbo, sugars > 10
cor(a$fiber, a$potass)                             # 0.91

m2 <- update(m1, . ~ . - calories)                 # Step 1
round(vif(m2), 4)
m2_nofiber <- update(m2, . ~ . - fiber)            # alternative: drop fiber instead
m3 <- update(m2, . ~ . - potass)                   # Step 2
round(vif(m3), 4)
summary(m3)$coefficients["carbo", ]                # t = -0.146, p = 0.884
m4 <- update(m3, . ~ . - carbo)                    # Step 3: final model
round(vif(m4), 4)

AIC(m1, m2, m2_nofiber, m3, m4)                    # 307.31 (drop fiber) vs 224.21
BIC(m3, m4)
anova(m4, m3)                                      # F test, p = 0.884

# ---------------------------------------------------------------------
# 5.5.4 Final model (Table 5.5.2)
# ---------------------------------------------------------------------
summary(m4)                                        # R2 = 0.9901, adj. R2 = 0.9890, F = 900.9
round(confint(m4), 4)

# ---------------------------------------------------------------------
# 5.5.5 Assumption checking (Table 5.5.3, Figures 5.5.1 - 5.5.3)
# ---------------------------------------------------------------------
diag_table <- function(m, name) {
  n <- nrow(model.frame(m)); p <- length(coef(m))
  cd <- cooks.distance(m); r <- rstandard(m)
  data.frame(model = name,
             shapiro_p = shapiro.test(residuals(m))$p.value,
             bp_p = bptest(m)$p.value,
             max_cook = max(cd), max_cook_name = ten[names(which.max(cd))],
             n_cook_above_4n = sum(cd > 4 / n),
             n_cook_above_1 = sum(cd > 1),
             n_abs_stdres_above_2 = sum(abs(r) > 2))
}
print(rbind(diag_table(m1, "initial (9 vars)"), diag_table(m4, "final (6 vars)")))

cd4 <- sort(cooks.distance(m4), decreasing = TRUE)[1:5]
print(data.frame(name = ten[names(cd4)], cook = cd4))        # top 5 of the final model

# Figure 5.5.1: initial model
png("figures_5_5/fig_5_5_1_initial_model.png", width = 1600, height = 750, res = 150, type = "cairo")
par(mfrow = c(1, 2))
plot(m1, which = 1, caption = "", main = "Residuals vs Fitted (initial model)",
     id.n = 2, labels.id = ten[rownames(train)])
plot(m1, which = 2, caption = "", main = "Normal Q-Q (initial model)",
     id.n = 2, labels.id = ten[rownames(train)])
dev.off()

# Figure 5.5.2a and 5.5.2b: final model
png("figures_5_5/fig_5_5_2a_residuals_vs_fitted.png", width = 1000, height = 750, res = 150, type = "cairo")
plot(m4, which = 1, caption = "", main = "Residuals vs Fitted (final model)",
     id.n = 3, labels.id = ten[rownames(train)])
dev.off()
png("figures_5_5/fig_5_5_2b_qq_residuals.png", width = 1000, height = 750, res = 150, type = "cairo")
plot(m4, which = 2, caption = "", main = "Normal Q-Q (final model)",
     id.n = 3, labels.id = ten[rownames(train)])
dev.off()

# Figure 5.5.3: Cook's distance of the final model
png("figures_5_5/fig_5_5_3_cooks_distance.png", width = 1000, height = 750, res = 150, type = "cairo")
cd <- cooks.distance(m4)
plot(cd, type = "h", lwd = 2, col = ifelse(cd > 4 / nrow(train), "firebrick", "grey40"),
     xlab = "Observation index (training set)", ylab = "Cook's distance",
     main = "Cook's distance (final model)")
abline(h = 4 / nrow(train), lty = 2, col = "blue")
text(length(cd), 4 / nrow(train), "4/n = 0.0656", adj = c(1, -0.5), col = "blue", cex = 0.8)
top <- order(cd, decreasing = TRUE)[1:4]
text(top, cd[top], labels = ten[names(cd)[top]], pos = 4, cex = 0.7)
dev.off()

# Sensitivity: refit without Quaker Oatmeal (row name "58")
m4_wo <- update(m4, data = train[rownames(train) != "58", ])
round(coef(m4_wo), 4)                              # sugars -1.790, fat -3.963

# ---------------------------------------------------------------------
# 5.5.6 Prediction performance on the test set (Table 5.5.4, Figure 5.5.4)
# ---------------------------------------------------------------------
perf <- function(m, dat) {
  p <- predict(m, newdata = dat); y <- dat$rating
  c(RMSE = sqrt(mean((y - p)^2)), MAE = mean(abs(y - p)),
    R2 = 1 - sum((y - p)^2) / sum((y - mean(y))^2))
}
rbind(train = perf(m4, train), test = perf(m4, test))  # test RMSE = 1.5169
c(sd_rating = sd(a$rating), min = min(a$rating), max = max(a$rating))

png("figures_5_5/fig_5_5_4_actual_vs_predicted.png", width = 1000, height = 750, res = 150, type = "cairo")
pr <- predict(m4, newdata = test)
plot(test$rating, pr, pch = 19, col = "steelblue",
     xlab = "Actual rating", ylab = "Predicted rating", main = "Actual vs Predicted (test set)")
abline(0, 1, col = "red", lwd = 2)
dev.off()

# ---------------------------------------------------------------------
# 5.5.7 Prediction for a new cereal (Table 5.5.5)
# ---------------------------------------------------------------------
new <- data.frame(protein = 8, fat = 2.5, sodium = 210, fiber = 2, sugars = 12, vitamins = 25)
predict(m4, newdata = new, interval = "prediction", level = 0.95)   # 35.26 (31.66, 38.87)
predict(m4, newdata = new, interval = "confidence", level = 0.95)   # (32.99, 37.54)

# Leverage of the new cereal vs the training set
X  <- model.matrix(m4)
x0 <- model.matrix(~ protein + fat + sodium + fiber + sugars + vitamins, data = new)
drop(x0 %*% solve(crossprod(X)) %*% t(x0))         # 0.6638
max(hatvalues(m4))                                 # 0.4318
sapply(a[, c("protein", "fat", "sodium", "fiber", "sugars", "vitamins")], range)  # protein max = 6

# Same cereal with protein = 3 g (inside the data range)
predict(m4, newdata = transform(new, protein = 3), interval = "prediction", level = 0.95)  # (23.07, 28.87)

sessionInfo()
