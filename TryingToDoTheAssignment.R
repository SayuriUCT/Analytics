# ------------------------------------------ QUESTION 1 ------------------------------------------
traindat <- read.csv("online_shopping_train.csv")
View(traindat)

install.packages("tinytex")

# Converting y (Revenue) to binary variable from integer
traindat$Revenue <- as.factor(traindat$Revenue)

# Converting independent variables to factors
# Month
traindat$Month <- as.factor(traindat$Month)
# Operating Systems
traindat$OperatingSystems <- as.factor(traindat$OperatingSystems)
# Browser
traindat$Browser <- as.factor(traindat$Browser)
# Vister Type
traindat$VisitorType <- as.factor(traindat$VisitorType)
# Weekend
traindat$Weekend <- as.factor(traindat$Weekend)

# a)
# ______ Logistic Regression ______
# Loading necessary packages
library(kableExtra) 
library(broom) 
library(glmnetUtils)

# Fitting logistic regression model
log_model <- glm(Revenue ~ ., data = traindat, family = binomial)

# Representing results of the model in a table
log_model |>
  tidy() |>
  kable(digits = 2, caption = "Summary of logistic regression model fitted to the Online Shopping dataset") |>
  kable_styling(full_width = F)

# Adding a linear decision boundary to the plot of the data
# Chose variables Exit Rates and Page Values (for example, not sure which one I am supposed to use)
mod2 <- glm(Revenue ~ ExitRates + PageValues, 'binomial', traindat)
coefs_2 <- coef(mod2)
plot(traindat$ExitRates, traindat$PageValues,
     col = ifelse(traindat$Revenue == 1, 'coral', 'springgreen4'),
     pch = ifelse(traindat$Revenue == 1, 3, 1),
     xlab = 'Exit Rates', ylab = 'Page Values')
legend('topright', c('Visit Finalised with a transaction', 'Not finalised'), 
       col = c('coral', 'springgreen4'), 
       pch = c(3, 1))
# Add the decision boundary
abline(-coefs_2[1]/coefs_2[3], -coefs_2[2]/coefs_2[3], col = 'darkslateblue', lwd = 3) 

# 10 fold cross-validation
y <- traindat[, 16]  
x <- traindat[, -16]
lasso_cv <- cv.glmnet(as.matrix(x), y, family = "binomial",
                      alpha = 1, nfolds = 10, type.measure = 'mse', standardize = T)
plot(lasso_cv)
lamdas <- lasso_cv$lambda
lamdabest <- lasso_cv$lambda.min
cat("The lamda value I will use is", round(lamdabest, 4), "since it has the lowest MSE")

# Elastic Net Regularisation
elasticnet <- cva.glmnet(Revenue ~ ., traindat, family = "binomial", alpha = seq(0, 1, 0.1))
plot(elasticnet)
alphas <- elasticnet$alpha 
cv_mses <- sapply(elasticnet$modlist, 
                  function(mod) min(mod$cvm) 
)
best_alpha <- alphas[which.min(cv_mses)]
cat("The alpha value I will use is", best_alpha, "since it has the minimum MSE")
plot(alphas, cv_mses, 'b', lwd = 2, pch = 16, col = 'cyan3', xlab = expression(alpha), ylab = 'CV MSE',
     ylim = c(0.56, 0.6)) 
abline(v = best_alpha, lty = 3, col = 'firebrick')


