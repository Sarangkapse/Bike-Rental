rm(list = ls(all = T))
getwd()

rm(list=ls())

#Set the directory
setwd("C:/Users/Asus/Documents/R programming")
getwd()

#Load libraries
#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')


#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

##Read the data
data_bike = read.csv("day.csv", header = T, na.strings = c(" ", "", "NA"))

#########Converting data types ##############
str(data_bike) # Checking the required data types
class(data_bike)
data_bike$season = as.factor(data_bike$season)
data_bike$yr = as.factor(data_bike$yr)
data_bike$mnth = as.factor(data_bike$mnth)
data_bike$holiday = as.factor(data_bike$holiday)
data_bike$weekday = as.factor(data_bike$weekday)
data_bike$workingday = as.factor(data_bike$workingday)
data_bike$weathersit = as.factor(data_bike$weathersit)
data_bike$casual = as.numeric(data_bike$casual)
data_bike$registered = as.numeric(data_bike$registered)
data_bike$cnt = as.numeric(data_bike$cnt)
data_bike$windspeed = as.numeric(data_bike$windspeed)
##################Missing values analysis##############################
#Create a dataframe with missing percentage
missing_val = data.frame(apply(data_bike,2,function(x){sum(is.na(x))}))

#Convert row names into columns
missing_val$Columns = row.names(missing_val)
row.names(missing_val) = NULL

#Rename the variable name
names(missing_val)[1] =  "Missing_percentage"

#Calculate the percentage
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(data_absent)) * 100

#Arrange in descending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]

#Rearrange the column names
missing_val = missing_val[,c(2,1)]


missing_val


##################### No missing values found ###############

#######################################Outlier Analysis############################
# ## BoxPlots - Distribution and Outlier Check
numeric_index_bike = sapply(data_bike,is.numeric) #selecting only numeric

numeric_data_bike = data_bike[,numeric_index_bike]

cnames_bike = colnames(numeric_data_bike)

for (i in 1:length(cnames_bike))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames_bike[i])), data = subset(data_bike))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames_bike[i])+
           ggtitle(paste("Box plot of Bike rent for variable ",cnames_bike[i])))
}


## Plotting plots together
gridExtra::grid.arrange(gn1,ncol=1)
gridExtra::grid.arrange(gn2,ncol=1)
gridExtra::grid.arrange(gn3,ncol=1)
gridExtra::grid.arrange(gn4,ncol=1)
gridExtra::grid.arrange(gn5,ncol=1)
gridExtra::grid.arrange(gn6,ncol=1)
gridExtra::grid.arrange(gn7,ncol=1)
gridExtra::grid.arrange(gn8,ncol=1)


##############Creating box plot for each variable #########
boxplot(data_bike$instant)
boxplot(data_bike$temp)
boxplot(data_bike$atemp)
boxplot(data_bike$hum)
boxplot(data_bike$windspeed)
boxplot(data_bike$casual)
boxplot(data_bike$registered)
boxplot(data_bike$cnt)


############## loop to remove outlier and impute using Knn############
for(i in cnames_bike)
{val = data_bike[,i][data_bike[,i] %in% boxplot.stats(data_bike[,i])$out]
#print(length(val))
data_bike[,i][data_bike[,i] %in% val] = NA
}

data_bike = knnImputation(data_bike, k = 5)

##################################Feature Selection################################################
## Correlation Plot 
corrgram(data_bike[,numeric_index_bike], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

####Drop temp and registered user ######

## Chi-squared Test of Independence
factor_index = sapply(data_bike,is.factor)
factor_data = data_bike[,factor_index]

factor_data_2 = factor_data[, c(1, 8, 3, 4, 5, 6, 7, 2)]

for (i in 1:7)
{
  print(names(factor_data_2)[i])
  print(chisq.test(table(factor_data_2$season,factor_data_2[,i])))
}

############Drop variable "dteday", "weathersit","yr", "holiday", "weekday", "working day"################# 

## Dimension Reduction
data_bike = subset(data_bike, 
                     select = -c(dteday, weathersit, yr, holiday, weekday, workingday, temp, registered ))

str(data_bike)

#Normality check
qqnorm(data_bike$atemp)
hist(data_bike$atemp)

qqnorm(data_bike$hum)
hist(data_bike$hum)

qqnorm(data_bike$windspeed)
hist(data_bike$windspeed)

qqnorm(data_bike$casual)
hist(data_bike$casual) #### data is left skewed

qqnorm(data_bike$cnt)
hist(data_bike$cnt)


numeric_index_norm = sapply(data_bike,is.numeric) #selecting only numeric

numeric_data_norm = data_bike[,numeric_index_norm]

cnames_absent_norm = colnames(numeric_data_norm)

cnames_norm = c("atemp","hum","windspeed","casual","cnt")

for(i in cnames_norm){
  print(i)
  data_bike[,i] = (data_bike[,i] - min(data_bike[,i]))/
    (max(data_bike[,i] - min(data_bike[,i])))
}

df = data_bike

####################################Train and test data ##################
###############Clean the environment
library(DataCombine)
rmExcept("df")

#Divide data into train and test using stratified sampling method
set.seed(1234)
train_index = sample(1:nrow(df), 0.8*nrow(df))
train = df[ train_index,]
test  = df[-train_index,]

#Load Libraries
library(rpart)
library(MASS)

###########################Linear regression
#check Multicollinearity

library(usdm)
vif(df[, -10])
numeric_index = sapply(df,is.numeric) #selecting only numeric

numeric_data = df[,numeric_index]

cnames_df = colnames(numeric_data)
cnames_df = data.frame(cnames_df)

vifcor(numeric_data[, -6], th = 0.9)

#Build regression model on train data
lm_model = lm(cnt ~., data = train)

#Summary of the model
summary(lm_model)

#Predict the values of test data by applying the model on test data
predictions_LR = predict(lm_model , test[,1:8])

#MODEL EVALUATION method
regr.eval(test[,'cnt'] , predictions_LR ,stats = c('mae','rmse','mape','mse'))


################################## Decision Tree Regression ##########################

#########rpart for regression #####################
fit_dt = rpart(cnt~ ., data = df, method = "anova")

#Predict for new test cases
#Predict for new test cases
predictions_DT = predict(fit_dt, test[,-8])

RMSE(predictions_DT, test$cnt)




