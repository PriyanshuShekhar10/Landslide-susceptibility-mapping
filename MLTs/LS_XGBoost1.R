
# 1 Introduction ------------------------------------------------------------


##################################################.
## Project: Prediction mapping using machine learning supervised classification techniques.
## Script purpose:Extreme Gradient Boosting (XGBoost) in CARET package
## Date: 5 Jan. 2020
## Author: Omar AlThuwaynee
##################################################.

# Disclaimer:
#            As with most of my R posts, I've arranged the functions using ideas from other people that are much more clever than me. 
#            I've simply converted these ideas into a useful form in R.
# References

#1#   https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
#2#   https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/


# 2 Data prepration ---------------------------------------------------------
# Go to URL of local folder and select and Copy. (H:\Projects\LS Machine L Udemy\WorkingDIR)

path=readClipboard()
setwd(path)

# With project, already the WD is set
getwd() # for checking
.libPaths("./pm10 library")
#.libPaths()
sessionInfo()
#installed.packages()
#"aish"="Hawaaaa"

# Install packages
install.packages("xgboost")
install.packages("rlang")
install.packages("doSNOW")
install.packages("RStoolbox") 
install.packages("doParallel")
install.packages("Matrix")
install.packages("e1071")

library(xgboost)
library(rgdal)        # spatial data processing
library(raster)       # raster processing
library(plyr)         # data manipulation 
library(dplyr)        # data manipulation 
library(RStoolbox)    # plotting spatial data 
library(RColorBrewer) # color
library(ggplot2)      # plotting
library(sp)           # spatial data
library(caret)        # machine laerning
library(doParallel)   # Parallel processing
library(doSNOW)
library(e1071)


# Import training and testing data ----
list.files( pattern = "csv$", full.names = TRUE)
#  2-1 Training Data -------------------------------------------------

data_train <-  read.csv("./Excel/TRAINING ANN.csv", header = T)
data_train <-(na.omit(data_train))
data_train <-data.frame(data_train)  # to remove the unwelcomed attributes


#data_train$TRAINING <- factor(data_train$TRAINING)
# Dealing with Categorial data (Converting numeric variable into groups in R)
#https://www.r-bloggers.com/from-continuous-to-categorical/

ASPECTr<-cut(data_train$ASPECT, seq(0,361,45), right=FALSE, labels=c("a","b","c","d","e","f","g","h"))
table(ASPECTr) 
class(ASPECTr) # double check if not a factor
ASPECTr <- factor(ASPECTr)

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top

flags = data.frame(Reduce(cbind,lapply(levels(ASPECTr),function(x){(ASPECTr == x)*1})
))
names(flags) = levels(ASPECTr)
data_train = cbind(data_train, flags) # combine the ASPECTS with original data

# Remove the original Aspect data
data_train <- data_train[,-9]


# Landcover setting
LANDCOVERTr<-cut(data_train$LANDCOVER, seq(0,8,1), right=FALSE, labels=c("l0","l1","l2","l3","l4","l5","l6","l7"))
table(LANDCOVERTr) 
class(LANDCOVERTr) # double check if not a factor

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top
flags = data.frame(Reduce(cbind,lapply(levels(LANDCOVERTr),function(x){(LANDCOVERTr == x)*1})
))
names(flags) = levels(LANDCOVERTr)
data_train = cbind(data_train, flags) # combine the landcover with original data

# Remove the original landcover and aspect data
data_train <- data_train[,-6] # to remove landcover
data_train <- data_train[,-2] # to remove Rough
data_train <- data_train[,-22] # to remove l7


# Count the number of 1 and 0 elements with the values of dependent vector
as.data.frame(table(data_train$Training))

# Do Scale the data
# Standarization or Normalization, see this for more info
# https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
# Z-score standardization or [ Min-Max scaling: typical neural network algorithm require data that on a 0-1 scale]
# https://vitalflux.com/data-science-scale-normalize-numeric-data-using-r/

# Original equation: n4 - unitization with zero minimum 
# >> ((x-min)/range))=y
# y: is the scaled value
# x :is the original value
# range: original range
# min:  original minmum value

#  > ((y*range)+min)= x
# Ref (https://stackoverflow.com/questions/15215457/standardize-data-columns-in-r)


maxs <- apply(data_train, 2, max) 
mins <- apply(data_train, 2, min)
scaled_train <- as.data.frame(scale(data_train, center = mins, scale = maxs - mins))
scaled_t <-scaled_train
scaled_t$Training <- ifelse(scaled_t$Training == 1, "yes","no")


# 2-2 Testing Data --------------------------------------------------------


data_test <-  read.csv("./Excel/TESTING ANN.csv", header = T)
data_test <-na.omit(data_test)
data_test <-data.frame(data_test)
str(data_test)
as.data.frame(table(data_test$Testing))


# Fix the categorial factor
ASPECTe<-cut(data_test$ASPECT, seq(0,361,45), right=FALSE, labels=c("a","b","c","d","e","f","g","h"))
table(ASPECTe)


# Dealing with Categorial data
ASPECTe <- factor(ASPECTe)
flagse = data.frame(Reduce(cbind, 
                          lapply(levels(ASPECTe), function(x){(ASPECTe == x)*1})
))
names(flagse) = levels(ASPECTe)
data_test = cbind(data_test, flagse) # combine the ASPECTS with original data
data_test <- data_test[,-8] # remove original Aspect


# Landcover setting

LANDCOVERTe<-cut(data_test$LANDCOVER, seq(0,8,1), right=FALSE, labels=c("l0","l1","l2","l3","l4","l5","l6","l7"))
table(LANDCOVERTe) 
class(LANDCOVERTe) # double check if not a factor

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top


flags = data.frame(Reduce(cbind,lapply(levels(LANDCOVERTe),function(x){(LANDCOVERTe == x)*1})
))
names(flags) = levels(LANDCOVERTe)
data_test = cbind(data_test, flags) # combine the landcover with original data

# Remove the original landcover 
data_test <- data_test[,-5]
data_test <- data_test[,-22]# remove original l7

# Match the columns position for all the input data
#head(data_train,1)
#head(data_test,1)
#colnames(data_test)[3] <- "SlOPE" # Match the columns names

#data_test <- data_testN[,c(1,2,3,4,5,6,7,9,8)] Used to re-arrange the columns
# Scale the data
maxs <- apply(data_test, 2, max) 
mins <- apply(data_test, 2, min)
scaled_test <- as.data.frame(scale(data_test, center = mins, scale = maxs - mins))
scaled_tst <-scaled_test
scaled_tst$Testing <- ifelse(scaled_tst$Testing == 1, "yes","no")

# Create one file contain all data
names(scaled_tst)
scaled_tst$Slides=scaled_tst$Testing
names(scaled_t)
scaled_t$Slides=scaled_t$Training

All_incidents <- merge(scaled_tst[,-1], scaled_t[,-1], all=TRUE) #Full outer join: To keep all rows from both data frames, specify all=TRUE.  https://www.dummies.com/programming/r/how-to-use-the-merge-function-with-data-sets-in-r/
str(All_incidents)
All_incidents <- All_incidents[,c(21,1:20)] # re-order columns

scaled_tst$Slides= NULL  # remove Slide column
scaled_t$Slides=NULL  # remove Slide column


##### to predict which variable would be the best one for splitting the Decision Tree, plot a graph that represents the split for each of the 9 variables, ####

#Creating seperate dataframe for '"LevelsAve" features which is our target.
number.perfect.splits <- apply(X=All_incidents[,c(-1)], MARGIN = 2, FUN = function(col){
  t <- table(All_incidents$Slides,col)
  sum(t == 0)})

# Descending order of perfect splits
order <- order(number.perfect.splits,decreasing = TRUE)
number.perfect.splits <- number.perfect.splits[order]

# Plot graph
par(mar=c(10,2,2,2))
barplot(number.perfect.splits,main="Number of perfect splits vs feature",xlab="",ylab="Feature",las=3,col="wheat") # Slope and SPI are the best classifiers


# 3 Run XGBoost function ------------------------------------------------

#Tunning prameters
myControl <- trainControl(method="repeatedcv", 
                          number=10, 
                          repeats=5,
                          returnResamp='all', 
                          allowParallel=TRUE)

#Parameter for Tree Booster
#In the grid, each algorithm parameter can be specified as a vector of possible values . These vectors combine to define all the possible combinations to try.
# We will follow the defaults proposed by https://xgboost.readthedocs.io/en/latest//parameter.html

tune_grid <- expand.grid(nrounds = 200,           # the max number of iterations INCREASE THE PROCESSING TIME COST
                         max_depth = 6,            # depth of a tree EFFECTIVE OPTIMIZATION
                         eta = 0.3,               # control the learning rate
                         gamma = 0,             # minimum loss reduction required
                         colsample_bytree = 1,  # subsample ratio of columns when constructing each tree
                         min_child_weight = 1,     # minimum sum of instance weight (hessian) needed in a child 
                         subsample = 1)          # subsample ratio of the training instance

# Step 5 modeling
set.seed(849)
names(scaled_t)
fit.xgb_train<- train(Training~., 
                      data=scaled_t,
                      method = "xgbTree",
                      metric= "Accuracy",
                      preProc = c("center", "scale"), 
                      trControl = myControl,
                      tuneGrid = tune_grid,
                      tuneLength = 10)
fit.xgb_train$results
#nrounds max_depth eta gamma colsample_bytree min_child_weight subsample  Accuracy     Kappa AccuracySD    KappaSD
#    200         6 0.3     0                1                1         1 0.8887128 0.6993225 0.02308641 0.06325432

fit.xgb_train$resample$Accuracy
X.xgb = varImp(fit.xgb_train)
plot(X.xgb)


#Confusion Matrix - train data
p1<-predict(fit.xgb_train, scaled_tst[,c(-1)], type = "raw")
confusionMatrix(p1, as.factor(scaled_tst$Testing))  # using more deep tree, the accuracy linearly increases! 


######## Hyperparameter----

tune_grid2 <- expand.grid(nrounds = c(200,210),           # the max number of iterations INCREASE THE PROCESSING TIME COST
                          max_depth = c(6,18,22),            # depth of a tree EFFECTIVE OPTIMIZATION
                          eta = c(0.05,0.3,1),               # control the learning rate
                          gamma = c(0,0.01,0.1),             # minimum loss reduction required
                          colsample_bytree = c(0.75,1),  # subsample ratio of columns when constructing each tree
                          min_child_weight = c(0,1,2),     # minimum sum of instance weight (hessian) needed in a child 
                          subsample = c(0.5,1))            # subsample ratio of the training instance

set.seed(849)
fit.xgb_train2<- train(Training~., 
                       data=scaled_t,
                       method = "xgbTree",
                       metric= "Accuracy",
                       preProc = c("center", "scale"), 
                       trControl = myControl,
                       tuneGrid = tune_grid2,
                       tuneLength = 10)

summaryRes=fit.xgb_train2$results # nrounds was fixed = 210
head(summaryRes)
summary(summaryRes)
head(summaryRes[order(summaryRes$Accuracy, decreasing = TRUE),],n=2)  # sort max to min for first 5 values based on Accuracy

# Plot
pairs(summaryRes[,c(-9:-11)])
# Save it
write.csv(fit.xgb_train2$results,file = "fit.xgb_train_hyper.csv")#, sep = "",row.names = T)

#### Read from saved file
#list.files( pattern = "csv$", full.names = TRUE)
#summaryRes <-  read.csv("./fit.xgb_train_hyper.csv", header = T,stringsAsFactors = FALSE)
#summaryRes=summaryRes[,c(-1,-8,-11:-12)] # nrounds was fixed = 210
#head(summaryRes)
#summary(summaryRes)
#head(summaryRes[order(summaryRes$Accuracy, decreasing = TRUE),],n=6)  # sort max to min for first 5 values based on Accuracy
# Plot
#pairs(summaryRes[-8])
 


### Recommended settings
#nrounds = c(200),           # the max number of iterations INCREASE THE PROCESSING TIME COST
#max_depth = c(6),            # depth of a tree EFFECTIVE OPTIMIZATION
#eta = c(0.05),               # control the learning rate
#gamma = c(0.1),             # minimum loss reduction required
#colsample_bytree = c(1),  # subsample ratio of columns when constructing each tree
#min_child_weight = c(0),     # minimum sum of instance weight (hessian) needed in a child 
#subsample = c(0.5))

# Re-run using recommended settings of expand.grid
tune_grid3 <- expand.grid(nrounds = c(200),           # the max number of iterations INCREASE THE PROCESSING TIME COST
                          max_depth = c(6),            # depth of a tree EFFECTIVE OPTIMIZATION
                          eta = c(0.05),               # control the learning rate
                          gamma = c(0.1),             # minimum loss reduction required
                          colsample_bytree = c(0.75),  # subsample ratio of columns when constructing each tree
                          min_child_weight = c(1),     # minimum sum of instance weight (hessian) needed in a child 
                          subsample = c(1))            # subsample ratio of the training instance

set.seed(849)
fit.xgb_train33<- train(Training~., 
                       data=scaled_t,
                       method = "xgbTree",
                       metric= "Accuracy",
                       preProc = c("center", "scale"), 
                       trControl = myControl,
                       tuneGrid = tune_grid3,
                       tuneLength = 10,
                       importance = TRUE)


fit.xgb_train33$results
# nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample  Accuracy     Kappa AccuracySD    KappaSD
#    200         6 0.05   0.1                1                0       0.5 0.8925124 0.7072363 0.02057746 0.05743894
X.xgb = varImp(fit.xgb_train33)
plot(X.xgb)

#Confusion Matrix - train data
p2_xgb_train33<-predict(fit.xgb_train33, scaled_tst[,c(-1)], type = "raw")
confusionMatrix(p2_xgb_train33, as.factor(scaled_tst$Testing))  # using more deep tree, the accuracy linearly increases! 
#while increase the iterations to 220 double the processing time with slight accuracy improvment!



## Plot ROC curves

# https://stackoverflow.com/questions/46124424/how-can-i-draw-a-roc-curve-for-a-randomforest-model-with-three-classes-in-r
#install.packages("pROC")
library(pROC)

# the model is used to predict the test data. However, you should ask for type="prob" here
predictions1 <- as.data.frame(predict(fit.xgb_train33, scaled_test, type = "prob"))

##  Since you have probabilities, use them to get the most-likely class.
# predict class and then attach test class
predictions1$predict <- names(predictions1)[1:2][apply(predictions1[,1:2], 1, which.max)]
predictions1$observed <- as.factor(scaled_tst$Testing)
head(predictions1)

#    Now, let's see how to plot the ROC curves. For each class, convert the multi-class problem into a binary problem. Also, 
#    call the roc() function specifying 2 arguments: i) observed classes and ii) class probability (instead of predicted class).
# 1 ROC curve, Moderate, Good, UHeal vs non Moderate non Good non UHeal
roc.yes <- roc(ifelse(predictions1$observed=="yes","no-yes","yes"), as.numeric(predictions1$yes))
roc.no <- roc(ifelse(predictions1$observed=="no","no-no", "no"), as.numeric(predictions1$no))

plot(roc.no, col = "green", main="XGBoost best tune prediction ROC plot using testing data", xlim=c(0.44,0.1))
lines(roc.yes, col = "red")
#lines(roc.Good, col = "green")

# calculating the values of AUC for ROC curve
results= c("Yes AUC" = roc.yes$auc) #,"No AUC" = roc.no$auc)
print(results)
legend("topleft",c("AUC = 0.84 "),fill=c("red"),inset = (0.42))

#Important note: In previous course (Prediction using ANN "regression") prediction rate = 0.80 using .

#Train xgbTree model USING aLL dependent data
#We will use the train() function from the of caret package with the "method" parameter "xgbTree" wrapped from the XGBoost package.

set.seed(849)
fit.xgbAll3<- train(Slides~., 
                   data=All_incidents,
                   method = "xgbTree",
                   metric= "Accuracy",
                   preProc = c("center", "scale"), 
                   trControl = myControl,
                   tuneGrid = tune_grid3,
                   tuneLength = 10,
                   importance = TRUE)

X.xgbAll = varImp(fit.xgbAll3)
plot(X.xgbAll, main="varImportance XB All tunned")

# Plot graph
# 1. Open jpeg file
jpeg("varImportance XB All tunned.jpg", width = 1000, height = 700)
# 2. Create the plot
plot(X.xgbAll,main="varImportance All XB" )
# 3. Close the file
dev.off()

####################################################################################################################################################################################################################################
###### http://www.sthda.com/english/articles/40-regression-analysis/166-predict-in-r-model-predictions-and-confidence-intervals/


# 6  Produce prediction map using Raster data ---------------------------


# 6-1 Import and process thematic maps ------------------------------------


#Produce LSM map using Training model results and Raster layers data

# Import Raster
install.packages("raster")
install.packages("rgdal")
library(raster)
library(rgdal)


# load all the data

# Load the Raster data
ELEVATION = raster("./Original layers/DEM.tif")  
SLOPE= raster("./Original layers/SLOPE.tif") 
CURVATURE= raster("./Original layers/CURVATURE.tif") 
TWI= raster("./Original layers/TWI.tif") 
SPI=raster("./Original layers/SPI.tif")
ASPECT=raster("./Original layers/ASPECT.tif")
LANDCOVER=raster("./Original layers/LANDCOVER.tif") 


# check attributes and projection and extent
extent(ELEVATION)
extent(SLOPE)
extent(TWI)
extent(SPI)
extent(ASPECT)
extent(CURVATURE)
extent(LANDCOVER)

# if you have diffrent extent, then try to Resample them using the smallest area
ELEVATION_r <- resample(ELEVATION,LANDCOVER, resample='bilinear') 
SLOPE_r <- resample(SLOPE,LANDCOVER, resample='bilinear') 
TWI_r <- resample(TWI,LANDCOVER, resample='bilinear') 
CURVATURE_r <- resample(CURVATURE,LANDCOVER, resample='bilinear') 
SPI_r <- resample(SPI,LANDCOVER, resample='bilinear') 
ASPECT_r <- resample(ASPECT,LANDCOVER, resample='bilinear') 

extent(ASPECT_r) # check the new extent
extent(LANDCOVER)

# write to a new geotiff file
# Create new folder in WD using manually or in R studio (lower right pan)
writeRaster(ASPECT_r,filename="resampled/ASPECT.tif", format="GTiff", overwrite=TRUE) 
writeRaster(SPI_r,filename="resampled/SPI.tif", format="GTiff", overwrite=TRUE)
writeRaster(CURVATURE_r,filename="resampled/CURVATURE.tif", format="GTiff", overwrite=TRUE)
writeRaster(TWI_r,filename="resampled/TWI.tif", format="GTiff", overwrite=TRUE)
writeRaster(ELEVATION_r,filename="resampled/ELEVATION.tif", format="GTiff", overwrite=TRUE)
writeRaster(SLOPE_r,filename="resampled/SLOPE.tif", format="GTiff", overwrite=TRUE)
writeRaster(LANDCOVER,filename="resampled/LANDCOVER.tif", format="GTiff", overwrite=TRUE)

#Stack_List= stack(ASPECT_r,LS_r)#,pattern = "tif$", full.names = TRUE)
#names(Stack_List)
#Stack_List.df = as.data.frame(Stack_List, xy = TRUE, na.rm = TRUE)
#head(Stack_List.df,1)


## stack multiple raster files
Stack_List= list.files(path = "resampled/",pattern = "tif$", full.names = TRUE)
Rasters=stack(Stack_List)

names(Rasters)


# 6-1-1 Convert rasters to dataframe with Long-Lat -----------------------
#Convert raster to dataframe with Long-Lat
Rasters.df = as.data.frame(Rasters, xy = TRUE, na.rm = TRUE)
head(Rasters.df,1)


# Now:Prediction using imported Rasters

# check the varaibles names to match with training data
#colnames(Rasters.df)[4] <- "ElEVATION"   # change columns names 
#colnames(Rasters.df)[6] <- "SlOPE"   # change columns names 

#head(Rasters.df[,c(-9,-10)],1)
#head(nn.ce$covariate,1)

Rasters.df_N <- Rasters.df[,c(-1,-2)] # remove x, y


# 6-1-2 Dealing with Categorial data --------------------------------------


# Dealing with Categorial data (Converting numeric variable into groups in R)
#https://www.r-bloggers.com/from-continuous-to-categorical/

# ASPECT
ASPECTras<-cut(Rasters.df_N$ASPECT, seq(0,361,45), right=FALSE, labels=c("a","b","c","d","e","f","g","h"))
table(ASPECTras)


# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top

ASPECTras <- factor(ASPECTras)
flagsras = data.frame(Reduce(cbind, 
                             lapply(levels(ASPECTras), function(x){(ASPECTras == x)*1})
))
names(flagsras) = levels(ASPECTras)
Rasters.df_N = cbind(Rasters.df_N, flagsras) # combine the ASPECTS with original data

# Remove the original aspect data
Rasters.df_N<- Rasters.df_N[,-1]
str(Rasters.df_N)


# LANDCOVER

# Dealing with Categorial data (Converting numeric variable into groups in R)
#https://www.r-bloggers.com/from-continuous-to-categorical/
LANDCOVERras<-cut(Rasters.df_N$LANDCOVER, seq(0,8,1), right=FALSE, labels=c("l0","l1","l2","l3","l4","l5","l6","l7"))
table(LANDCOVERras)


# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top

LANDCOVERras <- factor(LANDCOVERras)
flagsras = data.frame(Reduce(cbind, 
                             lapply(levels(LANDCOVERras), function(x){(LANDCOVERras == x)*1})
))
names(flagsras) = levels(LANDCOVERras)
Rasters.df_N = cbind(Rasters.df_N, flagsras) # combine the LANDCOVER with original data

# Remove the original LANDCOVER data
Rasters.df_N<- Rasters.df_N[,-3]
str(Rasters.df_N)


# 6-1-3 Scale the numeric variables --------------------------------------

# Check the relationship between the numeric varaibles, Scale the numeric var first!
maxss <- apply(Rasters.df_N, 2, max) 
minss <- apply(Rasters.df_N, 2, min)
Rasters.df_N_scaled <- as.data.frame(scale(Rasters.df_N, center = minss, scale = maxss - minss)) # we removed the Aspect levels because it might be changed to NA!
colnames(Rasters.df_N_scaled)[colnames(Rasters.df_N_scaled)=="CURVATURE"] <- "CURVE"


# PRODUCE PROBABILITY MAP
p3<-as.data.frame(predict(fit.xgbAll3, Rasters.df_N_scaled, type = "prob"))
summary(p3)
Rasters.df$Levels_yes<-p3$yes
Rasters.df$Levels_no<-p3$no

x<-SpatialPointsDataFrame(as.data.frame(Rasters.df)[, c("x", "y")], data = Rasters.df)
r_ave_yes <- rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_yes")])
proj4string(r_ave_yes)=CRS(projection(ELEVATION))

r_ave_no <- rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_no")])
proj4string(r_ave_no)=CRS(projection(ELEVATION))


# Plot Maps
spplot(r_ave_yes, main="Landslides SM using XGB")
writeRaster(r_ave_yes,filename="Prediction_XGBoostTunned_Landslides SM.tif", format="GTiff", overwrite=TRUE) 

spplot(r_ave_no, main="Non Slide XGB")
writeRaster(r_ave_no,filename="Prediction_XGBoostTunned_Non Slide.tif", format="GTiff", overwrite=TRUE) 


# PRODUCE CLASSIFICATION MAP
#Prediction at grid location
p3<-as.data.frame(predict(fit.xgbAll3, Rasters.df_N_scaled, type = "raw"))
summary(p3)
# Extract predicted levels class
head(Rasters.df, n=2)
Rasters.df$Levels_Slide_No_slide<-p3$`predict(fit.xgbAll3, Rasters.df_N_scaled, type = "raw")`
head(Rasters.df, n=2)

# Import levels ID file 
ID<-read.csv("./Levels_key.csv", header = T)

# Join landuse ID
grid.new<-join(Rasters.df, ID, by="Levels_Slide_No_slide", type="inner") 
# Omit missing values
grid.new.na<-na.omit(grid.new)    
head(grid.new.na, n=2)

#Convert to raster
x<-SpatialPointsDataFrame(as.data.frame(grid.new.na)[, c("x", "y")], data = grid.new.na)
r_ave_Slide_No_slide <- rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Level_ID")])

# coord. ref. : NA 
# Add coord. ref. system by using the original data info (Copy n Paste).
# borrow the projection from Raster data
proj4string(r_ave_Slide_No_slide)=CRS(projection(ELEVATION)) # set it to lat-long

# Export final prediction map as raster TIF ---------------------------
# write to a new geotiff file
writeRaster(r_ave_Slide_No_slide,filename="Classification_Map XGBoost Tunned SLIDE_NO SLIDE.tif", format="GTiff", overwrite=TRUE) 


#Plot Landuse Map:
# Color Palette follow Air index color style
#https://bookdown.org/rdpeng/exdata/plotting-and-color-in-r.html

myPalette <- colorRampPalette(c("light green","red" ))

# Plot Map
LU_ave<-spplot(r_ave_Slide_No_slide,"Level_ID", main="Landslide prediction: XGBoost tunned" , 
               colorkey = list(space="right",tick.number=1,height=1, width=1.5,
                               labels = list(at = seq(1,4.8,length=5),cex=1.0,
                                             lab = c("Yes" ,"No"))),
               col.regions=myPalette,cut=4)
LU_ave
jpeg("Prediction_Map XGBoost_Landslide .jpg", width = 1000, height = 700)
LU_ave
dev.off()

##########DDDDDDDDDDDDOOOOOOOOOOOOOOOONNNNNNNNNNNNNNEEEEEEEEE :) :)






