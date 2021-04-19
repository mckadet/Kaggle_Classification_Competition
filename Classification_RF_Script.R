# McKade Thomas
# Stat495R Kaggle
# Classification Competition


# Libraries needed
library(vroom)
library(caret)
library(tidyverse)
library(ranger)


ghost.train <- read.csv('train.csv')
ghost.test <- read.csv('test.csv')

ghost.train$type <- as.factor(ghost.train$type)
ghost.train$color <- as.factor(ghost.train$color)

ghost.test$color <- as.factor(ghost.test$color)

all_ids <- c(ghost.train$id,ghost.test$id)

## Random Forest
tgrid <- expand.grid(
  mtry = 5,
  splitrule = c("gini"),
  min.node.size = 10
)

ghost.forest <- train(form=type~bone_length+rotting_flesh +
                      hair_length+has_soul+color+has_soul*hair_length,
                     data=ghost.train, 
                     num.trees=500,
                     method="ranger",
                     trControl=trainControl(
                       method="repeatedcv",
                       number=10,
                       repeats=3,
                       verboseIter = TRUE),
                     tuneGrid = tgrid)

print(ghost.forest)

ghost.forest <- train(form=type~bone_length+rotting_flesh +
                        hair_length+has_soul+color+has_soul*hair_length,
                      data=ghost.train,
                      trControl=trainControl(
                        method="repeatedcv",
                        number=10,
                        repeats=3,
                        verboseIter = TRUE))

rng <- ranger(formula=type~bone_length+rotting_flesh +
                hair_length+has_soul+color,
              data = ghost.train,
              num.trees=500,mtry=5,splitrule="gini",min.node.size=10,probability=TRUE)
ghostPreds <- predict(rng, data=bind_rows(ghost.train,ghost.test ))

ghostPreds <- data.frame(predict(ghost.forest, newdata=ghost.test,type='prob')) %>% mutate(ID = ghost.test %>% pull(id))

max(ghostPreds$predictions)

all_preds <- data.frame(ghostPreds$predictions)
all_preds$ID <- 0:899

all_preds_new <- all_preds[, c("ID", "Ghost", "Ghoul", "Goblin")]


names(all_preds_new) <- c("ID", "PrGhoul_rf", "PrGob_rf", "PrGhost_rf")

write.csv(all_preds_new,file="./classification_submission_rf.csv",row.names=FALSE)


# Create Submission
submit_df <- data.frame(predict(ghost.forest, newdata=ghost.test))
submission_rf <- data.frame(id=ghost.test %>% pull(id),
                          type=submit_df %>% pull(predict.ghost.forest..newdata...ghost.test.))

write.csv(submission_rf, file="./Class_Submit_rf1.csv",row.names=FALSE)


# Create Submission 2 - Stacked Model
train <- vroom('train.csv')
test <- vroom('test.csv')

ghost <- bind_rows(train = train, test = test, .id = "Set")
ghost$type <- as.factor(ghost$type)
ghost$color <- as.factor(ghost$color)

nn_probs <- vroom('nn_Probs_65acc.csv')
gbm <- vroom('probs_gbm.csv')
knn <- vroom('Probs_KNN.csv')
svm <- vroom('probs_svm.csv')
xgbtree <- vroom('xgbTree_probs.csv')
rf <- vroom('classification_submission_rf.csv')
log_reg <- vroom('LogRegPreds.csv')


names(nn_probs)[4] <- "id"
names(gbm)[1] <- "id"
names(knn)[1] <- 'id'
names(svm)[1] <- 'id'
names(xgbtree)[1] <- 'id'
names(rf)[1] <- 'id'
names(log_reg)[1] <- 'id'

nn_probs <- nn_probs[c(4,1,2,3)]

all_ghost<- nn_probs %>% left_join(gbm, by = "id") %>% left_join(knn, by = "id") %>%
  left_join(svm, by = "id") %>% left_join(xgbtree, by = "id") %>%
  left_join(rf, by = "id") %>% left_join(log_reg, by = "id")


ghost_data <- preProcess(all_ghost, method = "pca")
pca_ghost_pred <- predict(ghost_data, newdata = all_ghost)

joint_ghost <- cbind(pca_ghost_pred, ghost)


ghost_new <- joint_ghost %>% select(-bone_length, -rotting_flesh, -hair_length, -has_soul, -color)

xgGrid <- expand.grid(nrounds = 250, 
                      max_depth = 4, 
                      eta = 1,
                      gamma = 0, 
                      colsample_bytree = 1,
                      min_child_weight = 1, 
                      subsample = 1)

ghost_new$type <- as.factor(ghost_new$type)
xgGhost <- train(form = type~. -id, 
                 data = ghost_new %>% filter(Set == "train") %>% select(-Set) , 
                 method = "xgbTree", 
                 tuneGrid = xgGrid,
                 trControl = trainControl(
                   method = "repeatedcv", 
                   number = 5, 
                   repeats = 1)) 

print(xgGhost)

xgGhost$results

xgPreds <- predict(xgGhost, newdata = joint_ghost %>% filter(Set == "test"))

submission <- data.frame(id = ghost %>% filter(Set == "test") %>% pull(id), type = xgPreds)

write.csv(submission, file = "./stack_submission.csv", row.names= FALSE)

# Create Submission 3 - nn
nn <- read.csv('xgbTree_probs.csv')
new_ids <- read.csv('Class_Submit_rf1.csv')

sub_nn <- nn[,2:4]
colnames(sub_nn) <- c("Goblin","Ghost","Ghoul")

type <- colnames(sub_nn)[apply(sub_nn,1,which.max)]

submission3 <- cbind(nn$ID,type)
colnames(submission3) <- c("id","type")

train_submit <- as.data.frame(submission3[1:529,])
train_submit$id <- new_ids$id



write.csv(train_submit, file = "./xgb_submission.csv", row.names= FALSE)
