
# reading data 

df = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv')

data = read.csv('https://s3.amazonaws.com/dspython.dezyre.com/notebook_files/02-06-18-12-47-40/Sales_Transactions_Dataset_Weekly.csv',header = T,sep = ',')

library(cluster);library(cclust);library(fastcluster)
library(caret);library(mlbench)

####Data Pre-processing
par(mfrow=c(2,3))
apply(df[,c(-1)],2,boxplot)

#computing quantiles
quant<-function(x){quantile(x,probs=c(0.95,0.90,0.99))}
out1<-sapply(df[,c(-1)],quant)

# normalized data
r_df = df[,56:107]

#raw data set
r_df1 = df[,0:51]

#Checking outliers after removal
apply(r_df,2,boxplot)

########################################################################

library(Rcmdr)

sumsq<-NULL
#Method 1
par(mfrow=c(1,2))
for (i in 1:15) sumsq[i] <- sum(KMeans(r_df,
                                       centers=i,
                                       iter.max=500, 
                                       num.seeds=50)$withinss)

plot(1:15,sumsq,type="b", xlab="Number of Clusters", 
     ylab="Within groups sum of squares",main="Screeplot using Rcmdr")

#Method 2
for (i in 1:15) sumsq[i] <- sum(kmeans(r_df,
                                       centers=i,
                                       iter.max=500, 
                                       algorithm = "Forgy")$withinss)

plot(1:15,sumsq,type="b", xlab="Number of Clusters", 
     ylab="Within groups sum of squares",main="Screeplot using Stats")

################################################################################
#clustering 
#Kmeans Clustering
library(cluster);library(cclust)
set.seed(121)
km<-kmeans(r_df,
           centers=4,
           nstart=17,
           iter.max=500, 
           algorithm = "Forgy",
           trace = T)


#checking results
summary(km)
km$centers
km$withinss

#attaching cluster information
Cluster<-cbind(r_df,Membership=km$cluster)
aggregate(Cluster[,-53],list(Cluster[,53]),mean)

#plotting cluster info
clusplot(Cluster, 
         km$cluster, 
         cex=0.9,
         color=TRUE, 
         shade=TRUE,
         labels=4, 
         lines=0)

############################################################################3

#Predicting new data for KMeans
predict.kmeans <- function(km, r_df)
{k <- nrow(km$centers)
n <- nrow(r_df)
d <- as.matrix(dist(rbind(km$centers, r_df)))[-(1:k),1:k]
out <- apply(d, 1, which.min)
return(out)}

#predicting cluster membership
Cluster$Predicted<-predict.kmeans(km,r_df)
table(Cluster$Membership,Cluster$Predicted)

#writing the result to a file  
write.csv(Cluster,"predout1.csv")

# cluster model deployment

#############################################################################

# deplymnet using an API (restful API)
# using PMML scipt
# calling custom functions from external applications such as T-SQL

#pmml code
library(pmml);
library(XML);
pmml(km)

#Hierarchical Clustering-agglomorative method
dev.off()
hfit<-hclust(dist(r_df,method = "euclidean"),method="ward.D2")
par(mfrow=c(1,2))
plot(hfit,hang=-0.005,cex=0.7)

hfit<-hclust(dist(r_df,method = "manhattan"),method="mcquitty")
plot(hfit,hang=-0.005,cex=0.7)

############################################################################

hfit<-hclust(dist(r_df,method = "minkowski"),method="ward.D2")
plot(hfit,hang=-0.005,cex=0.7)

hfit<-hclust(dist(r_df,method = "canberra"),method="ward.D2")
plot(hfit,hang=-0.005,cex=0.7)

#method	

###################################################################################

#attaching cluster information
summary(hfit)

#Hierarchical Clustering-divisive method
dfit<-diana(r_df,
            diss=F,
            metric = "euclidean",
            stand=T,
            keep.data = F)
summary(dfit)
plot(dfit)

####################################################################################



####################################################################################

#cutting the tree into groups
g_hfit<-cutree(hfit,k=4)
table(g_hfit)
plot(hfit)
rect.hclust(hfit,k=4,border = "blue")

########################################################################################

# next step is to find out the cluster goodness using silhoutte score on
# the raw dataset, instead of the normalized dataset

pr4 <- pam(r_df, 4)
str(si <- silhouette(pr4))
plot(si)
plot(si, col = c("red", "green", "blue", "purple"))# with cluster-wise coloring

ar <- agnes(r_df)
si3 <- silhouette(cutree(ar, k = 2), # k = 4 gave the same as pam() above
                  daisy(r_df))
plot(si3, nmax = 80, cex.names = 0.5)

# Model based clustering and how it works.

##########################################################################################sanket 

######model based clustering##################
library(mclust)
clus <- Mclust(r_df)
summary(clus)

# Plotting the BIC values:
plot(clus, data=r_df, what="BIC")

# The clustering vector:
clus_vec <- clus$classification
clus_vec

clust <- lapply(1:3, function(nc) row.names(r_df)[clus_vec==nc])  
clust   # printing the clusters

# This gives the probabilities of belonging to each cluster 
#for every object:

round(clus$z,2)
summary(clus, parameters = T)

#self organizing maps
library(kohonen)
som_grid <- somgrid(xdim = 20, ydim=20, topo="hexagonal")

som_model <- som(as.matrix(r_df))

plot(som_model, type="changes",col="blue")
plot(som_model, type="count")

plot(som_model, type="dist.neighbours")
plot(som_model, type="codes")

##############################################
install.packages("factoextra")
install.packages("cluster")
install.packages("magrittr")

library("cluster")
library("factoextra")
library("magrittr")

res.dist <- get_dist(r_df, stand = TRUE, method = "pearson")
fviz_dist(res.dist, 
          gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

############################

library("factoextra")
fviz_nbclust(r_df, kmeans, method = "gap_stat")
fviz_nbclust(r_df, kmeans, method = "silhouette")
fviz_nbclust(r_df, kmeans, method = "wss")

set.seed(123)
km.res <- kmeans(r_df, 6, nstart = 25)
# Visualize
library("factoextra")
fviz_cluster(km.res, data = r_df,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal())

# Compute PAM
library("cluster")
pam.res <- pam(r_df, 6)
# Visualize
fviz_cluster(pam.res)

##########################################################################

#####Hierarchical clustering######
# Compute hierarchical clustering
res.hc <- r_df %>%
  scale() %>%                    # Scale the data
  dist(method = "euclidean") %>% # Compute dissimilarity matrix
  hclust(method = "ward.D2")     # Compute hierachical clustering
# Visualize using factoextra
# Cut in 4 groups and color by groups
fviz_dend(res.hc, k = 6, # Cut in four groups
          cex = 0.5, # label size
          k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
          color_labels_by_k = TRUE, # color labels by groups
          rect = TRUE # Add rectangle around groups
)

###############################################################################