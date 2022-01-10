
library(ggplot2)
library(stringr)
library(magrittr)
library(tidyr)
library(dplyr)
library(DescTools)
library(pbmcapply)
library(data.table)
library(Hmisc)
library(stringi)
library(igraph)
library(e1071)
library(VGAM)
library(abind)
library(Rtsne)
library(irlba)
library(stats)
library(graphkernels)
library(kernlab)
library(dclust)
library(dbscan)
library(plyr)
library(MASS)
library(dplyr)
library(ComplexHeatmap)
library(dendsort)
library(patchwork)
library(ggplotify)
library(sp)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

test_data <- fread('../results/LRP/use_data/all_data.csv')
test_data$LRP <- test_data$LRP
##################
##############################################################################################################################
#load ground truth data
protein_data <- read.csv('../data/tcpa_data_051017.csv', check.names = F) %>% 
  pivot_longer(!c(ID,Cancer_Type), names_to = "proteins", values_to = "expression") %>% 
  dplyr::select(-Cancer_Type)


for_correlation <- test_data %>% dplyr::select(ID, y, y_pred) %>%
  unique() %>%
  group_by(ID)

correlation <- ddply(for_correlation, "ID", summarize, "corr" = cor(y, y_pred))
#mediancorrelation <- summary(correlation$corr)[3]
#highcorrelation <- correlation %>% filter(corr>mediancorrelation)

highcorrelation <- correlation %>% filter(corr>0.7)

rcorr(for_correlation$y, for_correlation$y_pred)

ID_numbers_old <- test_data %>% group_by(ID) %>% dplyr::summarize("Cancer_Type" = max(Cancer_Type))
summary(as.factor(ID_numbers_old$Cancer_Type))

test_data <- test_data %>% filter(ID %in% highcorrelation$ID)
######################################
test_data_dir <- test_data %>% dplyr::select(ID, predicting_protein, masked_protein, 'dLRP' = LRP, Cancer_Type)
test_data_trans <- test_data_dir
colnames(test_data_trans) <- c('ID', 'masked_protein', 'predicting_protein', 'tLRP', 'Cancer_Type')
test_data_sym <- left_join(test_data_dir, test_data_trans) %>% dplyr::mutate(LRP = 0.5*(abs(dLRP) + abs(tLRP))) %>%
  filter(predicting_protein > masked_protein)



united_whole_set <- test_data_sym %>% unite('interactions', c("predicting_protein", "masked_protein")) %>%
  dplyr::select(LRP, Cancer_Type, interactions, ID)

#united_whole_set$LRP <- log(1+abs(united_whole_set$LRP*100))
#united_whole_set$LRP <-(united_whole_set$LRP - mean(united_whole_set$LRP))/sd(united_whole_set$LRP)

united_whole_set_wide <- pivot_wider(united_whole_set, names_from=interactions, values_from = LRP)
united_whole_matrix <- as.matrix(united_whole_set_wide[,-c(1,2)])
#write.csv(united_whole_set_wide, '../results/LRP/use_data/tsne_matrix.csv')

is.na(united_whole_matrix) %>% sum()
#####
distances <- dist(united_whole_matrix, method = 'manhattan')
set.seed(0)
whole_tsne_values <- Rtsne(sqrt(distances), dim=2, perplexity = 15, is_distance=T)
######
#whole_tsne_values <- Rtsne(united_whole_matrix, dim=2, perplexity = 15)

set.seed(0)
dbclusters <- whole_tsne_values$Y %>% dbscan(eps = 2.0, minPts = 15) %>% .$cluster %>% as.factor() # 3.7, 15

cluster_data = data.frame(dbclusters, Cancer_Type = united_whole_set_wide$Cancer_Type, ID= united_whole_set_wide$ID, x =whole_tsne_values$Y[,1], y = whole_tsne_values$Y[,2] )

tsne_plot <- data.frame(x = whole_tsne_values$Y[,1], y = whole_tsne_values$Y[,2], Cancer_Type = united_whole_set_wide$Cancer_Type)

#library(gplots)
#xy <- gplots::space(whole_tsne_values$Y[,1], whole_tsne_values$Y[,2], s=1/100, direction="x")
#xy <- gplots::space(xy[[1]], xy[[2]], s=1/40, direction="y")
#tsne_plot <- data.frame(x = xy[1], y = xy[2], Cancer_Type = united_whole_set_wide$Cancer_Type)
#cluster_data = data.frame(dbclusters, Cancer_Type = united_whole_set_wide$Cancer_Type, ID= united_whole_set_wide$ID, x =xy[1], y = xy[2] )

########################
#plot_tsne with ellipses
mytsnecluster <-ggplot(tsne_plot, aes(x=x, y=y)) + 
  geom_point(data = tsne_plot[dbclusters!=0,], aes(x=x, y=y, color=dbclusters[dbclusters!=0])) + 
  geom_point(data = tsne_plot[dbclusters==0,], aes(x=x, y=y), color="black") + 
  stat_ellipse(data = tsne_plot[dbclusters!=0,], aes(color = dbclusters[dbclusters!=0])) + 
  theme(panel.background = element_blank(), 
        axis.line = element_line(colour = "black"))
plot(mytsnecluster)
####################
library(ggforce)
limit = 20
mytsne <- ggplot(tsne_plot) + 
  geom_point(aes(x=x, y=y, color=Cancer_Type)) + 
  geom_mark_ellipse(data = tsne_plot[dbclusters!=0 & as.numeric(dbclusters)<=limit,], aes(x=x, y=y, group = dbclusters[dbclusters!=0 & as.numeric(dbclusters)<=limit], label = dbclusters[dbclusters!=0 & as.numeric(dbclusters)<=limit]), 
                    label.buffer = unit(0.1, 'mm'), con.cap = 0.01, con.type = "straight", label.fontsize = 30, label.fontface = 'plain') +
  #geom_label(aes(x=x, y=y, label=Cancer_Type))+
  labs(col="Cancer")+
  theme(panel.background = element_blank(), 
        axis.line = element_line(colour = "black"),
        legend.text = element_text(size=20),
        legend.title = element_text(size=20),
        axis.text = element_text(size=15), 
        axis.title = element_text(size=20))+
  guides(color = guide_legend(override.aes = list(size = 4)))

png(paste('./figures/interaction_tsne_numbered', '.png', sep = ""), width = 1400, height = 1400, res = 120)
par(mar=c(10,10,10,10))
plot(mytsne)
dev.off()

#png(paste('./figures/interaction_tsne_numbered', '.png', sep = ""), width = 2800, height = 2800, res = 120)
#par(mar=c(10,10,10,10))
#plot(mytsne)
#dev.off()




################
#find largest clusters
summary_clusters <- summary(dbclusters[dbclusters!=0]) %>% sort(decreasing = T)
summary_clusters[1:8]
summary_clusters
#################
# generate median position
mean_frame <- aggregate(test_data$LRP, by = list(masked_protein = test_data$masked_protein, predicting_protein = test_data$predicting_protein), median)
mean_quantile <- quantile(mean_frame$x, 0.5)
mean_frame$x[mean_frame$x<mean_quantile] <- 0
mean_frame_wide <- mean_frame %>% pivot_wider(names_from = masked_protein, values_from = x)
mean_matrix <- mean_frame_wide[,-1] %>% as.matrix()
rownames(mean_matrix) <- mean_frame_wide$predicting_protein
mean_graph <- graph_from_adjacency_matrix(mean_matrix, mode = "directed", weighted = T)
E(mean_graph)[E(mean_graph)$weight>0]$color <- "red"
E(mean_graph)[E(mean_graph)$weight<0]$color <- "blue"
E(mean_graph)$weight = 0.01*E(mean_graph)$weight
set.seed(0)
mean_positions = layout_nicely(mean_graph)
plot(mean_graph,  layout = mean_positions, vertex.label=NA, vertex.size = 2, vertex.color = "black", edge.width = 3, vertex.label.dist=1, 
     vertex.label.cex = 0.7, edge.arrow.width=0.4, edge.arrow.size=0.1, rescale = T)

################
seedfunction <- function(clusternumber){
  seed=0
  if (clusternumber==5) {seed = 0}
  if (clusternumber==8) {seed = 0}
  print(seed)
  seed
}
cutofffunction <- function(clusternumber){
  #for visibility
  cutoff=0.9997
  if (clusternumber==2) {cutoff = 0.999}
  #if (clusternumber==6) {cutoff = 0.9998}
  #if (clusternumber==8) {cutoff = 0.999}
  cutoff
}

dir.create('./figures/temp/')
nclusters <- dbclusters %>% unique %>% length()-1
for (current_cluster in seq(nclusters)){

current_cluster_position <- cluster_data %>% filter(dbclusters==current_cluster) #CAVE: cluster_data is not yet spaced, only applies to t-SNE
current_cluster_data <- test_data %>% filter(ID %in% current_cluster_position$ID)
####
#symmetrize current cluster data for average network
current_cluster_data_dir <- current_cluster_data %>% dplyr::select(predicting_protein, masked_protein, LRP, ID)
current_cluster_data_trans <- current_cluster_data_dir
colnames(current_cluster_data_trans)[1] <- "masked_protein"
colnames(current_cluster_data_trans)[2] <- "predicting_protein"
colnames(current_cluster_data_trans)[3] <- "tLRP"

current_cluster_data_sym <- left_join(current_cluster_data_dir, current_cluster_data_trans) %>%
  mutate("LRP_sym" = 0.5*(abs(LRP)+abs(tLRP)))#%>%
#  dplyr::filter(predicting_protein>masked_protein)


####
average_frame <- current_cluster_data_sym %>% group_by(predicting_protein, masked_protein) %>%
  dplyr::summarize("meanLRP"= median(LRP_sym)) %>% 
  pivot_wider(names_from = masked_protein, values_from = meanLRP) 

average_matrix <- average_frame[,-1] %>% as.matrix()
rownames(average_matrix) <- average_frame$predicting_protein
average_matrix_ordered <- average_matrix[str_order(rownames(average_matrix)), str_order(colnames(average_matrix))]
cutoff_param <- cutofffunction(current_cluster) 
#cutoff_ID <- average_matrix_ordered %>% abs() %>% quantile(cutoff_param)
####
forthresh <- current_cluster_data_sym %>% group_by(predicting_protein, masked_protein) %>%
  dplyr::summarize("meanLRP"= median(LRP_sym)) %>%
  arrange(desc(meanLRP))

for (thresh in forthresh$meanLRP) {
  high_edges <- forthresh %>% filter(meanLRP>=thresh)
  if (c(high_edges$predicting_protein, high_edges$masked_protein) %>% unique() %>% length() >=7) {
    cutoff_ID <- thresh
    break
  }
}
####
average_matrix_ordered_select <- ifelse(abs(average_matrix_ordered) >= cutoff_ID, average_matrix_ordered,0) 

average_network <- graph_from_adjacency_matrix(average_matrix_ordered_select, weighted=T, mode="directed")
positions <- cbind(V(average_network),mean_positions)
#reduced_positions <- positions[degree(average_network) >0,]
vertices2delete <- positions[degree(average_network) ==0,1] %>% as.vector()
average_network2 <- delete_vertices(average_network, vertices2delete)
E(average_network2)[E(average_network2)$weight>0]$color <- "black"

thresh <- degree(average_network2) %>% sort(decreasing = T) %>% .[10] %>% max(c(.,1), na.rm=T) 
selected_names <- V(average_network2)$name[rev(order(degree(average_network2)))] %>% .[1:min(20, length(.))]
set.seed(seedfunction(current_cluster))
l = layout_in_circle(average_network2, sample(seq(length(V(average_network2)))))
png(paste('./figures/temp/average_ID_', current_cluster, '.png', sep=""), width=2000, height = 1700, res=200)
#par(mar=c(4,11,5,8))
par(mar=c(0,11,0,8))
#qgraph(as_edgelist(average_network2),edge.labels=T)
#plot(average_network,  layout = mean_positions, vertex.label= ifelse(degree(average_network)>=thresh,V(average_network)$name, NA), vertex.size = 0.01, vertex.color = NA, 
#     edge.width = 4.0, vertex.label.dist=1.0,
#     vertex.label.cex = 4.5, edge.arrow.width=0.1, edge.arrow.size=0.1, rescale = T)
plot(average_network2, layout = l, edge.width = 3.0, vertex.color = "white", vertex.size = 0, vertex.label.cex = 2.0, edge.arrow.size=0,
     vertex.label= ifelse(V(average_network2)$name %in% selected_names,V(average_network2)$name, NA), vertex.label.family = "Helvetica")
text(x=-1.7, y=1.2, label=as.character(current_cluster), cex=5) #x=-1.3, y=1.2, cex=5
dev.off()

for (current_ID in current_cluster_position$ID) {
  print(current_ID)
  current_network_position <- current_cluster_position %>% filter(ID==current_ID)
  current_network <- current_cluster_data %>% filter(ID==current_ID)
        
  
  IDframe <- current_network %>% 
    dplyr::select(predicting_protein, masked_protein, LRP ) %>%
    pivot_wider(names_from = "masked_protein", values_from = "LRP")
  
  IDmatrix <- IDframe[,-1] %>% as.matrix()
  rownames(IDmatrix) <- IDframe$predicting_protein
  IDmatrix_ordered <- IDmatrix[str_order(rownames(IDmatrix)), str_order(colnames(IDmatrix))]
  
  cutoff_ID <- IDmatrix_ordered %>% abs() %>% quantile(0.999)
  IDmatrix_ordered_select <- ifelse(abs(IDmatrix_ordered) >= cutoff_ID, IDmatrix_ordered,0) 
  IDgraph <- graph_from_adjacency_matrix(IDmatrix_ordered_select, mode = "directed", weighted=T)
  
  E(IDgraph)[E(IDgraph)$weight>0]$color <- "red"
  E(IDgraph)[E(IDgraph)$weight<0]$color <- "blue"
  
  current_positions <- data.frame(x = mean_positions[,1] + 10+current_network_position$x, y = mean_positions[,2] + 10+current_network_position$y) %>% as.matrix()
  
  png(paste('./figures/temp/', current_ID, '.png', sep=""), width=500, height = 500)
  par(bg=NA)
  par(oma = c(0,0,0,0), mar=c(0,0,0,0))
  plot(IDgraph,  layout = mean_positions, vertex.label=NA, vertex.size = 0.01, vertex.color = NA, edge.width = 4.0, vertex.label.dist=1, 
       vertex.label.cex = 0.7, edge.arrow.width=0.1, edge.arrow.size=0.1, rescale = T)
  dev.off()
}
#########################
library(png)
scale_it <- function(x,low,high) {
  x_scaled <- x/(max(x)-min(x))*(high-low)

  x_end <- x_scaled-min(x_scaled)+low

}

image_size = 0.14 # 0.15
current_position_normalized <- current_cluster_position %>% mutate("x" = scale_it(x,0.5*image_size,1-0.5*image_size), "y" = scale_it(y,0.5*image_size,1-0.5*image_size))

png(paste('./figures/temp/','cluster_',current_cluster ,'.png', sep=""), width = 2000, height = 2000, res=200)
par(oma = c(0,0,0,0), mar=c(0,0,0,0)) #delete this?! was added 3.5.2021
plot.new()
for (current_ID in current_cluster_position$ID) {
  print(current_ID)
current_image <- readPNG(paste('./figures/temp/', current_ID, '.png', sep=""))
#current_image[1:500, c(1:2,499:500),] <- 0
#current_image[c(1:2,499:500), 1:500, ] <- 0
current_network_position <- current_position_normalized %>% filter(ID==current_ID)
x = current_network_position$x
y = current_network_position$y
rasterImage(current_image, x-0.5*image_size,y-0.5*image_size,x+0.5*image_size,y+0.5*image_size)
}
text(x=0.0, y=0.95, label=as.character(current_cluster), cex=5)
dev.off()
}

###############################
#########################################################
#test interactions between c-MET caspase8 parpcleaved Snail ercc1
library(ggExtra)
selprots <- c("CMET", "CASPASE8", "PARPCLEAVED", "SNAIL", "ERCC1")

quintum <-test_data %>% ungroup()%>%
  dplyr::filter(predicting_protein %in% selprots, masked_protein %in% selprots, predicting_protein!=masked_protein) %>%
  dplyr::select(ID, Cancer_Type, predicting_protein, masked_protein, LRP) 

t_quintum <- quintum
colnames(t_quintum) <- c("ID", "Cancer_Type", "masked_protein", "predicting_protein", "tLRP")

symmetric_quintum <- left_join(quintum, t_quintum) %>% mutate("new_LRP" = 0.5*(abs(LRP)+abs(tLRP))) %>% filter(predicting_protein < masked_protein)
library('scales')
marginplots <- function(selCancer_Type) { 
  quintum_raw <-  symmetric_quintum %>% 
    dplyr::select(ID, Cancer_Type, predicting_protein, masked_protein, new_LRP) %>% 
    unite("interaction",predicting_protein:masked_protein) %>% mutate(logLRP = log(1+new_LRP)) %>% filter(Cancer_Type==selCancer_Type)
  
  p <- ggplot(quintum_raw, aes(x= as.numeric(as.factor(interaction)), y =logLRP)) + 
    geom_point(alpha = 0.0) + 
    geom_line(aes(color = ID, group = ID), show.legend = F) +
    ggtitle(selCancer_Type) +
    ylim(0,0.7)+ 
    xlab('Interactions')+
    theme_bw()+
    scale_x_continuous(breaks = pretty_breaks())
    theme(plot.title = element_text(size=15, margin = margin(10,0, 0,0)),
          axis.title.x = element_blank(),
          plot.margin = margin(0,0,0,0))
  p2 <- ggMarginal(p, type="histogram", margins = "y")
  
  p2
}


#################################
Cancer_Types <- symmetric_quintum$Cancer_Type %>% unique()%>% unlist()

marplots <- lapply(Cancer_Types, marginplots)
library(ggpubr)
png("./figures/quintum.png",width = 1000, height = 1000)
ggarrange(plotlist=marplots)
dev.off()

##correlation
quintum_raw <-  symmetric_quintum %>% 
  unite("interaction",predicting_protein:masked_protein) %>% 
  dplyr::select(ID, interaction, new_LRP) %>%
  #mutate("interaction" = as.factor(interaction)) %>%
  pivot_wider(names_from=interaction, values_from = new_LRP)

correlation <- cor(quintum_raw[,-1], method = "pearson")
min(correlation)
max(correlation)
sort(correlation %>% unique())

