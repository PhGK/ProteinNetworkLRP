library(ggplot2)
library(stringr)
library(magrittr)
library(tidyr)
library(dplyr)
library(DescTools)
library(gplots)
library(pbmcapply)
library(circlize)
library(data.table)
library(Hmisc)
library(stringi)
library(igraph)
library(e1071)
library(VGAM)
library(abind)
library(graphlayouts)
library(ComplexHeatmap)
library(plyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

####################################
#test_data for correlation
protein_data_dep <- fread('../data/artificial_homogeneous.csv')[-1,-1]

colnames(protein_data_dep) <- as.character(seq(32))
correlation_matrix <- cor(protein_data_dep)
Heatmap(correlation_matrix, cluster_rows = F, cluster_columns = F)
correlation_frame_dep <- correlation_matrix %>% as.data.frame() %>%
  cbind("predicting_protein" = as.numeric(rownames(correlation_matrix),.)) %>%
  pivot_longer(!predicting_protein, names_to = "masked_protein", values_to = "correlation") %>%
  mutate("masked_protein" = str_extract(masked_protein, "[^X](.*)")) %>%
  mutate("absCORR" = abs(correlation), "corr" = "correlated")


correlation_frame <- correlation_frame_dep
correlation_frame$predicting_protein <- as.numeric(correlation_frame$predicting_protein)
correlation_frame$masked_protein <- as.numeric(correlation_frame$masked_protein)
############################
#test_data for GENIE3
library(GENIE3)
genie_data <- protein_data_dep %>% as.matrix() %>% t()
colnames(genie_data) <- as.numeric(as.character(colnames(genie_data)))
rownames(genie_data) <- as.numeric(as.character(rownames(genie_data)))
genie_result <- GENIE3(genie_data, verbose=T, nCores=7)
genie_result <- genie_result[order(as.numeric(rownames(genie_result))), order(as.numeric(colnames(genie_result)))]
genie_result_sym <- (genie_result + t(genie_result))
Heatmap(genie_result_sym, cluster_rows = F, cluster_columns = F)

GENIE_frame <- genie_result_sym %>% as.data.frame() %>%
  cbind("predicting_protein" = as.numeric(rownames(genie_result),.)) %>%
  pivot_longer(!predicting_protein, names_to = "masked_protein", values_to = "genie3") %>%
  mutate("masked_protein" = str_extract(masked_protein, "[^X](.*)"))
GENIE_frame$genie3 %>% class()
GENIE_frame$predicting_protein <- as.numeric(GENIE_frame$predicting_protein)
GENIE_frame$masked_protein <- as.numeric(GENIE_frame$masked_protein)
###########################
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

RAWPATH = '../results/artificial/homogeneous/raw_data/'
filenames <- list.files(RAWPATH)

test_data <-rbindlist(lapply(filenames, function(filename) fread(paste0(RAWPATH, filename)))) %>%
  dplyr::select(-V1) 

test_data$LRP = abs(test_data$LRP)

#################################################
test_data_dir <- test_data %>% dplyr::select(LRP, predicting_protein, masked_protein, sample_name) 
test_data_trans <- test_data_dir
colnames(test_data_trans) <- c("tLRP", "masked_protein", "predicting_protein","sample_name")
test_data_sym <- left_join(test_data_dir, test_data_trans) %>%
  dplyr::mutate("LRP" = 0.5*(tLRP+LRP))

mean_test_data <- test_data_sym %>% 
  group_by(predicting_protein, masked_protein) %>%
  dplyr::summarize("meanLRP" = median(LRP)) %>%
  mutate("logLRP" = log(1+meanLRP)) %>% 
  mutate("interaction" = ifelse(predicting_protein %/%8 == masked_protein%/% 8,1,0)) %>%
  mutate(predicting_protein = predicting_protein+1, "masked_protein" = masked_protein+1) %>% 
  left_join(correlation_frame)%>%
  left_join(GENIE_frame)

mean_test_data$absCORR[mean_test_data$predicting_protein==mean_test_data$masked_protein] <- 0
mean_test_data$interaction[mean_test_data$predicting_protein==mean_test_data$masked_protein] <- 0

mean_test_data_correlated <- mean_test_data 

plot_theme <-   theme(
                      legend.position = "bottom",
                      legend.margin = margin(0,0,0,0),
                      legend.box.margin = margin(-25,-30,0,0),
                      legend.title = element_blank(),
                      legend.text = element_text(size=15),
                      plot.title = element_text(size = 30, face = 'plain'),
                      axis.title = element_blank(),
                      axis.line = element_blank(),
                      axis.text = element_blank(),
                      axis.ticks = element_blank()
                      )
plot_theme_adj <-   theme(plot.title = element_text(size = 20),
                          legend.position = "bottom",
                          legend.margin = margin(0,0,0,0),
                          legend.box.margin = margin(-25,-30,0,0),
                          legend.title = element_blank(),
                          legend.text = element_text(size=15),
                          
                      axis.title = element_blank(),
                      axis.line = element_blank(),
                      axis.text = element_blank(),
                      axis.ticks = element_blank()
)
plot_LRP_dep <- ggplot(mean_test_data_correlated, aes(x=predicting_protein, y = masked_protein, fill = meanLRP)) +
  geom_tile() +
  scale_fill_distiller(palette = "Spectral", breaks = c(0,10,20))+
  ggtitle("Median LRP")+
  scale_y_continuous(trans = "reverse") + 
  plot_theme
  #scale_fill_gradient(low ="gray50", high ="red")

plot_CORR_dep <- ggplot(mean_test_data_correlated, aes(x=predicting_protein, y = masked_protein, fill = absCORR)) +
  geom_tile() +
  scale_fill_distiller(breaks = c(0,1.0), limits = c(0,1.0),palette = "Spectral")+
  ggtitle("Peason's r")+
  scale_y_continuous(trans = "reverse") + 
  plot_theme

plot_GENIE3_dep <- ggplot(mean_test_data_correlated, aes(x=predicting_protein, y = masked_protein, fill = genie3)) +
  geom_tile() +
  scale_fill_distiller(breaks = c(0,1.0), limits = c(0,1.0),palette = "Spectral")+
  ggtitle("GENIE3")+
  scale_y_continuous(trans = "reverse") + 
  plot_theme
  #scale_fill_gradient(low ="gray50", high ="red")

  

#######################
example_set <- mean_test_data 
example_set$interaction[example_set$predicting_protein==example_set$masked_protein] <- 1  
example_heatmap <- ggplot(example_set, aes(x=predicting_protein, y = masked_protein, fill = interaction)) +
                            geom_tile() +
                            scale_fill_distiller(breaks = c(0,1), palette = "Spectral")+
  ggtitle("Ground truth")+
  scale_y_continuous(trans = "reverse") + 
  plot_theme

adj_mat <- example_set %>% dplyr::select(predicting_protein, masked_protein, interaction) %>% distinct() %>% 
  pivot_wider(names_from = masked_protein, values_from = interaction) 

adj_matrix <- adj_mat[,-1] %>% as.matrix()
rownames(adj_matrix) <- adj_mat$predicting_protein

example_graph <- graph_from_adjacency_matrix(adj_matrix, weighted=T, mode="directed") %>% simplify()
E(example_graph)$weight <- 0.01
l <- layout_nicely(example_graph)


plot(example_graph, layout = l, main = "Network", main.cex= 30, vertex.label= NA, vertex.label.cex = 0.8, vertex.size = 2, vertex.color = "black", arrow.width = 0.01, 
       edge.arrow.size = 0.05, edge.width=0.1,edge.color = "black", rescale = T)
######################################
library(pROC)
roc_LRP_dep <- roc(mean_test_data_correlated$interaction, mean_test_data_correlated$meanLRP,   plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE)
roc_CORR_dep <- roc(mean_test_data_correlated$interaction, mean_test_data_correlated$absCORR,  plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE)
roc_GENIE_dep <- roc(mean_test_data_correlated$interaction, mean_test_data_correlated$genie3,  plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE)

##################################################################
fontsize = 2.5
position = 0.7
library(gridExtra)
library(ggplotify)
library(grid)
library("cowplot")

img2grob <- function(x) as.grob(expression(x))
png('./figures/one_network_genie.png', width = 3000, height = 1400)

plot_grid(
example_heatmap,
plot_LRP_dep,
plot_CORR_dep,
plot_GENIE3_dep,
as.grob(expression((plot(example_graph, layout = l, main = "Graph", main.cex= 20, vertex.label= NA, vertex.label.cex = 0.8, vertex.label.dist=0.5, vertex.size = 2, vertex.color = "black", 
                         arrow.width = 0.01, 
     edge.arrow.size = 0.05, edge.width=0.1,edge.color = "black", rescale = T)))),

as.grob(expression(plot.roc(roc_LRP_dep,  auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, print.auc.cex=fontsize, print.auc.x=position,cex.lab = 1.5))),
as.grob(expression(plot.roc(roc_CORR_dep,  auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, print.auc.cex=fontsize, print.auc.x=position, cex.lab = 1.5))),
as.grob(expression(plot.roc(roc_GENIE_dep,  auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, print.auc.cex=fontsize, print.auc.x=position, cex.lab = 1.5))),
nrow=2,
labels = c("A", "C", "E", "G", "B", "D", "F", "H"),
label_size = 20
)

dev.off()



#############################

#par(cex.axis=1.8) # size of axis text of roc plots
for_other_script <- plot_grid(
  example_heatmap,
  plot_LRP_dep,
  plot_CORR_dep,
  plot_GENIE3_dep,
  as.grob(expression((plot(example_graph, layout = l, vertex.label= NA, vertex.label.cex = 0.8, vertex.label.dist=0.5, vertex.size = 2, vertex.color = "black", 
                           arrow.width = 0.01, 
                           edge.arrow.size = 0.05, edge.width=0.1,edge.color = "black", rescale = T)))),
 
  as.grob(expression(plot.roc(roc_LRP_dep,  auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, print.auc.cex=fontsize, print.auc.x=position, yaxp  = c(0, 1, 1),xaxp  = c(0, 1, 1), cex.axis = 2.0,cex.lab = 2.3))),
  as.grob(expression(plot.roc(roc_CORR_dep,  auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, print.auc.cex=fontsize, print.auc.x=position, yaxp  = c(0, 1, 1),xaxp  = c(0, 1, 1), cex.axis =2.0,cex.lab = 2.3))),
  as.grob(expression(plot.roc(roc_GENIE_dep,  auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, print.auc.cex=fontsize, print.auc.x=position, yaxp  = c(0, 1, 1),xaxp  = c(0, 1, 1), cex.axis =2.0,cex.lab = 2.3))),
  nrow=2,
  labels = c("A", "C", "E", "G", "B", "D", "F", "H"),
  label_size = 35,
  label_fontface = 'plain'
)

for_other_script

saveRDS(for_other_script, file = "./figures/part1.rds")
#################
roc(mean_test_data_correlated$interaction, mean_test_data_correlated$meanLRP,   plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, ci=T)
roc(mean_test_data_correlated$interaction, mean_test_data_correlated$absCORR,  plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE,ci=T)
roc(mean_test_data_correlated$interaction, mean_test_data_correlated$genie3,   plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, ci=T)

