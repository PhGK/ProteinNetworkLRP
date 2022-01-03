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
library(plyr)
############
####plots
############
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


test_data <- fread('../results/LRP/use_data/all_data.csv')
test_data$LRP <- test_data$LRP %>% abs()
##############################################################################################################################
#load ground truth data
# --> choose only cases with high correlation
##############################################################################################################################
#load ground truth data
protein_data <- read.csv('../data/tcpa_data_051017.csv', check.names = F) %>% 
  pivot_longer(!c(ID,Cancer_Type), names_to = "proteins", values_to = "expression") %>% 
  dplyr::select(-Cancer_Type)


for_correlation <- test_data %>% dplyr::select(ID, y, y_pred) %>%
  unique() %>%
  group_by(ID)

correlation <- ddply(for_correlation, "ID", summarize, "corr" = cor(y, y_pred))
highcorrelation <- correlation %>% filter(corr>0.01)

rcorr(for_correlation$y, for_correlation$y_pred)

case_numbers_old <- test_data %>% group_by(ID) %>% dplyr::summarize("Cancer_Type" = max(Cancer_Type))
summary(as.factor(case_numbers_old$Cancer_Type))

LRP_data_filtered <- test_data %>% filter(ID %in% highcorrelation$ID)

########################################
LRP_sym_dir <- LRP_data_filtered %>% dplyr::select(LRP, predicting_protein, masked_protein, ID)
LRP_sym_trans <- LRP_sym_dir
colnames(LRP_sym_trans) <- c('tLRP', 'masked_protein', 'predicting_protein', 'ID')


LRP_sym <- left_join(LRP_sym_dir, LRP_sym_trans) %>%
  dplyr::mutate("LRP" = 0.5*(LRP + tLRP))  %>%
  filter(predicting_protein > masked_protein)

mean_LRP_sym <- LRP_sym %>% 
  dplyr::group_by(predicting_protein, masked_protein) %>%
  dplyr::summarize('LRP' = median(LRP)) 


##################################################
# read ractome matrix
adj_react <- read.delim('int_react_147_060418.csv', header = F)  
colnames(adj_react) <- c("masked_protein", as.character(adj_react$V1))
adj_react_matrix <- as.matrix(adj_react[,-1])
rownames(adj_react_matrix) <- as.character(adj_react$masked_protein)

# order names alphanumerically
adj_react_matrix_ordered <- adj_react_matrix[str_order(rownames(adj_react_matrix)), str_order(colnames(adj_react_matrix))]
adj_react_long <- pivot_longer(adj_react,cols=!masked_protein, names_to = "predicting_protein", values_to = "edge")
adj_react_long$predicting_protein <- as.character(adj_react_long$predicting_protein)
adj_react_long$masked_protein <- as.character(adj_react_long$masked_protein)

adj_react_long_sym <- adj_react_long %>% filter((predicting_protein) > (masked_protein))
###################################################
full_frame <- left_join(adj_react_long_sym, mean_LRP_sym, by =c("masked_protein", "predicting_protein")) 

#######################################
#prepare correlation matrix
protein_data <- read.csv('tcpa_data_051017.csv', check.names=F) %>% 
  pivot_longer(!c(ID,Cancer_Type), names_to = "proteins", values_to = "expression") %>%
  filter(proteins %in% (adj_react_long$predicting_protein %>% unique() %>% unlist())) %>%
  filter(ID %in% (LRP_data_filtered$case %>% unique %>% unlist())) %>%
  pivot_wider(names_from =proteins, values_from=expression)

for_correlation <- protein_data[,-c(1:2)]
for_correlation_ordered <- for_correlation[,str_order(colnames(for_correlation))]
corr_matrix <- cor(for_correlation_ordered) %>% as.matrix() %>% abs()

corr_data_long <- as.data.frame(corr_matrix) %>% cbind("predicting_protein" = colnames(corr_matrix), .)%>% 
  pivot_longer(!predicting_protein, names_to = "masked_protein", values_to = "correlation") %>%
  filter(as.character(predicting_protein) > as.character(masked_protein))

full_frame_LRP_corr <- left_join(full_frame, corr_data_long)


#########
#prepare GENIE matrix
library(GENIE3)
protein_data <- read.csv('tcpa_data_051017.csv', check.names=F) %>% 
  pivot_longer(!c(ID,Cancer_Type), names_to = "proteins", values_to = "expression") %>%
  filter(proteins %in% (adj_react_long$predicting_protein %>% unique() %>% unlist())) %>%
  filter(ID %in% (LRP_data_filtered$case %>% unique %>% unlist())) %>%
  pivot_wider(names_from =proteins, values_from=expression)

for_correlation <- protein_data[,-c(1:2)]
for_correlation_ordered <- for_correlation[,str_order(colnames(for_correlation))] %>% t()
genie_matrix <- GENIE3(for_correlation_ordered, verbose=T, nCores=10)


genie_matrix_sym <- genie_matrix + t(genie_matrix)
genie_data_long <- as.data.frame(genie_matrix_sym) %>% cbind("predicting_protein" = colnames(genie_matrix_sym), .)%>% 
  pivot_longer(!predicting_protein, names_to = "masked_protein", values_to = "genie3") %>%
  filter(as.character(predicting_protein) > as.character(masked_protein))



k = 100
full_frame_all <- left_join(full_frame_LRP_corr, genie_data_long)

full_frame_subset <- full_frame_all %>% dplyr::arrange(desc(correlation)) %>% .[1:k,]


#####
#stat hypergeometric test
library(stats)
ncorrect <- full_frame_subset$edge %>% sum()
ncorrectall <- adj_react_long_sym$edge %>% sum()
nfalseall <- (1-adj_react_long_sym$edge) %>% sum()


phyper(ncorrect-1, ncorrectall, nfalseall, k, lower.tail = F, log.p = FALSE)
ncorrectall
nfalseall
ncorrectall+nfalseall
ncorrect

