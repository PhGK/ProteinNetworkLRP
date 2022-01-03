library(ggplot2)
library(stringr)
library(magrittr)
library(tidyr)
library(dplyr)
library(parallel)
library(DescTools)
library(gplots)
library(pbmcapply)
library(data.table)
library(ComplexHeatmap)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
numCores <- detectCores()
RAWPATH = '../results/artificial/homogeneous/raw_data/'


filenames <- list.files(RAWPATH)
all_data <-rbindlist(lapply(filenames, function(filename) fread(paste0(RAWPATH, filename)))) %>%
  dplyr::select(-V1) %>%
  dplyr::mutate(predicting_protein = as.numeric(predicting_protein), masked_protein = as.numeric(masked_protein)) %>%
  dplyr::group_by(predicting_protein, masked_protein) %>%
  dplyr::summarize(med = median(abs(LRP))) %>%
  pivot_wider(names_from=masked_protein, values_from=med)

mat = all_data %>%ungroup() %>% dplyr::select(-predicting_protein) %>% as.matrix()
rownames(mat) <- all_data$predicting_protein
mat = mat[order(as.numeric(rownames(mat))), order(as.numeric(colnames(mat)))]

Heatmap(log(1+mat), cluster_rows = F, cluster_columns = F)

