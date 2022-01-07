library(igraph)
library(magrittr)
library(data.table)
library(lionessR)
library(ggplot2)
library(SummarizedExperiment)
library(stringr)
library(dplyr)
library(parallel)
library(tidyr)
library(pROC)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
read.csv('../data/')
trainset <- fread('../data/artficial_heterogeneous_train.csv')[-1,-1]
testset <- fread('../data/artficial_heterogeneous_test.csv')[-1,-1]


subset <- trainset[1:2000,] %>% as.matrix() %>% t()
result <- lioness(subset)

values <- assay(result, "lioness") %>% data.frame() %>%.[,1801:1802] %>% abs()

meanvalues <- apply(values,1,mean) %>% data.frame()
meanvalues['id'] <- rownames(meanvalues)

sepvalues <- tidyr::separate(meanvalues, col = id, sep = "_", into=c('a', 'b'))

sepvalues['c'] <- as.numeric(gsub("^.{1}", "", sepvalues$a))
sepvalues['d'] <- as.numeric(gsub("^.{1}", "", sepvalues$b))

ggplot(sepvalues, aes(c, d, fill= .)) + 
  geom_tile()
##########################################################

infer <- function(i){
  print(i)
  sample <- testset[i,]
  inferset <- rbind(sample, trainset)
  infermatrix <- inferset %>% as.matrix() %>% t()
  values <- lioness(infermatrix) %>% assay("lioness") %>% data.frame() %>%.[,1]
  values
}

results <- sapply(seq(1000), infer) %>% cbind("c"=sepvalues$c, "d"=sepvalues$d, .) %>% data.frame()

write.table(results,'../plots_statistics/figures/lioness_result.csv', sep=",", row.names=F)
ggplot(results, aes(c, d, fill= V700)) + 
  geom_tile()

#####################################

results <- fread('~/PycharmProjects/Proteomics/data/lioness_result.csv')
colnames(results)[3:2002] <- 1:2000
#results$sample_id <- as.numeric(rownames(results))

results_long <- pivot_longer(results, cols = !c("c", "d"), names_to = "sample", values_to = "inf_interaction") 
results_long$inf_interaction <- abs(results_long$inf_interaction)
results_long$sample <- as.numeric(results_long$sample)
results_long$interaction_group <- (results_long$sample-1) %/% 500+1


results_long_struc <- results_long %>% 
  dplyr::mutate("ground_truth" = ifelse(c <= interaction_group*8 & c >= interaction_group*8-7 & 
                                          d <= interaction_group*8 & d >= interaction_group*8-7,1,0)) %>%
  filter(!(c == d))

rocplot <- roc(results_long_struc$ground_truth, results_long_struc$inf_interaction,  ci=TRUE, plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE)


spec <- results_long_struc %>% 
  filter((c-1)%/%8 == (d-1)%/%8)


rocplot <- roc(spec$ground_truth, spec$inf_interaction,  ci=TRUE, plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE)


png(paste('/mnt/scratch2/mlprot/single_sample_exp/plots_statistics/figures/single_roc_plot', '.png'), width = 1200, height = 1200)
plot.roc(rocplot, ci=TRUE,  auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, print.auc.cex=4, print.auc.x=0.7)

dev.off()


################

f <- results_long_struc %>% mutate("c_group" = (as.numeric(c)-1) %/%8+1, "d_group" = (as.numeric(d)-1) %/%8+1)

f$c_group<- as.factor(f$c_group)
f$d_group <- as.factor(f$d_group)

aggregated_data <- f %>% group_by(sample, c_group, d_group, interaction_group) %>%
  dplyr::summarize(mean_ = (mean(inf_interaction))) %>% mutate("grid_" = 10*c_group + d_group)

boxplot <- ggplot(aggregated_data, aes(x =interaction_group, y=mean_, group = interaction_group)) + 
  geom_boxplot() + 
  facet_grid(c_group ~ d_group, scales = "free")+
  ggtitle('B') +
  xlab("Interaction group")+
  ylab("LRP")+
  theme_bw() +
  theme(
    axis.title = element_text(size=10),
    axis.text = element_text(size=20),
    title = element_text(size = 30),
    strip.background = element_blank(),
    strip.text.x = element_text(size=0.8 * 20),
    strip.text.y = element_text(size=0.8 * 20)
  ) 

boxplot
################################################
data_now <- f %>% filter(c_group==3, d_group==3)
ggplot(data_now, aes(x = log(1+inf_interaction), fill = as.factor(interaction_group))) +
  geom_density(alpha = 0.5) 

######################################  
#plot heatmaps
meandata <- results_long_struc %>% group_by(c,d, interaction_group) %>% summarize(meanvalue = log(1+mean(inf_interaction)))

ggplot(meandata, aes(c, d, fill= meanvalue)) +
  facet_wrap(~interaction_group)+
  geom_tile()


meandata <- results_long_struc %>% filter(interaction_group == 3, sample %in% seq(1010, 1025)) %>% mutate("logint" = log(1+inf_interaction))

ggplot(meandata, aes(c, d, fill= inf_interaction)) +
  facet_wrap(~sample)+
  geom_tile()
