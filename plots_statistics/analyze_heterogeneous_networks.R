setwd('/mnt/scratch2/mlprot/')
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
library(stats)

setwd('/mnt/scratch2/mlprot/single_sample_exp/plots_statistics/use_data')
mode_ <- 'full'
depth <- '4'
mfactor <-  '0.01' 

textsize = 30
axistext=30
#LRP_data <- fread(paste('/mnt/scratch2/mlprot/single_sample_exp//plots_statistics/use_data/rawdata_y_value_new_', mode_,'_',depth,'_','.csv', sep = ""),
#                   colClasses = c('numeric', 'numeric', 'numeric', 'numeric','numeric','character','character','character')) %>%
#  filter(factor %in% mfactor)

LRP_data_raw <- fread(paste('/mnt/scratch2/mlprot/single_sample_exp/plots_statistics/use_data/rawdata_y_value_new_', mode_,'_',depth,'_',mfactor,'.csv', sep = ""),
                  colClasses = c('numeric', 'numeric', 'numeric', 'numeric','numeric','character','character','character')) %>%
  filter(factor %in% mfactor) %>%dplyr::select(-c(mode, depth, factor, yvalue))

LRP_data_raw$LRP = abs(LRP_data_raw$LRP)
LRP_data_d <- LRP_data_raw
colnames(LRP_data_d) <- c("dLRP", "predicting_protein", "masked_protein", "case")
LRP_data_t <- LRP_data_raw %>% dplyr::select(predicting_protein, masked_protein, case, LRP)
colnames(LRP_data_t) <- c("masked_protein", "predicting_protein", "case", "tLRP")

LRP_data <- left_join(LRP_data_d, LRP_data_t, by = c("predicting_protein", "masked_protein", "case")) %>% mutate("LRP" = 0.5*(dLRP+tLRP))

#specifiy number of samples per row
numberpresample = 15 

#add group variable and select cases for display
LRP_data_sel  <- LRP_data %>%
  dplyr::group_by(predicting_protein, masked_protein) %>%
  mutate("sample_group" = case %/%500+1) %>%
  filter(case %in% c(seq(numberpresample), seq(numberpresample)+500, seq(numberpresample)+1000, seq(numberpresample)+1500))

# build log LRP for better visibility
log_LRP_sel <- LRP_data_sel %>% dplyr::select(predicting_protein, masked_protein, LRP, case, sample_group) %>% 
  group_by(case) %>%
  mutate("LRP" = LRP, "group_case" = case%%500)

showcase <- log_LRP_sel%>% filter(group_case ==1) %>%
  mutate("LRP" = ifelse(predicting_protein <= sample_group*8 & predicting_protein >= sample_group*8-7 & 
                              masked_protein <= sample_group*8 & masked_protein >= sample_group*8-7,max(log_LRP_sel$LRP)*0.9,0))

showcase$group_case = 0
log_LRP_sel_combined <- rbind(log_LRP_sel, showcase) %>% dplyr::arrange(predicting_protein, masked_protein)

single_sample_plot <- ggplot(log_LRP_sel_combined, aes(predicting_protein, y = masked_protein, fill = LRP)) + 
  ggtitle('K') +
  geom_tile() + 
  scale_y_continuous(trans = "reverse") + 
  facet_grid(sample_group ~ group_case) +
  theme_bw() +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(),
    strip.text.y = element_text(size=axistext),
    axis.title = element_text(size=textsize, face = 'plain'),
    #title = element_text(size = 30, hjust = -1),
    plot.title = element_text(size = 35, hjust = -0.02, face='plain'),
    legend.title = element_text(size=textsize),
    legend.text = element_text(size = axistext),

    axis.text = element_text(size=axistext)
  ) + 

  xlab('Source protein') +
  ylab('Target protein') +
  #scale_fill_distiller(palette = "RdGy", direction = -1)+
  scale_fill_distiller(palette = "Spectral")+
  scale_x_continuous(breaks=c(1,20)) 


png('/mnt/scratch2/mlprot/single_sample_exp/plots_statistics/figures/heatmaps.png', width = 3000, height = 1000)
single_sample_plot
dev.off()

#########################################################################################################
#boxplots

grouped_LRP_data <- LRP_data %>% mutate("sample_group" = as.numeric(case) %/%500+1, "pp_group" = as.numeric(predicting_protein) %/%8+1, "mp_group" = as.numeric(masked_protein) %/%8+1)
aggregated_LRP_data <- grouped_LRP_data %>% group_by(case, pp_group, mp_group, sample_group) %>%
  dplyr::summarize(meanLRP = (mean(LRP))) %>% mutate("grid_" = 10*pp_group + mp_group)

aggregated_LRP_data$pp_group <- as.factor(aggregated_LRP_data$pp_group)
aggregated_LRP_data$mp_group <- as.factor(aggregated_LRP_data$mp_group)

levels(aggregated_LRP_data$pp_group) <- c("Pr 1-8", "Pr 9-16","Pr 17-24","Pr 25-32")
levels(aggregated_LRP_data$mp_group) <-c("Proteins 1-8", "Proteins 9-16","Proteins 17-24","Proteins 25-32") 


boxplot <- ggplot(aggregated_LRP_data, aes(x =sample_group, y=meanLRP, group = sample_group)) + 
  geom_boxplot() + 
  facet_grid(pp_group ~ mp_group)+
  ggtitle('L') +
  xlab("Interaction group")+
  ylab("LRP")+
  theme_bw() +
  theme(
    axis.title = element_text(size=textsize),
    axis.text = element_text(size=axistext),
    #title = element_text(size = 30),
    plot.title = element_text(size = 35, hjust = -0.04, face='plain'),
    strip.background = element_blank(),
    strip.text.x = element_text(size= 0.9*axistext),
    strip.text.y = element_text(size= 0.9* axistext)
    ) 

  

png(paste('/mnt/scratch2/mlprot/single_sample_exp/plots_statistics/figures/art_box_plot', '.png'), width = 1000, height = 800)
boxplot
dev.off()

############
library(Rtsne)
LRP_data_wide <- LRP_data %>% mutate("grid_" = 100*as.numeric(predicting_protein) + 1*as.numeric(masked_protein),"sample_group" = as.numeric(case) %/%500+1) %>%
  dplyr::select(grid_, LRP, case, sample_group) %>%
  pivot_wider(names_from ="grid_", values_from=LRP)

normalize <- function(x) {
  x <- as.numeric(x)
  (x-mean(x))/sd(x)
  }

LRP_data_matrix <- LRP_data_wide #apply(LRP_data_wide[,-c(1,2)],1,normalize) 

tsne<- Rtsne(as.matrix(LRP_data_matrix), check_duplicates = F)
tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2], sample_group = LRP_data_wide$sample_group)

art_tsne <- ggplot(tsne_plot, aes(x=x, y=y, color = as.factor(sample_group))) + geom_point() + 
  ggtitle('M') + 
  labs(color = "Interaction group") + 
  theme_bw()+
  theme(
    axis.title = element_text(size=textsize),
    #title = element_text(size = 30, hjust = -0.01),
    plot.title = element_text(size = 35, hjust = -0.04, face='plain'),
    legend.title = element_text(size=textsize),
    legend.text = element_text(size=textsize),
    axis.text = element_text(size=axistext)
  ) +
  guides(colour = guide_legend(override.aes = list(size=2.5)))

png(paste('/mnt/scratch2/mlprot/single_sample_exp/plots_statistics/figures/art_tsne', '.png'), width = 1000, height = 800)
art_tsne
dev.off()
########################################################
#combine plots
png('/mnt/scratch2/mlprot/single_sample_exp/plots_statistics/figures/Figure_2.png', height = 1000, width = 1400)

library(gridExtra)

lay <- rbind(c(1,1,1,1,1,1,1),
             c(1,1,1,1,1,1,1),
             c(1,1,1,1,1,1,1),
             c(1,1,1,1,1,1,1),
             #c(4,4,4,4,4,4),
             c(2,2,2,3,3,3,3),
             c(2,2,2,3,3,3,3),
             c(2,2,2,3,3,3,3),
             c(2,2,2,3,3,3,3))
grid.arrange(single_sample_plot, boxplot,art_tsne, layout_matrix = lay)

dev.off()

#########################################################
#ROC analysis

LRP_data_struc <- LRP_data %>% dplyr::mutate("sample_group" = as.numeric(case)%/%500+1) %>% 
  dplyr::mutate("ground_truth" = ifelse(predicting_protein <= sample_group*8 & predicting_protein >= sample_group*8-7 & 
                                          masked_protein <= sample_group*8 & masked_protein >= sample_group*8-7,1,0)) %>%
  filter(!(predicting_protein == masked_protein))

rocplot <- roc(LRP_data_struc$ground_truth, LRP_data_struc$LRP,  ci=TRUE, plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE)

library(pROC)
png(paste('/mnt/scratch2/mlprot/single_sample_exp/plots_statistics/figures/single_roc_plot', '.png'), width = 1200, height = 1200)
plot.roc(rocplot, ci=TRUE,  auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE, print.auc.cex=4, print.auc.x=0.7)
dev.off()


spec <- LRP_data_struc %>% 
  filter((predicting_protein%/%8 == masked_protein %/% 8))

rocplot <- roc(spec$ground_truth, spec$LRP,  ci=TRUE, plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE)

library(gridExtra)

############################################################################

pdf('/mnt/scratch2/mlprot/single_sample_exp/plots_statistics/figures/Figure_1_2.pdf', height = 2000, width = 2000)


lay <- rbind(c(4,4,4,4,4,4,4),
             c(4,4,4,4,4,4,4),
             c(4,4,4,4,4,4,4),
             c(4,4,4,4,4,4,4),
             c(4,4,4,4,4,4,4),
             c(4,4,4,4,4,4,4),
             c(1,1,1,1,1,1,1),
             c(1,1,1,1,1,1,1),
             c(1,1,1,1,1,1,1),
             c(1,1,1,1,1,1,1),
             #c(4,4,4,4,4,4),
             c(2,2,2,3,3,3,3),
             c(2,2,2,3,3,3,3),
             c(2,2,2,3,3,3,3),
             c(2,2,2,3,3,3,3))
grid.arrange(single_sample_plot, boxplot,art_tsne, for_other_script, layout_matrix = lay)

dev.off()


roc(LRP_data_struc$ground_truth, LRP_data_struc$LRP,  ci=TRUE, plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE)
roc(spec$ground_truth, spec$LRP,  ci=TRUE, plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE, print.auc=TRUE)
########################################

data_now <- LRP_data %>% mutate("sample_group" = as.numeric(case) %/%500+1, "pp_group" = as.numeric(predicting_protein) %/%8+1, "mp_group" = as.numeric(masked_protein) %/%8+1)

data_now2 <- data_now %>% filter(pp_group == 1, mp_group ==1) %>% filter(predicting_protein != masked_protein)


ggplot(data_now2, aes(x = log(1+LRP), fill = as.factor(sample_group))) +
  geom_density(alpha = 0.5) 

